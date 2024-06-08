import asyncio
import numpy as np
import time
import logging
import argparse

from enum import Flag, auto
from typing import Iterable, List, Any
from importlib.metadata import version

from aerospike_vector_search import types as vectorTypes, Client as vectorSyncClient
from aerospike_vector_search.aio import AdminClient as vectorASyncAdminClient, Client as vectorASyncClient
from aerospike_vector_search.shared.proto_generated.types_pb2_grpc import grpc  as vectorResultCodes

from baseaerospike import BaseAerospike, _distanceNameToAerospikeType as DistanceMaps
from datasets import DATASETS, load_and_transform_dataset

logger = logging.getLogger(__name__)

aerospikeIdxNames : list = []
  
class OperationActions(Flag):    
    POPULATION = auto()
    QUERY = auto()
    POPQUERY = POPULATION | QUERY
    
class Aerospike(BaseAerospike):
    
    @staticmethod
    def parse_arguments_population(parser: argparse.ArgumentParser) -> None:
        '''
        Adds the arguments required to populate an index. 
        '''
        parser.add_argument(
            '-d', "--dataset",
            metavar="DS",
            help="the dataset to load training points from",
            default="glove-100-angular",
            choices=DATASETS.keys(),
        )
        parser.add_argument(
            '-c', "--concurrency",
            metavar="N",
            type=int,
            help='''
    The maximum number of concurrent taks used to population the index.
    Values are:
        - < 0 -- All records are upserted, concurrently, and the app will only wait for the upsert completion before waiting for index completion.
        -   0 -- Disable Population (Index is still created and Wait for Idx Completion still performed)
        -   1 -- One record is upserted at a time (sync)
        -   > 1 -- The number of records upserted, concurrently (async), before the app waits for the upserts to complete.
    ''',
            default=500,
        )        
        parser.add_argument(
            "--idxdrop",        
            help="If the Vector Index existance, it will be dropped. Otherwise is is updated.",
            default=False,
            action='store_true'
        )        
        parser.add_argument(
            "--idxnowait",        
            help="Waiting for Index Complation is disabled.",
            default=False,
            action='store_true'
        )
        parser.add_argument(
            '-E', "--exhaustedevt",
            metavar="EVT",
            type=int,
            help='''
    This determines how the Resource Exhausted event is handled. 
    Values are:
        -  < 0 -- All population events are stopped and will not resume until the Idx queue is cleared
                    (wait for idx completion).
        -    0 -- Disable event handling (just re-throws the exception)
        - >= 1 -- All population events are stopped and this is the number of seconds to wait before re-starting the population.
                    This needs to be a large enough number to allow the Idx queue to somewhat clear.
    ''',
            default=-1,
        )
        BaseAerospike.parse_arguments(parser)
       
    def __init__(self, runtimeArgs: argparse.Namespace, actions: OperationActions):
        
        super().__init__(runtimeArgs, logger)
        
        self._actions = actions
        self._datasetname = runtimeArgs.dataset
        self._dimensions = None
        self._trainarray = None
        self._dataset = None
        
        if OperationActions.POPULATION in actions:
            self._idx_drop = runtimeArgs.idxdrop
            self._concurrency = runtimeArgs.concurrency
            self._idx_nowait = runtimeArgs.idxnowait
            self._idx_resource_event = runtimeArgs.exhaustedevt
            self._idx_resource_cnt = 0
        
    def __enter__(self):
        return self
 
    def __exit__(self, *args):
        super().done()
        
    async def get_dataset(self) -> None:
        
        self.print_log(f'get_dataset: {self}')
        
        self._trainarray, query, distance, self._dataset, self._dimensions = load_and_transform_dataset(self._datasetname)

        if self._idx_distance is None or not self._idx_distance:
            self._idx_distance = DistanceMaps.get(distance.lower())
        
        if self._idx_distance is None or not self._idx_distance:
             raise ValueError(f"Distance Map '{distance}' was not found.")
         
        if self._paramsetname:
            if self._idx_distance.casefold() == distance.casefold():
                setNameType = self._idx_distance
            else:
                setNameType = f'{distance}_{self._idx_distance}'
            self._setName = f'{self._setName}_{setNameType}_{self._dimensions}_{self._idx_hnswparams.m}_{self._idx_hnswparams.ef_construction}_{self._idx_hnswparams.ef}'
            self._idx_name = f'{self._setName}_Idx'
        
        self.print_log(f'get_dataset Exit: {self}, Train Array: {len(self._trainarray)}, Distance: {distance}, Dimensions: {self._dimensions}')
                
    async def drop_index(self, adminClient: vectorASyncAdminClient) -> None:
        self.print_log(f'Dropping Index {self._namespace}.{self._idx_name}')
        s = time.time()
        await adminClient.index_drop(namespace=self._namespace,
                                            name=self._idx_name)        
        existingIndexes = await adminClient.index_list()    
        t = time.time()
        print('\n')
        self.print_log(f'Drop Index Time (sec) = {t - s}')        
        
    async def create_index(self, adminClient: vectorASyncAdminClient) -> None:
        global aerospikeIdxNames
        self.print_log(f'Creating Index {self._namespace}.{self._idx_name}')        
        s = time.time()
        await adminClient.index_create(namespace=self._namespace,
                                                name=self._idx_name,
                                                sets=self._setName,
                                                vector_field=self._idx_binName,
                                                dimensions=self._dimensions,
                                                index_params= self._idx_hnswparams,
                                                vector_distance_metric=self._idx_distance
                                                )
        t = time.time()
        self.print_log(f'Index Creation Time (sec) = {t - s}')        
        aerospikeIdxNames.append(self._idx_name)

    async def _put_wait_completion_handler(self, key: int, embedding, i: int, client: vectorASyncClient, logLevel: int) -> None:
        s = time.time()
        await client.wait_for_index_completion(namespace=self._namespace,
                                                name=self._idx_name)            
        t = time.time()
        if logLevel == logging.WARNING:
            self.print_log(msg=f"Index Completed Time (sec) = {t - s}, Going to Reissue Puts for Idx: {self._namespace}.{self._setName}.{self._idx_name}",
                                logLevel=logging.WARNING)
        else:
            logger.debug(msg=f"Index Completed Time (sec) = {t - s}, Going to Reissue Puts for Count: {i}, Key: {key}, Idx: {self._namespace}.{self._setName}.{self._idx_name}")
        await self.put_vector(key, embedding, i, client, True)
        self._puasePuts = False
        if logLevel == logging.WARNING:
            self.print_log(msg=f"Resuming population for Idx: {self._namespace}.{self._setName}.{self._idx_name}",
                                logLevel=logging.WARNING)
        else:
            logger.debug(msg=f"Resuming population for Count: {i}, Key: {key}, Idx: {self._namespace}.{self._setName}.{self._idx_name}")
            
    async def _put_wait_sleep_handler(self, key: int, embedding, i: int, client: vectorASyncClient, logLevel: int) -> None:
        self._idx_resource_cnt += 1
        if logLevel == logging.WARNING:
            self.print_log(msg=f"Resource Exhausted Going to Sleep {self._idx_resource_event}: {self._namespace}.{self._setName}.{self._idx_name}",
                                logLevel=logging.WARNING)
        else:
            logger.debug(msg=f"Resource Exhausted Sleep {self._idx_resource_event}, Going to Reissue Puts for Count: {i}, Key: {key}, Idx: {self._namespace}.{self._setName}.{self._idx_name}")
        await asyncio.sleep(self._idx_resource_event)
        
        await self.put_vector(key, embedding, i, client, True)
        self._idx_resource_cnt -= 1        
        if(self._idx_resource_cnt <= 0):
            self._puasePuts = False
            
            if logLevel == logging.WARNING:
                self.print_log(msg=f"Resuming population for Idx: {self._namespace}.{self._setName}.{self._idx_name}",
                                    logLevel=logging.WARNING)
            else:
                logger.debug(msg=f"Resuming population for Count: {i}, Key: {key}, Idx: {self._namespace}.{self._setName}.{self._idx_name}")

    async def put_vector(self, key: int, embedding, i: int, client: vectorASyncClient, retry: bool = False) -> None:
        try:
            try:
                await client.upsert(namespace=self._namespace,
                                    set_name=self._setName,
                                    key=key,
                                    record_data={
                                        self._idx_binName:embedding.tolist()
                                    }
                )        
            except vectorTypes.AVSServerError as avse:
                if self._idx_resource_event != 0 and not retry and avse.rpc_error.code() == vectorResultCodes.StatusCode.RESOURCE_EXHAUSTED:
                    logLevel = logging.DEBUG
                    if not self._puasePuts:
                        self._puasePuts = True
                        logLevel = logging.WARNING
                        self.print_log(msg=f"\nResource Exhausted on Put first encounter on Count: {i}, Key: {key}, Idx: {self._namespace}.{self._setName}.{self._idx_name}. Going to Pause Population and Wait for Idx Completion...",
                                            logLevel=logging.WARNING)
                    else:
                        logger.debug(f"Resource Exhausted on Put on Count: {i}, Key: {key}, Idx: {self._namespace}.{self._setName}.{self._idx_name}. Going to Pause Population and Wait for Idx Completion...")
                    
                    if self._idx_resource_event < 0:
                        await self._put_wait_completion_handler(key, embedding, i, client, logLevel)
                    else:
                        await self._put_wait_sleep_handler(key, embedding, i, client, logLevel)
                else:
                    raise avse
        except Exception as e:
            print(f'\n** Count: {i} Key: {key} Exception: "{e}" **\r\n')
            logger.exception(f"Put Failure on Count: {i}, Key: {key}, Idx: {self._namespace}.{self._setName}.{self._idx_name}, Retry: {retry}")
            self.flush_log()
            raise e
        
    async def populate(self) -> None:
        global aerospikeIdxNames
        
        if self._trainarray.dtype != np.float32:
            self._trainarray = self._trainarray.astype(np.float32)
    
        self.print_log(f'populate: {self} Shape: {self._trainarray.shape}')
              
        populateIdx = True
            
        async with vectorASyncAdminClient(
                seeds=vectorTypes.HostPort(host=self._host, port=self._port, is_tls=self._verifyTLS),
                listener_name=self._listern,
                is_loadbalancer=self._useloadbalancer
            ) as adminClient:

            #If exists, no sense to try creation...
            existingIndexes = await adminClient.index_list()
            if(any(index["id"]["namespace"] == self._namespace
                                    and index["id"]["name"] == self._idx_name 
                            for index in existingIndexes)):
                self.print_log(f'Index {self._namespace}.{self._idx_name} Already Exists')
                
                #since this can be an external DB (not in a container), we need to clean up from prior runs
                #if the index name is in this list, we know it was created in this run group and don't need to drop the index.
                #If it is a fresh run, this list will not contain the index and we know it needs to be dropped.
                if self._idx_name in aerospikeIdxNames:
                    self.print_log(f'Index {self._namespace}.{self._idx_name} being reused (updated)')
                elif self._idx_drop:
                    await self.drop_index(adminClient)
                    await self.create_index(adminClient)
                else:
                    self.print_log(f'Index {self._namespace}.{self._idx_name} being updated')
            else:
                await self.create_index(adminClient)
                
        if populateIdx:
            async with vectorASyncClient(seeds=vectorTypes.HostPort(host=self._host, port=self._port, is_tls=self._verifyTLS),
                                                listener_name=self._listern,
                                                is_loadbalancer=self._useloadbalancer
                            ) as client:
                if self._concurrency == 0:
                    s = time.time()
                else:
                    self._puasePuts = False
                    self.print_log(f'Populating Index {self._namespace}.{self._idx_name}')                    
                    s = time.time()
                    taskPuts = []
                    i = 0
                    #async with asyncio. as tg: #only in 3.11
                    for key, embedding in enumerate(self._trainarray):
                        if self._puasePuts:
                            loopTimes = 0
                            print('\n')
                            while (self._puasePuts):
                                if loopTimes % 30 == 0:
                                    self.print_log(f"Paused Population still waiting for Idx Completion at {loopTimes} mins!", logging.WARNING)                                
                                loopTimes += 1
                                logger.debug(f"Putting Paused {loopTimes}")
                                await asyncio.sleep(60)
                            self.print_log(f"Resuming Population at {loopTimes} mins", logging.WARNING)
                            
                        i += 1
                        if self._concurrency < 0:
                            taskPuts.append(self.put_vector(key, embedding, i, client))
                        elif self._concurrency <= 1:
                            await self.put_vector(key, embedding, i, client)                    
                        else:
                            taskPuts.append(self.put_vector(key, embedding, i, client))
                            if len(taskPuts) >= self._concurrency:
                                logger.debug(f"Waiting for Put Tasks ({len(taskPuts)}) to Complete at {i}")
                                await asyncio.gather(*taskPuts)
                                logger.debug(f"Put Tasks Completed")
                                taskPuts.clear()                                                                                    
                        print('Aerospike: Index Put Counter [%d]\r'%i, end="")
                    
                    logger.debug(f"Waiting for Put Tasks (finial {len(taskPuts)}) to Complete at {i}")                            
                    await asyncio.gather(*taskPuts)
                    t = time.time()
                    logger.info(f"All Put Tasks Completed")                
                    print('\n')
                    self.print_log(f"Index Put {i:,} Recs in {t - s} (secs), TPS: {i/(t - s):,}")
                
                if self._idx_nowait:
                    self.print_log(f"Index Population Completed")
                else:
                    #Wait for Idx to complete                                
                    self.print_log("waiting for indexing to complete")
                    w = time.time()
                    await client.wait_for_index_completion(namespace=self._namespace,
                                                            name=self._idx_name)            
                    t = time.time()
                    self.print_log(f"Index Completion Time (secs) = {t - w} TPS = {len(self._trainarray)/(t - w):,}")
                    self.print_log(f"Index Population Completion with Idx Wait (sec) = {t - s}")

    def __str__(self):
        arrayLen = None
        if self._trainarray is not None:
            arrayLen = len(self._trainarray)
        if OperationActions.POPULATION in self._actions:
            popstr = f", DropIdx: {self._idx_drop}, Concurrency: {self._concurrency}, WaitIdxCompletion: {not self._idx_nowait} Exhausted Evt: {self._idx_resource_event}"
            
        return f"Aerospike([{self.basestring()}, Actions: {self._actions}, Dimensions: {self._dimensions}, Array: {arrayLen} DS: {self._datasetname}{popstr}]"
