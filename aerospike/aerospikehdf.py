import asyncio
import numpy as np
import time
import logging
import argparse

from enum import Flag, auto
from typing import Iterable, List, Union, Any
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
        if len([x.dest for x in parser._actions if "dataset" == x.dest]) == 0:
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
            action='store_true'
        )        
        parser.add_argument(
            "--idxnowait",        
            help="Waiting for Index Complation is disabled.",
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
        parser.add_argument(
            '-m', "--maxrecs",
            metavar="RECS",
            type=int,
            help="Determines the maximum number of records to populated. a value of -1 (default) all records in the HDF dataset are populated.",
            default=-1,
        )
        BaseAerospike.parse_arguments(parser) 
    
    @staticmethod
    def parse_arguments_query(parser: argparse.ArgumentParser) -> None:
        '''
        Adds the arguments required to query an index. 
        '''
        if len([x.dest for x in parser._actions if "dataset" == x.dest]) == 0:
            parser.add_argument(
                '-d', "--dataset",
                metavar="DS",
                help="the dataset to load training points from",
                default="glove-100-angular",
                choices=DATASETS.keys(),
            )
        parser.add_argument(
            '-r', "--runs",
            metavar="RUNS",
            type=int,
            help="The number of query runs",
            default=10,
        )
        parser.add_argument(
           "--limit",
            metavar="NEEIGHBORS",
            type=int,
            help="The number of neighbors to return from the query",
            default=100,
        )
        parser.add_argument(
           "--parallel",        
            help="Query runs are ran in parallel",
            action='store_true'
        )  
        parser.add_argument(
           "--check",        
            help="Check Query Results",
            action='store_true'
        )        
        BaseAerospike.parse_arguments(parser)
    
    def __init__(self, runtimeArgs: argparse.Namespace, actions: OperationActions):
        
        super().__init__(runtimeArgs, logger)
        
        self._actions: OperationActions = actions
        self._datasetname: str = runtimeArgs.dataset
        self._dimensions = None
        self._trainarray = None
        self._queryarray = None
        self._dataset = None
        self._puasePuts : bool = False
        
        if OperationActions.POPULATION in actions:
            self._idx_drop = runtimeArgs.idxdrop
            self._concurrency = runtimeArgs.concurrency
            self._idx_nowait = runtimeArgs.idxnowait
            self._idx_resource_event = runtimeArgs.exhaustedevt
            self._idx_resource_cnt = 0
            self._idx_maxrecs = runtimeArgs.maxrecs
        
        if OperationActions.QUERY in actions:
            self._query_runs = runtimeArgs.runs
            self._query_parallel = runtimeArgs.parallel
            self._query_check = runtimeArgs.check
            self._query_limit = runtimeArgs.limit
            
    def __enter__(self):
        return self
 
    def __exit__(self, *args):
        super().shutdown()
        
    async def get_dataset(self) -> None:
        
        self.print_log(f'get_dataset: {self}')
        
        self._trainarray, self._queryarray, distance, self._dataset, self._dimensions = load_and_transform_dataset(self._datasetname)
        
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
                
        self.prometheus_status(0)
        self.print_log(f'get_dataset Exit: {self}, Train Array: {len(self._trainarray)}, Query Array: {len(self._queryarray)}, Distance: {distance}, Dimensions: {self._dimensions}')
                
    async def drop_index(self, adminClient: vectorASyncAdminClient) -> None:
        self.print_log(f'Dropping Index {self._idx_namespace}.{self._idx_name}')
        s = time.time()
        await adminClient.index_drop(namespace=self._idx_namespace,
                                            name=self._idx_name)
        t = time.time()
        self._dropidx_histogram.record(t-s, {"ns":self._idx_namespace,"idx": self._idx_name})
        print('\n')
        self.print_log(f'Drop Index Time (sec) = {t - s}')        
        
    async def create_index(self, adminClient: vectorASyncAdminClient) -> None:
        global aerospikeIdxNames
        self.print_log(f'Creating Index {self._idx_namespace}.{self._idx_name}')        
        s = time.time()
        await adminClient.index_create(namespace=self._idx_namespace,
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

    async def _wait_for_idx_completion(self, client: vectorASyncClient):
        self._waitidx_counter.add(1, {"ns":self._idx_namespace,"idx": self._idx_name})
        try:
            await client.wait_for_index_completion(namespace=self._idx_namespace,
                                                    name=self._idx_name)
        except Exception as e:
            print(f'\n**Exception: "{e}" **\r\n')
            logger.exception(f"Wait for Idx Completion Failure on Idx: {self._idx_namespace}.{self._idx_name}")
            self.flush_log()
            self._exception_counter.add(1, {"exception_type":e, "handled_by_user":False,"ns":self._idx_namespace,"idx":self._idx_name})            
            raise
        finally:
            self._waitidx_counter.add(-1, {"ns":self._idx_namespace,"idx": self._idx_name})

    async def _put_wait_completion_handler(self, key: int, embedding, i: int, client: vectorASyncClient, logLevel: int) -> None:
        s = time.time()
        await self._wait_for_idx_completion(client)                    
        t = time.time()
        if logLevel == logging.WARNING:
            self.print_log(msg=f"Index Completed Time (sec) = {t - s}, Going to Reissue Puts for Idx: {self._idx_namespace}.{self._idx_name}",
                                logLevel=logging.WARNING)
        else:
            logger.debug(msg=f"Index Completed Time (sec) = {t - s}, Going to Reissue Puts for Count: {i}, Key: {key}, Idx: {self._idx_namespace}.{self._idx_name}")
        await self.put_vector(key, embedding, i, client, True)
        self._puasePuts = False
        if logLevel == logging.WARNING:
            self.print_log(msg=f"Resuming population for Idx: {self._idx_namespace}.{self._idx_name}",
                                logLevel=logging.WARNING)
        else:
            logger.debug(msg=f"Resuming population for Count: {i}, Key: {key}, Idx: {self._idx_namespace}.{self._idx_name}")
            
    async def _put_wait_sleep_handler(self, key: int, embedding, i: int, client: vectorASyncClient, logLevel: int) -> None:
        self._idx_resource_cnt += 1
        if logLevel == logging.WARNING:
            self.print_log(msg=f"Resource Exhausted Going to Sleep {self._idx_resource_event}: {self._idx_namespace}.{self._idx_name}",
                                logLevel=logging.WARNING)
        else:
            logger.debug(msg=f"Resource Exhausted Sleep {self._idx_resource_event}, Going to Reissue Puts for Count: {i}, Key: {key}, Idx: {self._idx_namespace}.{self._idx_name}")
        await asyncio.sleep(self._idx_resource_event)
        
        await self.put_vector(key, embedding, i, client, True)
        self._idx_resource_cnt -= 1        
        if(self._idx_resource_cnt <= 0):
            self._puasePuts = False
            
            if logLevel == logging.WARNING:
                self.print_log(msg=f"Resuming population for Idx: {self._idx_namespace}.{self._idx_name}",
                                    logLevel=logging.WARNING)
            else:
                logger.debug(msg=f"Resuming population for Count: {i}, Key: {key}, Idx: {self._idx_namespace}.{self._idx_name}")

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
                self._populate_counter.add(1, {"type": "upsert","ns":self._namespace,"set":self._setName})
            except vectorTypes.AVSServerError as avse:
                if self._idx_resource_event != 0 and not retry and avse.rpc_error.code() == vectorResultCodes.StatusCode.RESOURCE_EXHAUSTED:
                    self._exception_counter.add(1, {"exception_type": "Resource Exhausted", "handled_by_user": True,"ns":self._namespace,"set":self._setName})
                    logLevel = logging.DEBUG
                    if not self._puasePuts:
                        self._puasePuts = True
                        logLevel = logging.WARNING
                        self.print_log(msg=f"\nResource Exhausted on Put first encounter on Count: {i}, Key: {key}, Idx: {self._idx_namespace}.{self._idx_name}. Going to Pause Population and Wait for Idx Completion...",
                                            logLevel=logging.WARNING)
                    else:
                        logger.debug(f"Resource Exhausted on Put on Count: {i}, Key: {key}, Idx: {self._idx_namespace}.{self._idx_name}. Going to Pause Population and Wait for Idx Completion...")
                    
                    if self._idx_resource_event < 0:
                        await self._put_wait_completion_handler(key, embedding, i, client, logLevel)                        
                    else:
                        await self._put_wait_sleep_handler(key, embedding, i, client, logLevel)
                    self._exception_counter.add(-1, {"exception_type": "Resource Exhausted", "handled_by_user": True,"ns":self._namespace,"set":self._setName})                    
                else:
                    raise
        except Exception as e:
            print(f'\n** Count: {i} Key: {key} Exception: "{e}" **\r\n')
            logger.exception(f"Put Failure on Count: {i}, Key: {key}, Idx: {self._idx_namespace}.{self._idx_name}, Retry: {retry}")
            self.flush_log()
            self._exception_counter.add(1, {"exception_type":e, "handled_by_user":False,"ns":self._namespace,"set":self._setName})
            self._puasePuts = False
            raise
        
    async def index_exist(self, adminClient: vectorASyncAdminClient) -> bool:
        existingIndexes = await adminClient.index_list()
        return any(index["id"]["namespace"] == self._idx_namespace
                    and index["id"]["name"] == self._idx_name 
                        for index in existingIndexes)
        
    async def populate(self) -> None:
        '''
        Polulates the vector index based on HDF dataset.
        '''
        global aerospikeIdxNames
        
        if self._trainarray.dtype != np.float32:
            self._trainarray = self._trainarray.astype(np.float32)
    
        self.print_log(f'populate: {self} Shape: {self._trainarray.shape}')
              
        populateIdx = True
            
        async with vectorASyncAdminClient(seeds=self._host,
                                            listener_name=self._listern,
                                            is_loadbalancer=self._useloadbalancer
            ) as adminClient:
            
            #If exists, no sense to try creation...            
            if await self.index_exist(adminClient):
                self.print_log(f'Index {self._idx_namespace}.{self._idx_name} Already Exists')
                
                #since this can be an external DB (not in a container), we need to clean up from prior runs
                #if the index name is in this list, we know it was created in this run group and don't need to drop the index.
                #If it is a fresh run, this list will not contain the index and we know it needs to be dropped.
                if self._idx_name in aerospikeIdxNames:
                    self.print_log(f'Index {self._idx_name} being reused (updated)')
                elif self._idx_drop:
                    await self.drop_index(adminClient)
                    await self.create_index(adminClient)
                else:
                    self.print_log(f'Index {self._idx_namespace}.{self._idx_name} being updated')
            else:
                await self.create_index(adminClient)
                
        if populateIdx:
            async with vectorASyncClient(seeds=self._host,
                                            listener_name=self._listern,
                                            is_loadbalancer=self._useloadbalancer
                            ) as client:
                if self._concurrency == 0 or self._idx_maxrecs == 0:
                    s = time.time()
                else:
                    trainsize = len(self._trainarray)
                    self._populate_counter.add(0, {"type": "upsert","ns":self._namespace,"set":self._setName})
                    self._populate_recs_gauge.set(trainsize, {"ns":self._namespace,"set":self._setName})            
                    self._puasePuts = False
                    self.print_log(f'Populating Index {self._idx_namespace}.{self._idx_name}')                    
                    s = time.time()
                    taskPuts = []
                    i = 1                    
                    #async with asyncio. as tg: #only in 3.11
                    for key, embedding in enumerate(self._trainarray):
                        if self._puasePuts:
                            loopTimes = 0
                            print('\n')
                            while (self._puasePuts):
                                if loopTimes % 30 == 0:
                                    self.print_log(f"Paused Population still waiting for Idx Completion at {loopTimes} mins!", logging.WARNING)                                
                                loopTimes += 1
                                self._populate_recs_gauge.set(trainsize-i,{"ns":self._namespace,"set":self._setName})
                                logger.debug(f"Putting Paused {loopTimes}")
                                await asyncio.sleep(60)
                            self.print_log(f"Resuming Population at {loopTimes} mins", logging.WARNING)
                                                    
                        if self._concurrency < 0:
                            taskPuts.append(self.put_vector(key, embedding, i, client))
                        elif self._concurrency <= 1:
                            await self.put_vector(key, embedding, i, client)
                            self._populate_recs_gauge.set(trainsize-i,{"ns":self._namespace,"set":self._setName})
                        else:
                            taskPuts.append(self.put_vector(key, embedding, i, client))
                            if len(taskPuts) >= self._concurrency:
                                logger.debug(f"Waiting for Put Tasks ({len(taskPuts)}) to Complete at {i}")
                                await asyncio.gather(*taskPuts)
                                logger.debug(f"Put Tasks Completed")
                                taskPuts.clear()
                                self._populate_recs_gauge.set(trainsize-i,{"ns":self._namespace,"set":self._setName})                        
                        print('Index Put Counter [%d]\r'%i, end="")
                        if self._idx_maxrecs >= 0 and i >= self._idx_maxrecs:
                            break
                        i += 1
                        
                    logger.debug(f"Waiting for Put Tasks (finial {len(taskPuts)}) to Complete at {i}")                            
                    await asyncio.gather(*taskPuts)                    
                    t = time.time()
                    self._populate_recs_gauge.set(0,{"ns":self._namespace,"set":self._setName})                        
                    logger.info(f"All Put Tasks ({i}) Completed")                
                    print('\n')
                    self.print_log(f"Index Put {i:,} Recs in {t - s} (secs), TPS: {i/(t - s):,}")
                
                if self._idx_nowait:
                    self.print_log(f"Index Population Completed")
                else:
                    #Wait for Idx to complete                                
                    self.print_log("waiting for indexing to complete")
                    w = time.time()
                    await self._wait_for_idx_completion(client)
                    t = time.time()
                    self.print_log(f"Index Completion Time (secs) = {t - w} TPS = {len(self._trainarray)/(t - w):,}")
                    self.print_log(f"Index Population Completion with Idx Wait (sec) = {t - s}")

    async def query(self) -> None:
        '''
        Performs a query using the query array from the HDF dataset. 
        '''
        
        self.print_log(f'Query: {self} Shape: {self._trainarray.shape}')
        
        async with vectorASyncAdminClient(seeds=self._host,
                                            listener_name=self._listern,
                                            is_loadbalancer=self._useloadbalancer
            ) as adminClient:
           
            if not await self.index_exist(adminClient):
                self.print_log(f'Query: Vector Index: {self._idx_namespace}.{self._idx_name}, not found')
                self._exception_counter.add(1, {"exception_type":"Index not found", "handled_by_user":False,"ns":self._namespace,"set":self._setName})
                raise FileNotFoundError(f"Vector Index {self._idx_namespace}.{self._idx_name} not found")

        self.print_log(f'Starting Query Runs ({self._query_runs}) on {self._idx_namespace}.{self._idx_name}')
        
        async with vectorASyncClient(seeds=self._host,
                                        listener_name=self._listern,
                                        is_loadbalancer=self._useloadbalancer
                            ) as client:
            s = time.time()
            taskPuts = []
            queries = 0
            i = 1
            self._query_counter.add(0, {"type": "Vector Search","ns":self._idx_namespace,"idx":self._idx_name})
            self._query_runs_gauge.set(self._query_runs,{"ns":self._idx_namespace,"idx":self._idx_name})
            while i <= self._query_runs:                
                if self._query_parallel:
                    taskPuts.append(self.query_run(client, i))
                else:
                    queries += await self.query_run(client, i)
                    self._query_runs_gauge.set(self._query_runs-i,{"ns":self._idx_namespace,"idx":self._idx_name})                                                
                i += 1
                
            results = await asyncio.gather(*taskPuts)
            t = time.time()
            if self._query_parallel:
                queries =  sum(results)
            self._query_runs_gauge.set(0,{"ns":self._idx_namespace,"idx":self._idx_name}) 
            print('\n')
            self.print_log(f'Finished Query Runs on {self._idx_namespace}.{self._idx_name}; Total queries {queries} in {t-s} secs, {queries/(t-s)} TPS')
                        
    async def query_run(self, client:vectorASyncClient, runNbr:int) -> int:
        queryLen = 0
        resultCnt = 0
        queries = 1
        queryArraysize = len(self._queryarray)
        self._query_recs_gauge.set(queryArraysize,{"ns":self._idx_namespace,"idx":self._idx_name})
        for pos, searchValues in enumerate(self._queryarray):
            queryLen += len(searchValues)
            result = await self.vector_search(client, searchValues.tolist())            
            resultCnt += len(result)
            print('Query Run [%d] Search [%d] Array [%d] Result [%d]                         \r'%(runNbr,pos+1,queryLen,resultCnt), end="")
            self._query_recs_gauge.set(queryArraysize-queries,{"ns":self._idx_namespace,"idx":self._idx_name})
            result_ids = [neighbor.key.key for neighbor in result]
            if self._query_check:
                if len(result_ids) == 0:
                    print('\n')
                    self._exception_counter.add(1, {"exception_type":"No Query Results", "handled_by_user":False,"ns":self._namespace,"set":self._setName})
                    Aerospike.PrintLog(f'No Query Results for {self._idx_namespace}.{self._idx_name}', logging.WARNING)                    
                zeroDist = [record.key.key for record in result if record.distance == 0]
                if len(zeroDist) > 0:
                    print('\n')
                    self._exception_counter.add(1, {"exception_type":"Zero Distance Found", "handled_by_user":False,"ns":self._namespace,"set":self._setName})
                    Aerospike.PrintLog(f'Zero Distance Found for {self._idx_namespace}.{self._idx_name} Keys: {zeroDist}', logging.WARNING)
            queries += 1
        
        self._query_recs_gauge.set(0,{"ns":self._idx_namespace,"idx":self._idx_name})
        
        return queries
        
    async def vector_search(self, client:vectorASyncClient, query:List[float]) -> List[vectorTypes.Neighbor]:
        try:
            result = await client.vector_search(namespace=self._idx_namespace,
                                                index_name=self._idx_name,
                                                query=query,
                                                limit=self._query_limit,
                                                search_params=self._query_hnswparams)
            self._query_counter.add(1, {"type": "Vector Search","ns":self._idx_namespace,"idx":self._idx_name})
        except Exception as e:
            self._exception_counter.add(1, {"exception_type":e, "handled_by_user":False,"ns":self._namespace,"set":self._setName})
            raise
        return result

    def __str__(self):
        arrayLen = None
        if self._trainarray is not None:
            arrayLen = len(self._trainarray)
        if OperationActions.POPULATION in self._actions:
            popstr = f", DropIdx: {self._idx_drop}, Concurrency: {self._concurrency}, MaxRecs: {self._idx_maxrecs}, WaitIdxCompletion: {not self._idx_nowait} Exhausted Evt: {self._idx_resource_event}"
        else:
            popstr = ""
        if OperationActions.QUERY in self._actions:
            qrystr = f", Runs: {self._query_runs}, Parallel: {self._query_parallel}, Check: {self._query_check}"
        else:
            qrystr = ""
            
        return f"Aerospike([{self.basestring()}, Actions: {self._actions}, Dimensions: {self._dimensions}, Array: {arrayLen} DS: {self._datasetname}{popstr}{qrystr}]"
