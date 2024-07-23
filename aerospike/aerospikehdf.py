import asyncio
import numpy as np
import time
import logging
import argparse
import statistics
import json

from enum import Flag, auto
from typing import Iterable, List, Union, Any
from importlib.metadata import version

from aerospike_vector_search import types as vectorTypes, Client as vectorSyncClient
from aerospike_vector_search.aio import AdminClient as vectorASyncAdminClient, Client as vectorASyncClient
from aerospike_vector_search.shared.proto_generated.types_pb2_grpc import grpc  as vectorResultCodes

from baseaerospike import BaseAerospike, _distanceNameToAerospikeType as DistanceMaps, OperationActions
from datasets import DATASETS, load_and_transform_dataset, get_dataset_fn
from metrics import all_metrics as METRICS, DummyMetric
from distance import metrics as DISTANCES, Metric as DistanceMetric

logger = logging.getLogger(__name__)

aerospikeIdxNames : list = []
      
class Aerospike(BaseAerospike):
    
    @staticmethod
    def parse_arguments_population(parser: argparse.ArgumentParser) -> None:
        '''
        Adds the arguments required to populate an index. 
        '''
        group = parser.add_mutually_exclusive_group(required=True)
        group.add_argument(
            '-d', "--dataset",
            metavar="DS",
            help="the dataset to load training points from",
            default="glove-100-angular",
            choices=DATASETS.keys(),                
        )
        group.add_argument(
            "--hdf",
            metavar="HDFFILE",
            help="A HDF file that will be the dataset to load training points from... Defaults to 'data' folder",
            default=None,
            type=str                
        )
        parser.add_argument(
            '-c', "--concurrency",
            metavar="N",
            type=int,
            help='''
    The maximum number of concurrent task used to population the index.
    Values are:
        - < 0 -- All records are upserted, concurrently, and the app will only wait for the upsert completion before waiting for index completion.
        -   0 -- Disable Population (Index is still created and Wait for Idx Completion still performed)
        -   1 -- One record is upserted at a time (sync)
        -   > 1 -- The number of records upserted, concurrently (async), before the app waits for the upserts to complete.
    ''',
            default=500,
        )
        parser.add_argument(
            '-n', "--namespace",
            metavar="NS",
            help="The Aerospike Namespace",
            default="test",
        )
        parser.add_argument(
            '-s', "--setname",
            metavar="SET",
            help="The Aerospike Set Name. Will default to the dataset name",
            default=None,
        )
        parser.add_argument(
            '-N', "--idxnamespace",
            metavar="NS",
            help="Aerospike Namespace where the Vector Idx will be located. Defaults to --Namespace",
            default=None,
            type=str
        )
        parser.add_argument(
            '-I', "--idxname",
            metavar="IDX",
            help="The Vector Index Name. Defaults to the DataSet Name with the suffix of '_idx'",
            default=None,
        )
        parser.add_argument(
            '-b', "--vectorbinname",
            metavar="BIN",
            help="The Aerospike Bin Name where the Vector is stored",
            default="HDF_embedding",
        )
        parser.add_argument(
            '-g', "--generatedetailsetname",            
            help="Generates a Set name based on distance type, dimensions, index params, etc.",
            action='store_true'
        )        
        parser.add_argument(
            '-D', "--distancetype",
            metavar="DIST",
            help="The Vector's Index Distance Type. The default is to select the type based on the dataset",
            type=vectorTypes.VectorDistanceMetric, 
            choices=list(vectorTypes.VectorDistanceMetric),
            default=None
        )
        parser.add_argument(
            '-P', "--indexparams",
            metavar="PARM",
            type=json.loads,
            help="The Vector's Index Params (HnswParams)",
            default='{"m": 16, "ef_construction": 100, "ef": 100}'
        )
        parser.add_argument(
            "--idxdrop",        
            help="If the Vector Index existence, it will be dropped. Otherwise is is updated.",
            action='store_true'
        )        
        parser.add_argument(
            "--idxnowait",        
            help="Waiting for Index Completion is disabled.",
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
        group = parser.add_mutually_exclusive_group(required=True)
        group.add_argument(
            '-d', "--dataset",
            metavar="DS",
            help="the dataset to load the search points from",
            default="glove-100-angular",
            choices=DATASETS.keys(),
        )
        group.add_argument(
            "--hdf",
            metavar="HDFFILE",
            help="A HDF file that will be the dataset to load search points from... Defaults to 'data' folder",
            default=None,
            type=str                
        )
        parser.add_argument(
            '-N', "--idxnamespace",
            metavar="NS",
            help="Aerospike Namespace where the Vector Idx will be located. Defaults to 'test'",
            default="test",
            type=str
        )
        parser.add_argument(
            '-I', "--idxname",
            metavar="IDX",
            help="The Vector Index Name. If not provided, it defaults to the dataset name with a suffix of '_idx'",
            default=None
        )        
        parser.add_argument(
            '-S', "--searchparams",
            metavar="PARM",
            type=json.loads,
            help="The Vector's Search Params (HnswSearchParams)",
            default=None
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
            metavar="NEIGHBORS",
            type=int,
            help="The number of neighbors to return from the query. If <= 0, defaults to the the dataset's neighbor result array length.",
            default=-1,
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
        parser.add_argument(
           "--noresultfile",        
            help="Do not produce a HDF5 result file.",
            action='store_true'
        )
        parser.add_argument(
                "--metric",
                metavar="TYPE",
                help="Which metric to use to calculate Recall",
                default="k-nn",
                type=str,
                choices=METRICS.keys(),
            )        
                
        BaseAerospike.parse_arguments(parser)
        
    def __init__(self, runtimeArgs: argparse.Namespace, actions: OperationActions):
        
        super().__init__(runtimeArgs, logger)
        
        self._actions: OperationActions = actions
        self._datasetname: str = runtimeArgs.dataset
        self._dimensions = None
        self._trainarray : Union[np.ndarray, List[np.ndarray]] = None
        self._queryarray : Union[np.ndarray, List[np.ndarray]] = None
        self._neighbors : Union[np.ndarray, List[np.ndarray]] = None
        self._distances : Union[np.ndarray, List[np.ndarray]] = None
        self._dataset = None
        self._pausePuts : bool = False
        self._pks : Union[np.ndarray, List] = None
        self._pk_consecutivenbrs : bool = False
        self._hdf_file : str = None
        
        if runtimeArgs.hdf is not None:
            self._hdf_file, self._datasetname = get_dataset_fn(runtimeArgs.hdf)
        
        if OperationActions.POPULATION in actions:
            self._idx_drop = runtimeArgs.idxdrop
            self._concurrency = runtimeArgs.concurrency
            self._idx_nowait = runtimeArgs.idxnowait
            self._idx_resource_event = runtimeArgs.exhaustedevt
            self._idx_resource_cnt = 0
            self._idx_maxrecs = runtimeArgs.maxrecs
            self._namespace = runtimeArgs.namespace
            if runtimeArgs.idxnamespace is None or not runtimeArgs.idxnamespace:
                self._idx_namespace = self._namespace
            else:
                self._idx_namespace = runtimeArgs.idxnamespace
            
            if runtimeArgs.setname is None or not runtimeArgs.setname:
                self._setName = self._datasetname
            else:
                self._setName = runtimeArgs.setname
            self._paramsetname = runtimeArgs.generatedetailsetname
            if runtimeArgs.idxname is None or not runtimeArgs.idxname:
                self._idx_name = f'{self._datasetname}_Idx'
            else:
                self._idx_name = runtimeArgs.idxname              
            self._idx_binName = runtimeArgs.vectorbinname
            
            self._idx_distance = runtimeArgs.distancetype
            if runtimeArgs.indexparams is None or len(runtimeArgs.indexparams) == 0:
                self._idx_hnswparams = None
            else:
                self._idx_hnswparams = BaseAerospike.set_hnsw_params_attrs(
                                            vectorTypes.HnswParams(),
                                            runtimeArgs.indexparams
                                        )                        
                    
        if OperationActions.QUERY in actions:
            self._idx_namespace = runtimeArgs.idxnamespace
            if runtimeArgs.idxname is None or not runtimeArgs.idxname:
                self._idx_name = f'{self._datasetname}_Idx'
            else:
                self._idx_name = runtimeArgs.idxname            
            self._query_runs = runtimeArgs.runs
            self._query_parallel = runtimeArgs.parallel
            self._query_check = runtimeArgs.check
            self._query_nbrlimit = runtimeArgs.limit
            self._query_metric = METRICS[runtimeArgs.metric]
            self._query_metric_result = DummyMetric()
            self._query_metric_aerospike_result = DummyMetric()
            self._query_metric_bigann_result = DummyMetric()
            self._query_neighbors : List = None
            self._query_distance : List = None
            self._query_distance_aerospike : List = None
            self._query_latencies : List[float] = None
            self._query_produce_resultfile : bool = not runtimeArgs.noresultfile
            
            if runtimeArgs.searchparams is None or len(runtimeArgs.searchparams) == 0:
                self._query_hnswparams = None
            else:
                self._query_hnswparams = BaseAerospike.set_hnsw_params_attrs(
                                            vectorTypes.HnswSearchParams(),
                                            runtimeArgs.searchparams
                                        )

    async def __aenter__(self):
        return self
 
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self._logger.exception(f"error detected on exit.")
        await super().shutdown(exc_type is None)
        
    async def get_dataset(self) -> None:
        
        self.print_log(f'get_dataset: {self} hdf: {self._hdf_file}')
        
        (self._trainarray,
            self._queryarray,
            self._neighbors,
            self._distances,
            distance,
            self._dataset,
            self._dimensions,
            self._pks) = load_and_transform_dataset(self._datasetname, self._hdf_file)
        
        distance = distance.lower() 
        if self._ann_distance is None:
            self._ann_distance = distance
        elif distance != self._ann_distance:
            self.print_log(f"ANN distance types do not match! Found: {distance} Provided: {self._ann_distance}. Distance calculations could be wrong!", logging.WARN)
            
        if self._idx_distance is None or not self._idx_distance:
            self._idx_distance = DistanceMaps.get(distance)
        
        if self._idx_distance is None or not self._idx_distance:
             raise ValueError(f"Distance Map '{distance}' was not found.")
         
        if self._dataset.attrs.get('sourceidxname', None) is not None:
            if (self._idx_name != self._dataset.attrs['sourceidxname']
                or self._idx_namespace != self._dataset.attrs['sourceisxnamespace']):
                self.print_log(f"Index Name doesn't match source idx name in HDF file. Names are: {self._idx_namespace}.{self._idx_name} != {self._dataset.attrs['sourceisxnamespace']}.{ self._dataset.attrs['sourceidxname']}",
                                logging.WARN)
         
        if self._paramsetname:
            if self._idx_distance.casefold() == distance.casefold():
                setNameType = self._idx_distance
            else:
                setNameType = f'{distance}_{self._idx_distance}'
            self._setName = f'{self._setName}_{setNameType}_{self._dimensions}_{self._idx_hnswparams.m}_{self._idx_hnswparams.ef_construction}_{self._idx_hnswparams.ef}'
            self._idx_name = f'{self._setName}_Idx'             
        
        self._canchecknbors = self._neighbors is not None and len(self._neighbors) > 0
        if self._canchecknbors and (self._query_nbrlimit is None or self._query_nbrlimit <= 0 or self._query_nbrlimit > len(self._neighbors[0])):
            self._query_nbrlimit = len(self._neighbors[0])
            
        self._remainingrecs = 0           
        self._remainingquerynbrs = 0
        
        if self._dataset.attrs.get("recall", None) is not None:
            self.print_log(f"Precalculated Recall value found {self._dataset.attrs['recall']}")

        if self._pks is not None and type(self._pks) is np.ndarray:
                self._pks = self._pks.tolist()
                
        if (self._pks is not None
                and len(self._pks) > 1 
                and type(self._pks[0]) is int
                and type(self._pks[-1]) is int
                and self._pks[0] == 0
                and len(self._pks) == len(self._trainarray)):
            numRange = list(range(0, self._pks[-1]+1))
            self._pk_consecutivenbrs = self._pks == numRange
        else:
            self._pk_consecutivenbrs = False
            
        self._heartbeat_stage = 1
        self.prometheus_status()
        self.print_log(f'get_dataset Exit: {self}, {self._ann_distance}, {self._idx_distance}, Train Array: {len(self._trainarray)}, Query Array: {len(self._queryarray)}, Distance: {distance}, Dimensions: {self._dimensions}, Neighbors: {0 if self._neighbors is None else len(self._neighbors)}, Distances: {0 if self._distances is None else len(self._distances)}, PK array: {0 if self._pks is None else len(self._pks)}, PK consistence: {self._pk_consecutivenbrs}, Neighbors Check: {self._canchecknbors}')
                
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
        await adminClient.index_create(namespace=self._namespace,
                                                name=self._idx_name,
                                                sets=self._setName,
                                                vector_field=self._idx_binName,
                                                dimensions=self._dimensions,
                                                index_params= self._idx_hnswparams,
                                                vector_distance_metric=self._idx_distance,
                                                index_storage= vectorTypes.IndexStorage(namespace=self._idx_namespace)
                                                )
        t = time.time()
        self.print_log(f'Index Creation Time (sec) = {t - s}')        
        aerospikeIdxNames.append(self._idx_name)

    async def _wait_for_idx_completion(self, client: vectorASyncClient):
        self._waitidx_counter.add(1, {"ns":self._idx_namespace,"idx": self._idx_name})
        try:
            self._waitidx = True
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
            self._waitidx = False
            
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
        self._pausePuts = False
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
            self._pausePuts = False
            
            if logLevel == logging.WARNING:
                self.print_log(msg=f"Resuming population for Idx: {self._idx_namespace}.{self._idx_name}",
                                    logLevel=logging.WARNING)
            else:
                logger.debug(msg=f"Resuming population for Count: {i}, Key: {key}, Idx: {self._idx_namespace}.{self._idx_name}")

    async def _resourceexhaused_handler(self, key: int, embedding, i: int, client: vectorASyncClient) -> None:
        self._exception_counter.add(1, {"exception_type": "Resource Exhausted", "handled_by_user": True,"ns":self._namespace,"set":self._setName})
        logLevel = logging.DEBUG
        if not self._pausePuts:
            self._pausePuts = True
            logLevel = logging.WARNING
            self.print_log(msg=f"\nResource Exhausted on Put first encounter on Count: {i}, Key: {key}, Idx: {self._idx_namespace}.{self._idx_name}. Going to Pause Population and Wait for Idx Completion...",
                                logLevel=logging.WARNING)
        else:
            logger.debug(f"Resource Exhausted on Put on Count: {i}, Key: {key}, Idx: {self._idx_namespace}.{self._idx_name}. Going to Pause Population and Wait for Idx Completion...")
        
        if self._idx_resource_event < 0:
            await self._put_wait_completion_handler(key, embedding, i, client, logLevel)                        
        else:
            await self._put_wait_sleep_handler(key, embedding, i, client, logLevel)
        
    async def put_vector(self, key, embedding, i: int, client: vectorASyncClient, retry: bool = False) -> None:
        try:
            try:
                try:
                    if type(key).__module__ == np.__name__:
                        key = key.item()
                    await client.upsert(namespace=self._namespace,
                                        set_name=self._setName,
                                        key=key,
                                        record_data={
                                            self._idx_binName:embedding.tolist()
                                        }
                    )                
                except vectorTypes.AVSServerError as avse:
                    if self._idx_resource_event != 0 and not retry and avse.rpc_error.code() == vectorResultCodes.StatusCode.RESOURCE_EXHAUSTED:
                        await self._resourceexhaused_handler(key, embedding, i, client)
                    else:
                        raise
            except vectorTypes.AVSServerError as avse:
                    if str(avse.rpc_error.details()).find("Server memory error") != -1:
                        raise RuntimeError(f"A Stop Write was detected which indicates the Aerospike DB is out of memory. AVS Error: {avse.rpc_error.debug_error_string()}")
                    else:
                        raise
        except Exception as e:
            print(f'\n** Count: {i} Key: {key} Exception: "{e}" **\r\n')
            logger.exception(f"Put Failure on Count: {i}, Key: {key}, Idx: {self._idx_namespace}.{self._idx_name}, Retry: {retry}")
            self.flush_log()
            self._exception_counter.add(1, {"exception_type":f"upsert: {e}", "handled_by_user":False,"ns":self._namespace,"set":self._setName})
            self._pausePuts = False
            raise

    async def index_exist(self, adminClient: vectorASyncAdminClient) -> Union[dict, None]:        
        existingIndexes = await adminClient.index_list()
        if len(existingIndexes) == 0:
            return None
        indexInfo = [(index if index["id"]["namespace"] == self._idx_namespace
                            and index["id"]["name"] == self._idx_name else None)
                        for index in existingIndexes]
        if all(v is None for v in indexInfo):
            return None
        return next(i for i in indexInfo if i is not None)
        
    async def populate(self) -> None:
        '''
        Polulates the vector index based on HDF dataset.
        '''
        global aerospikeIdxNames
        
        if self._trainarray.dtype != np.float32:
            self._trainarray = self._trainarray.astype(np.float32)
    
        self.print_log(f'populate: {self} Shape: {self._trainarray.shape}')

        self._heartbeat_stage = 2
        self.prometheus_status()

        async with vectorASyncAdminClient(seeds=self._host,
                                            listener_name=self._listern,
                                            is_loadbalancer=self._useloadbalancer
            ) as adminClient:
            
            #If exists, no sense to try creation...
            idxinfo = await self.index_exist(adminClient)
            if idxinfo is not None:
                self.print_log(f'Index {self._idx_namespace}.{self._idx_name} Already Exists. Idx Info: {idxinfo}')
                
                #since this can be an external DB (not in a container), we need to clean up from prior runs
                #if the index name is in this list, we know it was created in this run group and don't need to drop the index.
                #If it is a fresh run, this list will not contain the index and we know it needs to be dropped.
                if self._idx_name in aerospikeIdxNames:
                    self.print_log(f'Index {self._idx_name} being reused (updated)')
                    self._idx_hnswparams = BaseAerospike.set_hnsw_params_attrs(vectorTypes.HnswParams(),
                                                                                idxinfo)
                elif self._idx_drop:
                    await self.drop_index(adminClient)
                    await self.create_index(adminClient)
                else:
                    self.print_log(f'Index {self._idx_namespace}.{self._idx_name} being updated')
                    self._idx_hnswparams = BaseAerospike.set_hnsw_params_attrs(vectorTypes.HnswParams(),
                                                                                idxinfo)
            else:
                await self.create_index(adminClient)

        async with vectorASyncClient(seeds=self._host,
                                        listener_name=self._listern,
                                        is_loadbalancer=self._useloadbalancer
                        ) as client: 
            
            self._remainingrecs = len(self._trainarray) if self._idx_maxrecs < 0 else self._idx_maxrecs >= 0
            if self._concurrency == 0 or self._idx_maxrecs == 0:
                s = time.time()
            else:
                self._pausePuts = False
                self.print_log(f'Populating Index {self._idx_namespace}.{self._idx_name}')                    
                s = time.time()
                taskPuts = []
                i = 1
                self._populate_counter.add(0, {"type": "upsert","ns":self._namespace,"set":self._setName})
                usePKValues : bool = self._pks is not None
                
                for key, embedding in enumerate(self._trainarray):
                    if usePKValues:
                        key = self._pks[key]
                    if self._pausePuts:
                        loopTimes = 0
                        await asyncio.gather(*taskPuts)
                        self._populate_counter.add(len(taskPuts), {"type": "upsert","ns":self._namespace,"set":self._setName})                        
                        logger.debug(f"Put Tasks Completed for Paused Population")
                        self._remainingrecs -= len(taskPuts)
                        taskPuts.clear()
                        print('\n')
                        while (self._pausePuts):
                            if loopTimes % 30 == 0:
                                self.print_log(f"Paused Population still waiting for Idx Completion at {loopTimes} mins!", logging.WARNING)                                
                            loopTimes += 1                                
                            logger.debug(f"Putting Paused {loopTimes}")
                            await asyncio.sleep(60)
                        self.print_log(f"Resuming Population at {loopTimes} mins", logging.WARNING)
                                                
                    if self._concurrency < 0:
                        taskPuts.append(self.put_vector(key, embedding, i, client))
                    elif self._concurrency <= 1:
                        await self.put_vector(key, embedding, i, client)
                        self._populate_counter.add(1, {"type": "upsert","ns":self._namespace,"set":self._setName})
                        self._remainingrecs -= 1
                    else:
                        taskPuts.append(self.put_vector(key, embedding, i, client))
                        if len(taskPuts) >= self._concurrency:
                            logger.debug(f"Waiting for Put Tasks ({len(taskPuts)}) to Complete at {i}")
                            await asyncio.gather(*taskPuts)
                            self._populate_counter.add(len(taskPuts), {"type": "upsert","ns":self._namespace,"set":self._setName})
                            logger.debug(f"Put Tasks Completed")
                            self._remainingrecs -= len(taskPuts)
                            taskPuts.clear()
                            
                    print('Index Put Counter [%d]\r'%i, end="")
                    i += 1
                    if self._idx_maxrecs >= 0 and i > self._idx_maxrecs:
                        break
                    
                i -= 1
                logger.debug(f"Waiting for Put Tasks (finial {len(taskPuts)}) to Complete at {i}")                            
                await asyncio.gather(*taskPuts)
                self._populate_counter.add(len(taskPuts), {"type": "upsert","ns":self._namespace,"set":self._setName})
                t = time.time()
                self._remainingrecs -= len(taskPuts)
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

    async def create_query_hdf(self) -> str:
        import h5py
        import os
        from string import digits
        from datasets import get_dataset_fn
    
        hdfpath, _ = get_dataset_fn(self._datasetname, "results")
        self.print_log(f'Creating Query HDF dataset {hdfpath}')
 
        with h5py.File(hdfpath, "w") as f:
            f.attrs["algo"] = "aerospike-standalone-ann"
            f.attrs["batch_mode"] = self._query_parallel
            f.attrs["best_search_time"] = round(min(self._query_latencies) * 0.001, 3)
            f.attrs["build_time"] = 0
            f.attrs["candidates"] = self._query_nbrlimit
            f.attrs["count"] = len(self._query_neighbors[0])
            f.attrs["dataset"] = self._datasetname
            if self._hdf_file is not None:
                f.attrs["query_hdf"] = self._hdf_file
            f.attrs["distance"] = self._ann_distance
            f.attrs["distance_aerospike"] = self._idx_distance.name
            f.attrs["hnsw"] = self.hnswstr()
            
            if self._query_hnswparams is None:
                queryef = '' if self._idx_hnswparams is None else str(self._idx_hnswparams.ef)
            else:
                queryef = self._query_hnswparams.ef            
            f.attrs["ef"] = queryef
            
            f.attrs["expect_extra"] = False
            f.attrs["index_size"] = 0
            if self._namespace == self._idx_namespace:
                f.attrs["name"] = f'{self._namespace}.{self._setName}.{self._idx_name}, ef:{queryef}, hnsw:{f.attrs["hnsw"]}, k:{self._query_nbrlimit}'
            else:
                f.attrs["name"] = f'{self._namespace}.{self._setName}.{self._idx_name}, ef:{queryef}, hnsw:{f.attrs["hnsw"]}, k:{self._query_nbrlimit}'
                
            f.attrs["run_count"] = self._query_runs
            
            if self._query_distance is not None:
                f.create_dataset("distances", data=self._query_distance)
            f.create_dataset("distance_aerospike", data=self._query_distance_aerospike)
            f.create_dataset("neighbors", data=self._query_neighbors)            
            f.create_dataset("times", data=[round(l * 0.001,3) for l in self._query_latencies])
            
            grp = f.create_group("metrics")
            
            if self._query_metric is not None:
                if self._query_metric_result is not None:
                    metrictype = self._query_metric["type"]
                    mgrp = grp.create_group(metrictype)
                    mgrp.attrs["mean"] = self._query_metric_result.d[metrictype].attrs["mean"]
                    mgrp.attrs["std"] = self._query_metric_result.d[metrictype].attrs["std"]
                    mgrp.create_dataset("recalls", data=self._query_metric_result.d[metrictype].d["recalls"])
                
                if self._query_metric_aerospike_result is not None:
                    metrictype = self._query_metric["type"]
                    mgrp = grp.create_group(f'{metrictype}_aerospike')
                    mgrp.attrs["mean"] = self._query_metric_aerospike_result.d[metrictype].attrs["mean"]
                    mgrp.attrs["std"] = self._query_metric_aerospike_result.d[metrictype].attrs["std"]
                    mgrp.create_dataset("recalls", data=self._query_metric_aerospike_result.d[metrictype].d["recalls"])
                            
            if self._query_metric_bigann_result is not None:
                metrictype = "knn"               
                mgrp = grp.create_group("knn_big")
                mgrp.attrs["mean"] = self._query_metric_bigann_result.d[metrictype].attrs["mean"]
                mgrp.attrs["std"] = self._query_metric_bigann_result.d[metrictype].attrs["std"]
                mgrp.create_dataset("recalls", data=self._query_metric_bigann_result.d[metrictype].d["recalls"])
                     
            self.print_log(f"Created HDF dataset '{hdfpath}'")
            
        return hdfpath

    async def query(self) -> None:
        '''
        Performs a query using the query array from the HDF dataset. 
        '''
        from distance import metrics as DISTANCES
        from bigann.metrics import knn as bigknn
        
        self.print_log(f'Query: {self} Shape: {self._trainarray.shape}')
        
        async with vectorASyncAdminClient(seeds=self._host,
                                            listener_name=self._listern,
                                            is_loadbalancer=self._useloadbalancer
            ) as adminClient:
           
            idxinfo = await self.index_exist(adminClient)
            if idxinfo is None:
                self.print_log(f'Query: Vector Index: {self._idx_namespace}.{self._idx_name}, not found')
                self._exception_counter.add(1, {"exception_type":"Index not found", "handled_by_user":False,"ns":self._idx_namespace,"set":self._idx_name})
                raise FileNotFoundError(f"Vector Index {self._idx_namespace}.{self._idx_name} not found")
            
            self.print_log(f'Found Index {self._idx_namespace}.{self._idx_name} with Info {idxinfo}')
            self._idx_hnswparams = BaseAerospike.set_hnsw_params_attrs(vectorTypes.HnswParams(),
                                                                        idxinfo['hnsw_params'])
            self._idx_binName = idxinfo["field"]
            self._setName = idxinfo["setFilter"]
            self._namespace = idxinfo["id"]["namespace"]
            if self._query_hnswparams is None and self._idx_hnswparams is not None:           
                self._query_hnswparams = vectorTypes.HnswSearchParams(ef=self._idx_hnswparams.ef)
            
        self._heartbeat_stage = 2
        self.prometheus_status()
            
        self.print_log(f'Starting Query Runs ({self._query_runs}) on {self._idx_namespace}.{self._idx_name}')
        metricfunc = None
        distancemetric : DistanceMetric= None
        if self._canchecknbors:
            metricfunc = None if self._query_metric is None else self._query_metric["function"]
            distancemetric = DISTANCES[self._ann_distance]
            
        async with vectorASyncClient(seeds=self._host,
                                        listener_name=self._listern,
                                        is_loadbalancer=self._useloadbalancer
                            ) as client:
            s = time.time()
            totalquerytime : float = 0.0
            taskPuts = []
            queries = []
            metricValues = []
            metricValuesAS = []
            metricValuesBig = []
            i = 1
            self._remainingquerynbrs = len(self._queryarray) * self._query_runs
            while i <= self._query_runs:
                if self._query_parallel:
                    taskPuts.append(self.query_run(client, i, distancemetric))
                else:
                    (self._query_distance,
                        self._query_distance_aerospike,
                        self._query_neighbors,
                        self._query_latencies) = await self.query_run(client, i, distancemetric)

                    totalquerytime += sum(self._query_latencies)
                    
                    if metricfunc is not None:                        
                        if len(self._query_distance) == 0:
                            self._query_metric_value = None
                        else:                            
                            self._query_metric_value = metricfunc(self._distances, self._query_distance, self._query_metric_result, i-1, len(self._query_distance[0]))
                            metricValues.append(self._query_metric_value)
                        self._aerospike_metric_value = metricfunc(self._distances, self._query_distance_aerospike, self._query_metric_aerospike_result, i-1, len(self._query_distance_aerospike[0]))
                        metricValuesAS.append(self._aerospike_metric_value)
                    
                    if len(self._query_neighbors) == 0 or not self._canchecknbors:
                        self._query_metric_big_value = None
                    else:
                        self._query_metric_big_value = bigknn((self._neighbors,self._distances), self._query_neighbors, len(self._query_neighbors[0]), self._query_metric_bigann_result).attrs["mean"]
                        metricValuesBig.append(self._query_metric_big_value)
                
                    self._logger.info(f"Run: {i}, Neighbors: {len(self._query_neighbors)}, {'No distance' if distancemetric is None else distancemetric.type} {self._query_metric['type']}: {self._query_metric_value}, aerospike recall: {self._aerospike_metric_value}, Big: {self._query_metric_big_value}")
            
                i += 1
                
            if len(taskPuts) > 0:
                queries = await asyncio.gather(*taskPuts)                
                i = 0
                totalquerytime = 0.0
                for rundist, runasdist, runnns, times in queries:
                    if metricfunc is not None:
                        if len(rundist) > 0:
                            metricValues.append(metricfunc(self._distances, rundist, self._query_metric_result, i, len(rundist[0])))
                        metricValuesAS.append(metricfunc(self._distances, runasdist, self._query_metric_aerospike_result, i, len(runasdist[0])))
                    if len(runnns) > 0 and self._canchecknbors:
                            metricValuesBig.append(bigknn((self._neighbors,self._distances), runnns, len(runnns[0]), self._query_metric_bigann_result).attrs["mean"])                        
                    self._query_distance = rundist
                    self._query_distance_aerospike = runasdist
                    self._query_neighbors = runnns
                    self._query_latencies = times
                    totalquerytime += sum(times)
                    i += 1
                
            t = time.time()
            self._query_metric_value = statistics.mean(metricValues)
            self._aerospike_metric_value = statistics.mean(metricValuesAS)
            self._query_metric_big_value = statistics.mean(metricValuesBig)
                        
            print('\n')
            totqueries = sum([len(x[1]) for x in queries])

            self.print_log(f'Finished Query Runs on {self._idx_namespace}.{self._idx_name}; Total queries {totqueries} in {totalquerytime} secs (overall {{t-s}} secs), {totqueries/totalquerytime} TPS, {None if self._query_metric is None else self._query_metric["type"]}: {self._query_metric_value}, Aerospike recall: {self._aerospike_metric_value}, Big Recall: {self._query_metric_big_value}')
            
            if self._query_produce_resultfile and self._query_neighbors is not None:
                await self.create_query_hdf()

    async def _check_query_neighbors(self, results:List[Any], idx:int, runNbr:int) -> bool:
    
        compareresult = None                
        
        if len(results) == len(self._neighbors[idx]):
            compareresult = results == self._neighbors[idx]
            if all(compareresult):
                return True
        elif self._query_nbrlimit > len(self._neighbors[idx]):
            compareresult = results[:len(self._neighbors[idx])] == self._neighbors[idx]
            if all(compareresult):
                return True
        else:
            compareresult = results == self._neighbors[idx][:self._query_nbrlimit]
            if all(compareresult):
                return True
        self._exception_counter.add(1, {"exception_type":"Results don't Match Expected Neighbors", "handled_by_user":False,"ns":self._idx_namespace,"idx":self._idx_name,"run":runNbr})
        logger.warn(f"Results do not Match Expected Neighbors for {self._idx_namespace}.{self._idx_name} for Query {idx}")
        logger.warn(f"Comparison Results: {compareresult}")
        
        return False
    
    async def _check_query_distances(self, distances:List[float], distances_aerospike:List[float], idx:int, runNbr:int) -> bool:    
        import operator

        subresult = np.array(list(map(operator.sub, distances, distances_aerospike)))
        subresult = np.around(subresult, 6)
        
        if np.all(subresult == 0):
            return True;
        
        self._exception_counter.add(1, {"exception_type":"Distances don't match", "handled_by_user":False,"ns":self._idx_namespace,"idx":self._idx_name,"run":runNbr})
        logger.warn(f"Distance Results do not Match between Calculated and Aerospike (negative values towards Aerospike)  {self._idx_namespace}.{self._idx_name} for Query {idx}")
        logger.warn(f"Results: {subresult.tolist()}")
        
        return False

    def _get_orginal_vector_from_pk(self, pk : any) -> Union[np.ndarray, List]:
        
        self._logger.debug(f"_get_orginal_vector_from_pk: pk:{pk}")
        try:
            if self._pk_consecutivenbrs or self._pks is None:
                return self._trainarray[pk]

            fndidx = self._pks.index(pk)
            self._logger.debug(f"_get_orginal_vector_from_pk: {pk} idx:{fndidx}")
            return self._trainarray[fndidx]
        except IndexError as e:
            self._logger.exception(f"pk: {pk}")
            self._exception_counter.add(1, {"exception_type":f"PK index lookup Failed", "handled_by_user":True,"ns":self._idx_namespace,"set":self._idx_name})
            return np.zeros(len(self._trainarray[0]))
        except ValueError as e:
            self._logger.exception(f"pk: {pk}")
            self._exception_counter.add(1, {"exception_type":f"PK value error lookup Failed", "handled_by_user":True,"ns":self._idx_namespace,"set":self._idx_name})
            return np.zeros(len(self._trainarray[0]))

    async def query_run(self, client:vectorASyncClient, runNbr:int, distancemetric : DistanceMetric) -> tuple[List, List, List, List[float]]:
        '''
        Returns a tuple of calculated distances, aerospike distances, neighbors, latency times
        '''
        queryLen = 0
        resultCnt = 0
        rundistance = []
        runasdistance = []
        runnneighbors = []
        runlatencies = []
        
        self._query_current_run = runNbr
        self._query_counter.add(0, {"type": "Vector Search","ns":self._idx_namespace,"idx":self._idx_name, "run": runNbr})
        msg = "                                   "
        for pos, searchValues in enumerate(self._queryarray):
            queryLen += len(searchValues)
            result, latency = await self.vector_search(client, searchValues.tolist(),runNbr)
            resultCnt += len(result) if result is not None else 0
            print('Query Run [%d] Search [%d] Array [%d] Result [%d] %s\r'%(runNbr,pos+1,queryLen,resultCnt,msg), end="")
            self._remainingquerynbrs -= 1
            if result is None:
                print('\n')
                self.print_log(f"Skipping Query Run {runNbr} Search {pos+1} since search failed...")
                runnneighbors.append([])
                runasdistance.append([])
                rundistance.append([])
                runlatencies.append(latency)
                continue
            result_ids = [neighbor.key.key for neighbor in result]            
            aerospike_distances = [neighbor.distance for neighbor in result]
            
            if self._query_check:
                if len(result_ids) == 0:
                    self._exception_counter.add(1, {"exception_type":"No Query Results", "handled_by_user":False,"ns":self._idx_namespace,"set":self._idx_name,"run":runNbr})
                    logger.warn(f'No Query Results for {self._idx_namespace}.{self._idx_name}', logging.WARNING)
                    msg = "Warn: No Results"
                elif self._canchecknbors and len(self._neighbors[len(rundistance)]) > 0:
                    if not await self._check_query_neighbors(result_ids, len(rundistance), runNbr):
                        msg = "Warn: Neighbor Compare Failed"                    
                else:                        
                    zeroDist = [record.key.key for record in result if record.distance == 0]
                    if len(zeroDist) > 0:
                        self._exception_counter.add(1, {"exception_type":"Zero Distance Found", "handled_by_user":False,"ns":self._idx_namespace,"set":self._idx_name,"run":runNbr})
                        logger.warn(f'Zero Distance Found for {self._idx_namespace}.{self._idx_name} Keys: {zeroDist}', logging.WARNING)
                        msg = "Warn: Zero Distance Found"
            if distancemetric is not None:
                try:
                    distances = [float(distancemetric.distance(searchValues, self._get_orginal_vector_from_pk(idx))) for idx in result_ids]
                    
                    if self._query_check and not await self._check_query_distances(distances, aerospike_distances, len(rundistance), runNbr):
                        if len(msg) == 0:
                            msg = "Warn: Distances don't match"
                        else:
                            msg += ", Distances don't match"
                    rundistance.append(distances)
                except Exception as e:
                    msg = f"Distance Calculation Failed: {e}"
                    self._logger.exception(f"Distance Calculation Failed Run: {runNbr}")
                    self._exception_counter.add(1, {"exception_type":f"Distance Calculation Failed", "handled_by_user":True,"ns":self._idx_namespace,"set":self._idx_name,"run":runNbr})
                    rundistance.append([])
                    
            runnneighbors.append(result_ids)
            runasdistance.append(aerospike_distances)
            runlatencies.append(latency)

        return rundistance, runasdistance, runnneighbors, runlatencies
        
    async def vector_search(self, client:vectorASyncClient, query:List[float], runNbr:int) -> tuple[List[vectorTypes.Neighbor], float]:
        import math
        try:
            latency : int
            s = time.time_ns()
            result = await client.vector_search(namespace=self._idx_namespace,
                                                index_name=self._idx_name,
                                                query=query,
                                                limit=self._query_nbrlimit,
                                                search_params=self._query_hnswparams)
            t = time.time_ns()
            latency = (t-s)*math.pow(10,-6)
            self._query_counter.add(1, {"type": "Vector Search","ns":self._idx_namespace,"idx":self._idx_name, "run": runNbr})
            self._query_histogram.record(latency, {"ns":self._idx_namespace,"idx": self._idx_name, "run": runNbr})
        except vectorTypes.AVSServerError as e:
            if "unknown vector" in e.rpc_error.details():
                self._exception_counter.add(1, {"exception_type":f"vector_search: {e.rpc_error.details()}", "handled_by_user":True,"ns":self._idx_namespace,"set":self._idx_name,"run":runNbr})
                return None, (time.time_ns()-s)*math.pow(10,-6)
            
            self._exception_counter.add(1, {"exception_type":f"vector_search: {e.rpc_error.details()}", "handled_by_user":False,"ns":self._idx_namespace,"set":self._idx_name,"run":runNbr})
            raise
        except Exception as e:
            self._exception_counter.add(1, {"exception_type":f"vector_search: {e}", "handled_by_user":False,"ns":self._idx_namespace,"set":self._idx_name,"run":runNbr})
            raise
        return result, latency

    def __str__(self):
        return super().__str__()
