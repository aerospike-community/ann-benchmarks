import asyncio
import os
import numpy as np
import time
import enum
import os
import logging

from typing import Iterable, List, Any
from pythonping import ping

from aerospike_vector_search import types as vectorTypes, Client as vectorSyncClient
from aerospike_vector_search.aio import AdminClient as vectorASyncAdminClient, Client as vectorASyncClient

from ..base.module import BaseANN

loggerASClient = logging.getLogger("aerospike_vector_search")
logger = logging.getLogger(__name__)

_AerospikeIdxNames : list = []

class OperationActions(enum.Enum):
    ALLOPS = 0
    IDXPOPULATEONLY = 1
    QUERYONLY = 2
    
class Aerospike(BaseANN):
           
    def __init__(self,
                    metric: str, 
                    dimension: int,
                    idx_type: str,
                    hnswParams: dict,
                    uniqueIdxName: bool = True,
                    dropIdx: bool = True,
                    actions: str = "ALLOPS",
                    ping: bool = False):
        
        asLogFile = os.environ.get("APP_LOGFILE")
        self._indocker = Aerospike.InDocker()
        
        if self._indocker:
            print("Aerospike: Running In Docker Container")
        elif asLogFile is None:
                asLogFile = "AerospikeANN.log"
        
        if not self._indocker and asLogFile is not None and asLogFile:
            print(f"Aerospike: Logging to file {os.getcwd()}/{asLogFile}")
            self._logFileHandler = logging.FileHandler(asLogFile, "w+")        
            logFormatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            self._logFileHandler.setFormatter(logFormatter)
            loggerASClient.addHandler(self._logFileHandler)
            logger.addHandler(self._logFileHandler)
            loggerASClient.setLevel(logging.DEBUG)
            logger.setLevel(logging.DEBUG)                        
            logger.info('Start Aerospike ANN Client')
            
        self._metric = metric
        self._dims = dimension
        self._idx_type = idx_type.upper()        
        self._idx_value = vectorTypes.VectorDistanceMetric[self._idx_type]
        self._actions = OperationActions[actions.upper()]
        if hnswParams is None or len(hnswParams) == 0:
            self._idx_hnswparams = None
        else:
            self._idx_hnswparams = Aerospike.SetHnswParamsAttrs(
                                        vectorTypes.HnswParams(),
                                        hnswParams
                                    )
        self._idx_drop = dropIdx
        #self._username = os.environ.get("APP_USERNAME") or ""
        #self._password = os.environ.get("APP_PASSWORD") or ""
        self._host = os.environ.get("PROXIMUS_HOST") or "localhost"
        self._port = int(os.environ.get("PROXIMUS_PORT") or 5000)
        self._listern = None #os.environ.get("PROXIMUS_ADVERTISED_LISTENER") or None          
        self._namespace = os.environ.get("PROXIMUS_NAMESPACE") or "test"
        self._setName = os.environ.get("PROXIMUS_SET") or "ANN-data"
        self._verifyTLS = os.environ.get("VERIFY_TLS") or True
        self._idx_sleep = int(os.environ.get("INDEX_SLEEP") or 300)
        self._populateTasks = int(os.environ.get("APP_POPULATE_TASKS") or 5000)
        
        if not uniqueIdxName or self._idx_hnswparams is None:
            self._idx_name = f'{self._setName}_{self._idx_type}_Idx'
        else:
            self._idx_name = f'{self._setName}_{self._idx_type}_{self._dims}_{self._idx_hnswparams.m}_{self._idx_hnswparams.ef_construction}_{self._idx_hnswparams.ef}_Idx'
        self._idx_binName = "ANN_embedding"
        self._idx_binKeyName = "ANN_key"
        self._query_hnswsearchparams = None
        
        if ping:
            print(f'Aerospike: Trying Ping to {self._host} {self._verifyTLS} {self._listern}')
            print(ping(self._host, verbose=True))
                    
        print('Aerospike: Try Create Sync Client')
        self._syncClient = vectorSyncClient(
                                    seeds=vectorTypes.HostPort(host=self._host,
                                                            port=self._port,
                                                            is_tls=self._verifyTLS), 
                                    listener_name=self._listern)
        
        print(f'Aerospike: init completed: {self} Time: {time.strftime("%Y-%m-%d %H:%M:%S")}')
        logger.info(f"init completed: {self}")
        
    @staticmethod
    def InDocker() -> bool:
        """ Returns: True if running in a Docker container, else False """
        with open('/proc/1/cgroup', 'rt') as ifh:
            return 'docker' in ifh.read()
        
    @staticmethod
    def SetHnswParamsAttrs(__obj :object, __dict: dict) -> object:
        for key in __dict: 
            if key == 'batching_params':
                setattr(
                    __obj,
                    key,
                    Aerospike.SetHnswParamsAttrs(
                            vectorTypes.HnswBatchingParams(),
                            __dict[key].asdict()
                    )
                )
            else:
                setattr(__obj, key, __dict[key])
        return __obj
        
    def done(self) -> None:
        """Clean up BaseANN once it is finished being used."""
        print(f'Aerospike: done: {self} Time: {time.strftime("%Y-%m-%d %H:%M:%S")}')
        logger.info(f"done: {self}")
        
        if self._syncClient is not None:
            clientCloseTask = self._syncClient.close()
                    
    async def DropIndex(self, adminClient: vectorASyncAdminClient) -> bool:
        print(f'Aerospike: Dropping Index {self._namespace}.{self._idx_name}...')
        logger.info(f'Dropping Index {self._namespace}.{self._idx_name}')
        result = False
        s = time.time()

        await adminClient.index_drop(namespace=self._namespace,
                                            name=self._idx_name)
        
        existingIndexes = await adminClient.index_list()
        loopTimes = 0
        result = True
        if self._idx_sleep > 0:
            while (any(index["id"]["namespace"] == self._namespace
                                    and index["id"]["name"] == self._idx_name 
                                for index in existingIndexes)):
                if loopTimes>= self._idx_sleep:
                    print(f"\nAerospike: Drop Index Timed Out!")
                    logger.info("Drop Index Timed Out")
                    result = False
                    break
                loopTimes += 1
                print('Aerospike: Waiting on Index Drop [%d]\r'%loopTimes, end="")
                await asyncio.sleep(1)                       
                existingIndexes = await adminClient.index_list()

        t = time.time()
        print(f'\nAerospike: Result: {result}, Drop Index Time (sec) = {t - s}, Time: {time.strftime("%Y-%m-%d %H:%M:%S")}')
        logger.info("Drop Index Completed")
        return result
    
    async def CreateIndex(self, adminClient: vectorASyncAdminClient) -> None:
        global _AerospikeIdxNames
        print(f'Aerospike: Creating Index {self._namespace}.{self._idx_name}')
        logger.info(f'Creating Index {self._namespace}.{self._idx_name}')
        s = time.time()
        await adminClient.index_create(namespace=self._namespace,
                                                name=self._idx_name,
                                                sets=self._setName,
                                                vector_field=self._idx_binName,
                                                dimensions=self._dims,
                                                index_params= self._idx_hnswparams,
                                                vector_distance_metric=self._idx_value
                                                )            
        t = time.time()
        print(f'Aerospike: Index Creation Time (sec) = {t - s}, Time: {time.strftime("%Y-%m-%d %H:%M:%S")}')
        logger.info("Index Created")
        _AerospikeIdxNames.append(self._idx_name)
                        
    async def PutVector(self, key: int, embedding, i: int, client: vectorASyncClient) -> None:
        try:
            await client.put(namespace=self._namespace,
                                set_name=self._setName,
                                key=key,
                                record_data={
                                    self._idx_binName:embedding.tolist(),
                                    self._idx_binKeyName:key
                                }
            )
        except Exception as e:
            print(f'\n** Count: {i} Key: {key} Exception: "{e}" **\r\n')
            logger.exception(f"Put Failure on Count: {i} Key: {key}")       
            raise e
        
    async def fitAsync(self, X: np.array) -> None:
        global _AerospikeIdxNames
        
        if X.dtype != np.float32:
            X = X.astype(np.float32)
                
        print(f'Aerospike: fitAsync: {self} Shape: {X.shape}')
        
        populateIdx = True
            
        async with vectorASyncAdminClient(
                seeds=vectorTypes.HostPort(host=self._host, port=self._port, is_tls=self._verifyTLS),
                listener_name=self._listern
            ) as adminClient:

            #If exists, no sense to try creation...
            existingIndexes = await adminClient.index_list()
            if(any(index["id"]["namespace"] == self._namespace
                                    and index["id"]["name"] == self._idx_name 
                            for index in existingIndexes)):
                print(f'Aerospike: Index {self._namespace}.{self._idx_name} Already Exists')
                logger.info(f"Index {self._namespace}.{self._idx_name} Already Exists")
                
                #since this can be an external DB (not in a container), we need to clean up from prior runs
                #if the index name is in this list, we know it was created in this run group and don't need to drop the index.
                #If it is a fresh run, this list will not contain the index and we know it needs to be dropped.
                if self._idx_name in _AerospikeIdxNames:
                    print(f'Aerospike: Index {self._namespace}.{self._idx_name} being reused (not re-populated)')
                    logger.info(f'Index {self._namespace}.{self._idx_name} being reused (not re-populated)')
                    populateIdx = False
                elif self._idx_drop:
                    if await self.DropIndex(adminClient):
                        await self.CreateIndex(adminClient)
                    else:
                        populateIdx = False            
            else:
                await self.CreateIndex(adminClient)
                
        if populateIdx:
            print(f'Aerospike: Populating Index {self._namespace}.{self._idx_name}')
            logger.info(f'Populating Index {self._namespace}.{self._idx_name}')
            async with vectorASyncClient(seeds=vectorTypes.HostPort(host=self._host, port=self._port, is_tls=self._verifyTLS),
                                            listener_name=self._listern
                        ) as client:
                s = time.time()
                taskPuts = []
                i = 0
                #async with asyncio. as tg: #only in 3.11
                for key, embedding in enumerate(X):
                    i += 1                    
                    if self._populateTasks == 1:
                        await self.PutVector(key, embedding, i, client)
                    elif self._populateTasks < 1:
                        taskPuts.append(self.PutVector(key, embedding, i, client))
                    else:
                        taskPuts.append(self.PutVector(key, embedding, i, client))
                        if len(taskPuts) >= self._populateTasks:
                            logger.debug(f"Waiting for Put Tasks ({len(taskPuts)}) to Complete at {i}")
                            await asyncio.gather(*taskPuts)
                            logger.debug(f"Put Tasks Completed")
                            taskPuts.clear()
                                                                                 
                    if not self._indocker:
                        print('Aerospike: Index Put Counter [%d]\r'%i, end="")
                logger.debug(f"Waiting for Put Tasks (finial {len(taskPuts)}) to Complete at {i}")                            
                await asyncio.gather(*taskPuts)
                logger.debug(f"All Put Tasks Completed")
                t = time.time()
                print(f"\nAerospike: Index Put {i:,} Recs in {t - s} (secs)")
                logger.info(f'Index Put Records {i,}')
                print("Aerospike: waiting for indexing to complete")            
                await client.wait_for_index_completion(namespace=self._namespace,
                                                        name=self._idx_name)            
                t = time.time()
                print(f"Aerospike: Index Total Populating Time (sec) = {t - s}")
                logger.info(f'Populating Index Completed')
            
    def fit(self, X: np.array) -> None:              
        
        if self._actions == OperationActions.QUERYONLY:
            print(f'Aerospike: No Idx Population: {self} Shape: {X.shape} Time: {time.strftime("%Y-%m-%d %H:%M:%S")}')
            logger.info(f'No Idx Population: {self} Shape: {X.shape}')
            return
        
        print(f'Aerospike: Start fit: {self} Shape: {X.shape} Time: {time.strftime("%Y-%m-%d %H:%M:%S")}')
        logger.info(f'Start fit: {self} Shape: {X.shape}')
                
        asyncio.run(self.fitAsync(X))
        
        print(f'Aerospike: End fit: {self} Shape: {X.shape} Time: {time.strftime("%Y-%m-%d %H:%M:%S")}')
        logger.info(f"End fit: {self} Shape: {X.shape}")
        
    def set_query_arguments(self, hnswParams: dict = None):
        if self._actions == OperationActions.IDXPOPULATEONLY:
            print(f'Aerospike: No Query: Time: {time.strftime("%Y-%m-%d %H:%M:%S")}')
            logger.info("No Query")
        else:
            print(f'Aerospike: Set Query: Time: {time.strftime("%Y-%m-%d %H:%M:%S")}')
            logger.info("Set Query")
            if hnswParams is not None and len(hnswParams) > 0:
                self._query_hnswsearchparams = Aerospike.SetHnswParamsAttrs(
                                                        vectorTypes.HnswSearchParams(),
                                                        hnswParams
                                                    )
          
    def query(self, q, n):
        if self._actions == OperationActions.IDXPOPULATEONLY:
            return []
        else:
            result = self._syncClient.vector_search(namespace=self._namespace,
                                                    index_name=self._idx_name,
                                                    query=q.tolist(),
                                                    limit=n,
                                                    search_params=self._query_hnswsearchparams,
                                                    bin_names=[self._idx_binKeyName])
            result_ids = [neighbor.bins[self._idx_binKeyName] for neighbor in result]
            return result_ids                
        
    #def get_batch_results(self):
    #    return self.batch_results

    #def get_batch_latencies(self):
    #    return self.batch_latencies

    def __str__(self):
        batchingparams = f"maxrecs:{self._idx_hnswparams.batching_params.max_records}, interval:{self._idx_hnswparams.batching_params.interval}, disabled:{self._idx_hnswparams.batching_params.disabled}"
        hnswparams = f"m:{self._idx_hnswparams.m}, efconst:{self._idx_hnswparams.ef_construction}, ef:{self._idx_hnswparams.ef}, batching:{{{batchingparams}}}"
        return f"Aerospike([{self._metric}, {self._host}, {self._port}, {self._namespace}, {self._setName}, {self._idx_name}, {self._idx_type}, {self._idx_value}, {self._dims}, {self._actions}, {self._idx_sleep}, {self._populateTasks} {{{hnswparams}}}, {{{self._query_hnswsearchparams}}}])"
