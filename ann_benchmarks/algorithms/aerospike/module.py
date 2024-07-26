import asyncio
import os
import numpy as np
import time
import enum
import os
import logging

from typing import Iterable, List, Any
from pythonping import ping as PingHost
from importlib.metadata import version

from aerospike_vector_search import types as vectorTypes, Client as vectorSyncClient
from aerospike_vector_search.aio import AdminClient as vectorASyncAdminClient, Client as vectorASyncClient
from aerospike_vector_search.shared.proto_generated.types_pb2_grpc import grpc  as vectorResultCodes

from ..base.module import BaseANN

loggerASClient = logging.getLogger("aerospike_vector_search")
logger = logging.getLogger(__name__)
logFileHandler = None

aerospikeIdxNames : list = []

loggingEnabled : bool = False

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
                    uniqueSetIdxName: bool = True,
                    dropIdx: bool = True,
                    actions: str = "ALLOPS"):
        
        global logFileHandler
        global loggingEnabled
        
        asLogFile = os.environ.get("APP_LOGFILE")
        self._asLogLevel = os.environ.get("AVS_LOGLEVEL")
        self._logLevel = os.environ.get("APP_LOGLEVEL") or "INFO"
        self._indocker = Aerospike.InDocker()
        
        if self._indocker:
            print("Aerospike: Running In Docker Container")
        elif asLogFile is None:
                asLogFile = "AerospikeANN.log"
        
        if not self._indocker and asLogFile is not None and asLogFile and self._logLevel != "NOTSET":
            print(f"Aerospike: Logging to file {os.getcwd()}/{asLogFile}")
            if logFileHandler is None:
                logFileHandler = logging.FileHandler(asLogFile, "w")                
                logFormatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                logFileHandler.setFormatter(logFormatter)
                if self._asLogLevel is not None and self._asLogLevel:
                    loggerASClient.addHandler(logFileHandler)
                    loggerASClient.setLevel(logging.getLevelName(self._asLogLevel))
                logger.addHandler(logFileHandler)            
                logger.setLevel(logging.getLevelName(self._logLevel))
            self._logFileHandler = logFileHandler
            loggingEnabled = True
            logger.info(f'Start Aerospike ANN Client: Metric: {metric}, Dimension: {dimension}')
            logger.info(f"  aerospike-vector-search: {version('aerospike_vector_search')}")
            
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
        dropIdxOverride = os.environ.get("APP_DROP_IDX")
        if dropIdxOverride is not None:
            self._idx_drop = dropIdxOverride.lower() in ['true', '1', 't']
        
        #self._username = os.environ.get("APP_USERNAME") or ""
        #self._password = os.environ.get("APP_PASSWORD") or ""
        self._host = os.environ.get("AVS_HOST") or "localhost"
        self._port = int(os.environ.get("AVS_PORT") or 5000)
        self._listern = None #os.environ.get("AVS_ADVERTISED_LISTENER") or None          
        self._isloadbalancer = os.environ.get("AVS_USELOADBALANCER")
        if self._isloadbalancer is not None and self._isloadbalancer.lower() in ['true', '1', 't', '']:
            self._isloadbalancer = True
        else:
            self._isloadbalancer = False
        
        self._namespace = os.environ.get("AVS_NAMESPACE") or "test"
        self._setName = os.environ.get("AVS_SET") or "ANN-data"
        
        if self._idx_type.casefold() == self._metric.casefold():
            setNameType = self._idx_type
        else:
            setNameType = f'{self._metric}_{self._idx_type}'
            
        if not uniqueSetIdxName or self._idx_hnswparams is None:
            self._setName = f'{self._setName}_{setNameType}'            
        else:
            self._setName = f'{self._setName}_{setNameType}_{self._dims}_{self._idx_hnswparams.m}_{self._idx_hnswparams.ef_construction}_{self._idx_hnswparams.ef}'
        self._idx_name = f'{self._setName}_Idx'
        
        self._idx_sleep = int(os.environ.get("APP_INDEX_SLEEP") or 0)
        self._populateTasks = int(os.environ.get("APP_POPULATE_TASKS") or 5000)
        pingAVS = os.environ.get("APP_PINGAVS")
        if pingAVS is None or pingAVS.lower() in ['false', '0', 'f', '']:
            pingAVS = False
        else:
            pingAVS = True
        self._checkResult = os.environ.get("APP_CHECKRESULT")
        if self._checkResult is None or self._checkResult.lower() in ['true', '1', 't', '']:
            self._checkResult = not self._indocker
        else:
            self._checkResult = False            
        self._idx_binName = "ANN_embedding"
        self._query_hnswsearchparams = None
        
        if pingAVS:
            print(f'Aerospike: Trying Ping to {self._host} {self._listern}')
            pingresult = PingHost(self._host, verbose=True)
            print(pingresult)
            logger.info(pingresult)
                    
        Aerospike.PrintLog('Try Create Sync Client')
        self._syncClient = vectorSyncClient(
                                    seeds=vectorTypes.HostPort(host=self._host,
                                                            port=self._port), 
                                    listener_name=self._listern,
                                    is_loadbalancer=self._isloadbalancer)
        
        Aerospike.PrintLog(f'init completed: {self}')        
        
    @staticmethod
    def InDocker() -> bool:
        """ Returns: True if running in a Docker container, else False """
        with open('/proc/1/cgroup', 'rt') as ifh:
            return 'docker' in ifh.read()
        
    @staticmethod
    def set_hnsw_params_attrs(__obj :object, __dict: dict) -> object:
        for key in __dict: 
            if key == 'batching_params':
                setattr(
                    __obj,
                    key,
                    Aerospike.set_hnsw_params_attrs(
                            vectorTypes.HnswBatchingParams(),
                            __dict[key],
                    )
                )
            elif key == 'caching_params':
                setattr(
                    __obj,
                    key,
                    Aerospike.set_hnsw_params_attrs(
                            vectorTypes.HnswCachingParams(),
                            __dict[key],
                    )
                )
            elif key == 'healer_params':
                setattr(
                    __obj,
                    key,
                    Aerospike.set_hnsw_params_attrs(
                            vectorTypes.HnswHealerParams(),
                            __dict[key],
                    )
                )
            elif key == 'merge_params':
                setattr(
                    __obj,
                    key,
                    Aerospike.set_hnsw_params_attrs(
                            vectorTypes.HnswIndexMergeParams(),
                            __dict[key],
                    )
                )
            else:
                setattr(__obj, key, __dict[key])
        return __obj
    
    @staticmethod
    def FlushLog() -> None:
        if(logger.handlers is not None):
            for handler in logger.handlers:
                handler.flush()                
          
    @staticmethod
    def PrintLog(msg :str, logLevel :int = logging.INFO) -> None:
        if loggingEnabled:
            logger.log(level=logLevel, msg=msg)
        else:
            levelName = "" if logLevel == logging.INFO else f" {logging.getLevelName(logLevel)}: "
            print("Aerospike: " + levelName + msg + f', Time: {time.strftime("%Y-%m-%d %H:%M:%S")}')                    

    def done(self) -> None:
        """Clean up BaseANN once it is finished being used."""
        Aerospike.PrintLog(f'done: {self}')
        
        if self._syncClient is not None:
            clientCloseTask = self._syncClient.close()
        Aerospike.FlushLog()
                    
    async def DropIndex(self, adminClient: vectorASyncAdminClient) -> bool:
        Aerospike.PrintLog(f'Dropping Index {self._namespace}.{self._idx_name}')
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
                    print(f'\n')
                    Aerospike.PrintLog("Drop Index Timed Out!", logging.WARNING)
                    result = False
                    break
                loopTimes += 1
                print('Aerospike: Waiting on Index Drop [%d]\r'%loopTimes, end="")
                await asyncio.sleep(1)                       
                existingIndexes = await adminClient.index_list()

        t = time.time()
        print('\n')
        Aerospike.PrintLog(f'Result: {result}, Drop Index Time (sec) = {t - s}')        
        return result
    
    async def CreateIndex(self, adminClient: vectorASyncAdminClient) -> None:
        global aerospikeIdxNames
        Aerospike.PrintLog(f'Creating Index {self._namespace}.{self._idx_name}')        
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
        Aerospike.PrintLog(f'Index Creation Time (sec) = {t - s}')        
        aerospikeIdxNames.append(self._idx_name)
                        
    async def PutVector(self, key: int, embedding, i: int, client: vectorASyncClient, retry: bool = False) -> None:
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
                if not retry and avse.rpc_error.code() == vectorResultCodes.StatusCode.RESOURCE_EXHAUSTED:
                    logLevel = logging.DEBUG
                    if not self._puasePuts:
                        self._puasePuts = True
                        logLevel = logging.WARNING
                        Aerospike.PrintLog(msg=f"\nResource Exhausted on Put first encounter on Count: {i}, Key: {key}, Idx: {self._namespace}.{self._setName}.{self._idx_name}. Going to Pause Population and Wait for Idx Completion...",
                                            logLevel=logging.WARNING)
                    else:
                        logger.debug(f"Resource Exhausted on Put on Count: {i}, Key: {key}, Idx: {self._namespace}.{self._setName}.{self._idx_name}. Going to Pause Population and Wait for Idx Completion...")
                    s = time.time()                    
                    await client.wait_for_index_completion(namespace=self._namespace,
                                                            name=self._idx_name)            
                    t = time.time()
                    if logLevel == logging.WARNING:
                        Aerospike.PrintLog(msg=f"Index Completed Time (sec) = {t - s}, Going to Reissue Puts for Idx: {self._namespace}.{self._setName}.{self._idx_name}",
                                            logLevel=logging.WARNING)
                    else:
                        logger.debug(msg=f"Index Completed Time (sec) = {t - s}, Going to Reissue Puts for Count: {i}, Key: {key}, Idx: {self._namespace}.{self._setName}.{self._idx_name}")
                        
                    await self.PutVector(key, embedding, i, client, True)
                    self._puasePuts = False
                    
                    if logLevel == logging.WARNING:
                        Aerospike.PrintLog(msg=f"Resuming population for Idx: {self._namespace}.{self._setName}.{self._idx_name}",
                                            logLevel=logging.WARNING)
                    else:
                        logger.debug(msg=f"Resuming population for Count: {i}, Key: {key}, Idx: {self._namespace}.{self._setName}.{self._idx_name}")                        
                else:
                    raise avse
        except Exception as e:
            print(f'\n** Count: {i} Key: {key} Exception: "{e}" **\r\n')
            logger.exception(f"Put Failure on Count: {i}, Key: {key}, Idx: {self._namespace}.{self._setName}.{self._idx_name}, Retry: {retry}")
            Aerospike.FlushLog()
            raise e
        
    async def fitAsync(self, X: np.array) -> None:
        global aerospikeIdxNames
        
        if X.dtype != np.float32:
            X = X.astype(np.float32)
    
        Aerospike.PrintLog(f'fitAsync: {self} Shape: {X.shape}')
              
        populateIdx = True
            
        async with vectorASyncAdminClient(
                seeds=vectorTypes.HostPort(host=self._host, port=self._port),
                listener_name=self._listern,
                is_loadbalancer=self._isloadbalancer
            ) as adminClient:

            #If exists, no sense to try creation...
            existingIndexes = await adminClient.index_list()
            if(any(index["id"]["namespace"] == self._namespace
                                    and index["id"]["name"] == self._idx_name 
                            for index in existingIndexes)):
                Aerospike.PrintLog(f'Index {self._namespace}.{self._idx_name} Already Exists')
                
                #since this can be an external DB (not in a container), we need to clean up from prior runs
                #if the index name is in this list, we know it was created in this run group and don't need to drop the index.
                #If it is a fresh run, this list will not contain the index and we know it needs to be dropped.
                if self._idx_name in aerospikeIdxNames:
                    Aerospike.PrintLog(f'Index {self._namespace}.{self._idx_name} being reused (not re-populated)')
                    populateIdx = False
                elif self._idx_drop:
                    if await self.DropIndex(adminClient):
                        await self.CreateIndex(adminClient)
                    else:
                        populateIdx = False            
            else:
                await self.CreateIndex(adminClient)
                
        if populateIdx:
            self._puasePuts = False
            Aerospike.PrintLog(f'Populating Index {self._namespace}.{self._idx_name}')
            async with vectorASyncClient(seeds=vectorTypes.HostPort(host=self._host, port=self._port, is_tls=self._verifyTLS),
                                            listener_name=self._listern,
                                            is_loadbalancer=self._isloadbalancer
                        ) as client:
                s = time.time()
                taskPuts = []
                i = 0
                #async with asyncio. as tg: #only in 3.11
                for key, embedding in enumerate(X):
                    if self._puasePuts:
                        loopTimes = 0
                        print('\n')
                        while (self._puasePuts):
                            if loopTimes % 30 == 0:
                                Aerospike.PrintLog(f"Paused Population still waiting for Idx Completion at {loopTimes} mins!", logging.WARNING)                                
                            loopTimes += 1
                            logger.debug(f"Putting Paused {loopTimes}")
                            await asyncio.sleep(60)
                        Aerospike.PrintLog(f"Resuming Population at {loopTimes} mins", logging.WARNING)
                        
                    i += 1
                    if self._populateTasks < 0:
                        taskPuts.append(self.PutVector(key, embedding, i, client))
                    elif self._populateTasks <= 1:
                        await self.PutVector(key, embedding, i, client)                    
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
                t = time.time()
                logger.info(f"All Put Tasks Completed")                
                print('\n')
                Aerospike.PrintLog(f"Index Put {i:,} Recs in {t - s} (secs)")
                Aerospike.PrintLog("waiting for indexing to complete")            
                await client.wait_for_index_completion(namespace=self._namespace,
                                                        name=self._idx_name)            
                t = time.time()
                Aerospike.PrintLog(f"Index Total Populating Time and Idx Completed (sec) = {t - s}")
                
    def fit(self, X: np.array) -> None:              
        
        if self._actions == OperationActions.QUERYONLY:
            Aerospike.PrintLog(f'No Idx Population: {self} Shape: {X.shape}')            
            return
        
        Aerospike.PrintLog(f'Start fit: {self} Shape: {X.shape}')
                
        asyncio.run(self.fitAsync(X))
        
        Aerospike.PrintLog(f'End fit: {self} Shape: {X.shape}')
        Aerospike.FlushLog()
        
    def set_query_arguments(self, hnswParams: dict = None):
        if self._actions == OperationActions.IDXPOPULATEONLY:
            Aerospike.PrintLog(f'No Query')            
        elif hnswParams is not None and len(hnswParams) > 0:
            self._query_hnswsearchparams = Aerospike.SetHnswParamsAttrs(
                                                    vectorTypes.HnswSearchParams(),
                                                    hnswParams
                                                )
            if hnswParams is None:
                Aerospike.PrintLog(f'Set Query {self}')
            else:    
                Aerospike.PrintLog(f'Set Query {self}, ef: {hnswParams["ef"]}')
        else:
            Aerospike.PrintLog(f"Set Query {self}")
          
    def query(self, q, n):
        if self._actions == OperationActions.IDXPOPULATEONLY:
            return []
        else:
            result = self._syncClient.vector_search(namespace=self._namespace,
                                                    index_name=self._idx_name,
                                                    query=q.tolist(),
                                                    limit=n,
                                                    search_params=self._query_hnswsearchparams)            
            result_ids = [neighbor.key.key for neighbor in result]
            if self._checkResult:
                if len(result_ids) == 0:
                    Aerospike.PrintLog(f'No Query Results for {self._idx_name}', logging.WARNING)                    
                zeroDist = [record.key.key for record in result if record.distance == 0]
                if len(zeroDist) > 0:
                    Aerospike.PrintLog(f'Zero Distance Found for {self._idx_name} Keys: {zeroDist}', logging.WARNING)
                    
            return result_ids                
        
    #def get_batch_results(self):
    #    return self.batch_results

    #def get_batch_latencies(self):
    #    return self.batch_latencies

    def __str__(self):
        batchingparams = f"maxrecs:{self._idx_hnswparams.batching_params.max_records}, interval:{self._idx_hnswparams.batching_params.interval}"
        hnswparams = f"m:{self._idx_hnswparams.m}, efconst:{self._idx_hnswparams.ef_construction}, ef:{self._idx_hnswparams.ef}, batching:{{{batchingparams}}}"
        return f"Aerospike([{self._metric}, {self._host}:{self._port}, {self._isloadbalancer}, {self._namespace}.{self._setName}.{self._idx_name}, {self._idx_type}, {self._idx_value}, {self._dims}, {self._actions}, {self._idx_sleep}, {self._populateTasks}, {{{hnswparams}}}, {{{self._query_hnswsearchparams}}}])"
