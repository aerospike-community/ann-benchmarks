import asyncio
import os
import numpy as np
import time
import enum
import logging

from typing import Dict, Any, Union # Iterable, List, Any
from pythonping import ping as PingHost
from importlib.metadata import version

from aerospike_vector_search import types as vectorTypes, Client as vectorSyncClient
from aerospike_vector_search.aio import Client as vectorASyncClient
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
                    actions: str = "ALLOPS",
                    ignoreExhaustedEvent: bool = False):

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
                level_mapping = {
                        "DEBUG": logging.DEBUG,
                        "INFO": logging.INFO,
                        "WARNING": logging.WARNING,
                        "ERROR": logging.ERROR,
                        "CRITICAL": logging.CRITICAL,
                }
                logFileHandler = logging.FileHandler(asLogFile, "w")
                logFormatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                logFileHandler.setFormatter(logFormatter)
                if self._asLogLevel is not None and self._asLogLevel:
                    loggerASClient.addHandler(logFileHandler)
                    loggerASClient.setLevel(level_mapping[self._asLogLevel.upper()])
                logger.addHandler(logFileHandler)
                logger.setLevel(level_mapping[self._logLevel.upper()])
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
        self._idx_ignoreExhEvt = ignoreExhaustedEvent
        dropIdxOverride = os.environ.get("APP_DROP_IDX")
        if dropIdxOverride is not None:
            self._idx_drop = dropIdxOverride.lower() in ['true', '1', 't']

        self._username = os.environ.get("APP_USERNAME")
        self._password = os.environ.get("APP_PASSWORD")
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

        Aerospike.PrintLog('Try Create Query Client')
        self._queryClient = vectorSyncClient(
                                    seeds=vectorTypes.HostPort(host=self._host,
                                                            port=self._port),
                                    listener_name=self._listern,
                                    is_loadbalancer=self._isloadbalancer,
                                    username=self._username,
                                    password=self._password)

        self._population_tot_time_sec:float = 0
        self._idx_completion_time_sec:float = 0
        self._upserted_time_sec:float = 0
        self._upserted_vectors:int = 0
        self._train_shape:Union[tuple,None] = None
        self._query_no_results:int = 0
        self._query_no_neighbors:int = 0
        self._idx_Definition:Union[None,vectorTypes.IndexDefinition] = None

        Aerospike.PrintLog(f'init completed: {self}')

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
                            __dict[key],
                    )
                )
            elif key == 'index_caching_params' or key == 'caching_params':
                setattr(
                    __obj,
                    key,
                    Aerospike.SetHnswParamsAttrs(
                            vectorTypes.HnswCachingParams(),
                            __dict[key],
                    )
                )
            elif key == 'healer_params':
                setattr(
                    __obj,
                    key,
                    Aerospike.SetHnswParamsAttrs(
                            vectorTypes.HnswHealerParams(),
                            __dict[key],
                    )
                )
            elif key == 'merge_params':
                setattr(
                    __obj,
                    key,
                    Aerospike.SetHnswParamsAttrs(
                            vectorTypes.HnswIndexMergeParams(),
                            __dict[key],
                    )
                )
            elif key == 'record_caching_params':
                setattr(
                    __obj,
                    key,
                    Aerospike.SetHnswParamsAttrs(
                            vectorTypes.HnswCachingParams(),
                            __dict[key],
                    )
                )
            elif (type(__dict[key]) is str
                    and (__dict[key].lower() == "none"
                        or __dict[key].lower() == "null")):
                setattr(__obj, key, None)
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

        if self._queryClient is not None:
            self._queryClient.close()
        Aerospike.FlushLog()

    async def DropIndex(self, client: vectorASyncClient) -> bool:
        Aerospike.PrintLog(f'Dropping Index {self._namespace}.{self._idx_name}')
        result = False
        s = time.time()

        await client.index_drop(namespace=self._namespace,
                                            name=self._idx_name)

        existingIndexes = await client.index_list()
        loopTimes = 0
        result = True
        if self._idx_sleep > 0:
            while (any(index["id"]["namespace"] == self._namespace
                                    and index["id"]["name"] == self._idx_name
                                for index in existingIndexes)):
                if loopTimes>= self._idx_sleep:
                    print('\n')
                    Aerospike.PrintLog("Drop Index Timed Out!", logging.WARNING)
                    result = False
                    break
                loopTimes += 1
                print('Aerospike: Waiting on Index Drop [%d]\r'%loopTimes, end="")
                await asyncio.sleep(1)
                existingIndexes = await client.index_list()

        t = time.time()
        print('\n')
        Aerospike.PrintLog(f'Result: {result}, Drop Index Time (sec) = {t - s}')
        return result

    async def CreateIndex(self, client: vectorASyncClient) -> None:
        global aerospikeIdxNames
        Aerospike.PrintLog(f'Creating Index {self._namespace}.{self._idx_name}')
        s = time.time()
        await client.index_create(namespace=self._namespace,
                                                name=self._idx_name,
                                                sets=self._setName,
                                                vector_field=self._idx_binName,
                                                dimensions=self._dims,
                                                index_params= self._idx_hnswparams,
                                                vector_distance_metric=self._idx_value,
                                                index_labels={"Benchmark":"ANN",
                                                              "ANN_Train_Shape":self._train_shape.__str__()}
                                                )
        t = time.time()
        Aerospike.PrintLog(f'Index Creation Time (sec) = {t - s}')
        aerospikeIdxNames.append(self._idx_name)

    async def WaitForIndexing(self, client: vectorASyncClient):
        Aerospike.PrintLog("waiting for indexing to complete")
        idxParams = await client.index_get(namespace=self._namespace,
                                            name=self._idx_name)
        s = time.time()
        index = await client.index(namespace=self._namespace,
                                    name=self._idx_name)
        vertices = 0
        unmerged_recs = 0
        i = 1
        await asyncio.sleep(1)
        try:
            # Wait for the index to have Vertices and no unmerged records
            while vertices == 0 or unmerged_recs > 0:
                status = await index.status()
                vertices = status.index_healer_vertices_valid
                unmerged_recs = status.unmerged_record_count
                if not self._indocker:
                    print('Aerospike: Secs %d -- Unmerged Idx recs: %d Vertices Idx Valid (healer): %d            \r'%(i,unmerged_recs,vertices), end="")
                if vertices > 0 and unmerged_recs == 0:
                    break
                if unmerged_recs == 0 and vertices == 0:
                    await client.index_update(namespace=self._namespace,
                                                name=self._idx_name,
                                                hnsw_update_params=vectorTypes.HnswIndexUpdate(healer_params=vectorTypes.HnswHealerParams(schedule="* * * * * ?")))
                await asyncio.sleep(1)
                i += 1
            t = time.time()
            print('\n')
            self._idx_completion_time_sec = t - s
            Aerospike.PrintLog(f"Index Completion Time (sec) = {self._idx_completion_time_sec} Vertices Idx Valid (healer) = {vertices}")
        finally:
            await client.index_update(namespace=self._namespace,
                                            name=self._idx_name,
                                            hnsw_update_params=vectorTypes.HnswIndexUpdate(healer_params=idxParams.hnsw_params.healer_params))
            await asyncio.sleep(0.1)

    async def PutVector(self, key: int, embedding, i: int, client: vectorASyncClient, retry: bool = False) -> None:
        try:
            try:
                await client.upsert(namespace=self._namespace,
                                    set_name=self._setName,
                                    key=key,
                                    ignore_mem_queue_full=self._idx_ignoreExhEvt,
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
        self._train_shape = X.shape

        populateIdx = True

        async with vectorASyncClient(seeds=vectorTypes.HostPort(host=self._host, port=self._port),
                                            listener_name=self._listern,
                                            is_loadbalancer=self._isloadbalancer,
                                            username=self._username,
                                            password=self._password
                        ) as client:

            #If exists, no sense to try creation...
            existingIndexes = await client.index_list()
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
                    if await self.DropIndex(client):
                        await self.CreateIndex(client)
                    else:
                        populateIdx = False
            else:
                await self.CreateIndex(client)

            if populateIdx:
                self._puasePuts = False
                Aerospike.PrintLog(f'Populating Index {self._namespace}.{self._idx_name}')
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
                            logger.debug("Put Tasks Completed")
                            taskPuts.clear()

                    if not self._indocker:
                        print('Aerospike: Index Put Counter [%d]\r'%i, end="")

                logger.debug(f"Waiting for Put Tasks (finial {len(taskPuts)}) to Complete at {i}")
                await asyncio.gather(*taskPuts)
                t = time.time()
                logger.info("All Put Tasks Completed")
                print('\n')
                self._upserted_time_sec = t - s
                self._upserted_vectors = i
                Aerospike.PrintLog(f"Index Put {i:,} Recs in {self._upserted_time_sec} (secs)")
                await self.WaitForIndexing(client)
                t = time.time()
                self._population_tot_time_sec = t - s

                Aerospike.PrintLog(f"Index Total Populating Time and Idx Completed (sec) = {self._population_tot_time_sec}")
                Aerospike.PrintLog(f"Checking Existance of Index {self._idx_name}")
                self._idx_Definition = await client.index_get(namespace=self._namespace,
                                                                name=self._idx_name)
                Aerospike.PrintLog("\tCompleted")

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
            Aerospike.PrintLog('No Query')
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
            result = self._queryClient.vector_search(namespace=self._namespace,
                                                    index_name=self._idx_name,
                                                    query=q.tolist(),
                                                    limit=n,
                                                    search_params=self._query_hnswsearchparams)
            result_ids = [neighbor.key.key for neighbor in result]
            if self._checkResult:
                if len(result_ids) == 0:
                    Aerospike.PrintLog(f'No Query Results for {self._idx_name}', logging.WARNING)
                    self._query_no_results += 1
                zeroDist = [record.key.key for record in result if record.distance == 0]
                if len(zeroDist) > 0:
                    Aerospike.PrintLog(f'Zero Distance Found for {self._idx_name} Keys: {zeroDist}', logging.WARNING)
                    self._query_no_neighbors += 1

            return result_ids

    def get_additional(self) -> Dict[str, Any]:
        """Returns additional attributes to be stored with the result.

        Returns:
            dict: A dictionary of additional attributes.
        """

        def jsonDefault(o) -> Any:

            if isinstance(o,enum.Enum):
                return o.__str__()

            if hasattr(o, '__dict__'):
                return {k:v for k, v in o.__dict__.items() if not k.startswith('_') and v is not None }

            return o.__str__()

        def getjson(obj) -> str:
            import json

            if obj is None:
                return '{}'

            return json.dumps(obj,
                              default=jsonDefault)

        return {"as_indockercontainer": self._indocker,
                "as_idx_name": self._idx_name,
                "as_idx_type": self._idx_type,
                "as_idx_binname": self._idx_binName,
                "as_idx_hnswparams": getjson(self._idx_hnswparams),
                "as_idx_drop": self._idx_drop,
                "as_idx_ignoreexhuseevents": self._idx_ignoreExhEvt,
                "as_idx_definition_built": getjson(self._idx_Definition),
                "as_actions": self._actions.__str__(),
                "as_host": self._host,
                "as_isloadbalancer": 'None' if self._isloadbalancer is None else self._isloadbalancer,
                "as_namespace": self._namespace,
                "as_set": self._setName,
                "as_train_shape": self._train_shape,
                "as_query_hnswsearchparams": getjson(self._query_hnswsearchparams),
                "as_query_checkresults": self._checkResult,
                "as_query_no_result_cnt": self._query_no_results,
                "as_query_no_neighbors_fnd": self._query_no_neighbors,
                "as_upserted_vectors": self._upserted_vectors,
                "as_upserted_time_secs": self._upserted_time_sec,
                "as_idx_completion_secs": self._idx_completion_time_sec,
                "as_total_population_time_secs": self._population_tot_time_sec
                }

    #def get_batch_results(self):
    #    return self.batch_results

    #def get_batch_latencies(self):
    #    return self.batch_latencies

    def __str__(self):
        batchingparams = f"maxidxrecs:{self._idx_hnswparams.batching_params.max_index_records}, interval:{self._idx_hnswparams.batching_params.index_interval}"
        healingparams = f"schedule:{self._idx_hnswparams.healer_params.schedule}, parallelism:{self._idx_hnswparams.healer_params.parallelism}"
        hnswparams = f"m:{self._idx_hnswparams.m}, efconst:{self._idx_hnswparams.ef_construction}, ef:{self._idx_hnswparams.ef}, batching:{{{batchingparams}}}, healer:{{{healingparams}}}"
        return f"Aerospike([{self._metric}, {self._host}:{self._port}, {self._isloadbalancer}, {self._namespace}.{self._setName}.{self._idx_name}, {self._idx_type}, {self._idx_value}, {self._dims}, {self._actions}, {self._idx_sleep}, {self._idx_ignoreExhEvt}, {self._populateTasks}, {{{hnswparams}}}, {{{self._query_hnswsearchparams}}}])"
