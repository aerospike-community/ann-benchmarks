import asyncio
import os
import numpy as np
import time
import logging
import argparse

from enum import Flag, auto
from typing import List, Dict, Union, OrderedDict
from importlib.metadata import version
from logging import _nameToLevel as LogLevels
from threading import Thread

from prometheus_client import start_http_server

from opentelemetry import metrics
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.sdk.metrics import MeterProvider, Meter
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.util.types import Attributes

from aerospike_vector_search import types as vectorTypes
from metrics import all_metrics as METRICS

_distanceNameToAerospikeType: Dict[str, vectorTypes.VectorDistanceMetric] = {
    'angular': vectorTypes.VectorDistanceMetric.COSINE,
    'euclidean': vectorTypes.VectorDistanceMetric.SQUARED_EUCLIDEAN,
    'hamming': vectorTypes.VectorDistanceMetric.HAMMING,
    'jaccard': None,
    'dot': vectorTypes.VectorDistanceMetric.DOT_PRODUCT,
}

_distanceAerospikeTypeToAnn: Dict[str, str] = {
    'COSINE' : 'angular',
    'SQUARED_EUCLIDEAN' : 'euclidean',
    'HAMMING' : 'hamming',    
    'DOT_PRODUCT' : 'dot',
    'jaccard' : 'jaccard'
}

loggerASClient = logging.getLogger("aerospike_vector_search")
logFileHandler = None

class OperationActions(Flag):    
    POPULATION = auto()
    QUERY = auto()
    POPQUERY = POPULATION | QUERY
    
class BaseAerospike(object):

    @staticmethod
    def parse_arguments(parser: argparse.ArgumentParser) -> None:
        '''
        Adds the arguments required for the BaseAerospike
        '''
        parser.add_argument(
            '-p', "--vectorport",
            metavar="port",
            type=int,
            help="The Vector DB Port",
            default=5000
        )
        parser.add_argument(
            '-a', "--host",
            metavar="HOST",
            help="the Vector DB Ip Address or Host Name",
            default="localhost",
        )
        parser.add_argument(
            '-A', '--hosts',
            metavar="HOST:PORT",            
            nargs='+',
            help="A list of host and optional ports. Example: 'hosta:5000' or 'hostb'",
            default=[],
        )
        parser.add_argument(
            '-l', "--vectorloadbalancer",            
            help="Use Vector's DB Load Balancer",
            action='store_true'
        )
        parser.add_argument(
            '-T', "--vectortls",            
            help="Use TLS to connect to the Vector DB Server",
            action='store_true'
        )                    
        parser.add_argument(
            '-L', "--logfile",
            metavar="LOG",
            help="The logging file path. Default is no logging to a file.",
            default=None,
        )
        parser.add_argument(
            "--loglevel",
            metavar="LEVEL",
            help="The Logging level",
            default="INFO",
            choices=LogLevels.keys(),
        )
        parser.add_argument(
            "--driverloglevel",
            metavar="LEVEL",           
            help="The Driver's Logging level",
            default="NOTSET",
            choices=LogLevels.keys(),
        )
        parser.add_argument(
            "--prometheus",
            metavar="PORT",           
            help="Prometheus Port",
            default=9464,
            type=int
        )
        parser.add_argument(
            "--prometheushb",
            metavar="SECS",           
            help="Prometheus Heart Beat in secs",
            default=5,
            type=int
        )
        parser.add_argument(
            "--exitdelay",
            metavar="wait",           
            help="upon exist application will sleep",
            default=20,
            type=int
        )
    
    def __init__(self, runtimeArgs: argparse.Namespace, logger: logging.Logger):
        '''
        Meters:
        aerospike.hdf.populate          -- Counter can be used for rate (records per sec)
        aerospike.hdf.query             -- Counter can be used for rate (queries per sec)
        aerospike.hdf.exception         -- Up/Down Counter Exceptions
        aerospike.hdf.waitidxcompletion -- Up/Down counter (inc one upon start of wait/dec on end of wait)
        aerospike.hdf.dropidxtime       -- drop idx latency        
        '''
                
        self._prometheus_init(runtimeArgs)
        
        self._port = runtimeArgs.vectorport
        self._verifyTLS = runtimeArgs.vectortls
        
        if runtimeArgs.hosts is None or len(runtimeArgs.hosts) == 0:            
            self._host = [vectorTypes.HostPort(host=runtimeArgs.host,port=self._port,is_tls=self._verifyTLS)]
        else:
            self._host = []
            for pos, host in enumerate(runtimeArgs.hosts):
                parts = host.split(':')
                if len(parts) == 1:
                    self._host.append(vectorTypes.HostPort(host=host,port=self._port,is_tls=self._verifyTLS))
                elif len(parts) == 2:
                    self._host.append(vectorTypes.HostPort(host=parts[0],port=parts[1],is_tls=self._verifyTLS))
                    
        self._listern = None          
        self._useloadbalancer = runtimeArgs.vectorloadbalancer        
        
        self._namespace : str = None
        self._idx_namespace : str = None        
        self._setName : str = None
        self._paramsetname = None
        self._idx_name : str = None
        self._idx_binName : str = None
        
        self._idx_distance = None
        self._ann_distance : str = None
        self._idx_hnswparams : vectorTypes.HnswParams = None
        self._query_hnswparams : vectorTypes.HnswSearchParams = None

        self._concurrency : int = None
        self._idx_nowait : bool = None
        self._idx_resource_event : int = None
        self._idx_resource_cnt : int = None
        self._idx_state : str = ''

        self._sleepexit = runtimeArgs.exitdelay
        self._actions : OperationActions = None
        self._waitidx : bool = None
        self._datasetname : str = None
        self._dimensions = None
        self._trainarray : Union[np.ndarray, List[np.ndarray]] = None
        self._queryarray : Union[np.ndarray, List[np.ndarray]] = None
        self._neighbors : Union[np.ndarray, List[np.ndarray]] = None
        self._pausedPuts : bool = False
        self._heartbeat_thread : Thread = None
        self._query_nbrlimit : int = None
        self._query_runs : int = None
        self._remainingrecs : int = None
        self._remainingquerynbrs : int = None
        self._query_current_run : int = None
        self._query_metric_value : float = None
        self._query_metric_big_value : float = None
        self._aerospike_metric_value : float = None
        self._query_metric : dict[str,any] = None
        self._canchecknbors : bool = False
        self._query_distancecalc : str = None
        
        self._logging_init(runtimeArgs, logger)
        
        self._heartbeat_stage = 0
        self._start_prometheus_heartbeat()        

    def _prometheus_init(self, runtimeArgs: argparse.Namespace) -> None:
        
        # Service name is required for most backends
        self._prometheus_resource = Resource(attributes={
            SERVICE_NAME: "aerospike.hdf"
        })

        # Start Prometheus client
        self._prometheus_http_server = start_http_server(port=runtimeArgs.prometheus, addr="0.0.0.0")
        # Initialize PrometheusMetricReader which pulls metrics from the SDK
        # on-demand to respond to scrape requests
        self._prometheus_metric_reader = PrometheusMetricReader()
        
        self._prometheus_meter_provider = MeterProvider(resource=self._prometheus_resource,
                                                            metric_readers=[self._prometheus_metric_reader])
        metrics.set_meter_provider(self._prometheus_meter_provider)
        
        self._meter:Meter = metrics.get_meter("aerospike.hdf", meter_provider=self._prometheus_meter_provider)
        
        self._populate_counter = self._meter.create_counter("aerospike.hdf.populate", 
                                                        unit="1",
                                                        description="Cnts the recs upserted into the vector idx"
                                                      )        
        self._query_counter = self._meter.create_counter("aerospike.hdf.query", 
                                                        unit="1",
                                                        description="Cnts the nbr of queries performed"
                                                      )        
        self._query_histogram = self._meter.create_histogram("aerospike.hdf.queryhist",
                                                                unit="ms",
                                                                description="The amount of time it took one vector search to complete")        
        self._exception_counter = self._meter.create_up_down_counter("aerospike.hdf.exception", 
                                                                unit="1",
                                                                description="Cnts the nbr exceptions"
                                                      )
        self._waitidx_counter = self._meter.create_up_down_counter("aerospike.hdf.waitidxcompletion",
                                                        unit="1",
                                                        description="Waiting Idx completions")
        self._dropidx_histogram = self._meter.create_histogram("aerospike.hdf.dropidxtime",
                                                            unit="sec",
                                                            description="The amount of time it took t drop the idx")
        
        self._prometheus_heartbeat_gauge = self._meter.create_gauge("aerospike.hdf.heartbeat")
        
        self._prometheus_hb : int = runtimeArgs.prometheushb
        
        self._heartbeat_current_stage : int = -1

    def _logging_init(self, runtimeArgs: argparse.Namespace, logger: logging.Logger) -> None:
       
        global logFileHandler
        
        self._logFilePath = runtimeArgs.logfile        
        self._asLogLevel = runtimeArgs.driverloglevel
        self._logLevel = runtimeArgs.loglevel
        self._logger = logger
        self._loggingEnabled = False
           
        if self._logFilePath is not None and self._logFilePath and self._logLevel != "NOTSET":
            print(f"Logging to file {self._logFilePath}")
            if logFileHandler is None:
                logFileHandler = logging.FileHandler(self._logFilePath, "w")                
                logFormatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                logFileHandler.setFormatter(logFormatter)
                if self._asLogLevel is not None:
                    loggerASClient.addHandler(logFileHandler)
                    loggerASClient.setLevel(logging.getLevelName(self._asLogLevel))
                self._logger.addHandler(logFileHandler)            
                self._logger.setLevel(logging.getLevelName(self._logLevel))
            self._logFileHandler = logFileHandler
            self._loggingEnabled = True
            self._logger.info(f'Start Aerospike: Metric: {self.basestring()}')
            self._logger.info(f"  aerospike-vector-search: {version('aerospike_vector_search')}")
            self._logger.info(f"Prometheus HTTP Server: {self._prometheus_http_server[0].server_address}")
            self._logger.info(f"  Metrics Name: {self._meter.name}")
            self._logger.info(f"Arguments: {runtimeArgs}")
        elif self._asLogLevel is not None:            
            if self._asLogLevel == "NOTSET":                
                loggerASClient.setLevel(logging.CRITICAL)
            else:
                loggerASClient.setLevel(logging.getLevelName(self._asLogLevel))       

    @staticmethod
    def set_hnsw_params_attrs(__obj :object, __dict: dict) -> object:
        for key in __dict: 
            if key == 'batching_params':
                setattr(
                    __obj,
                    key,
                    BaseAerospike.set_hnsw_params_attrs(
                            vectorTypes.HnswBatchingParams(),
                            __dict[key],
                    )
                )
            else:
                setattr(__obj, key, __dict[key])
        return __obj
    
    def prometheus_status(self, done:bool = False) -> None:
        
        self.__cnthb__ += 1
        
        if self._heartbeat_stage == 0:
            if self._heartbeat_current_stage == self._heartbeat_stage:
                return
            self._heartbeat_current_stage = self._heartbeat_stage
            
        if self._heartbeat_stage == 1:
            if self._heartbeat_current_stage == self._heartbeat_stage:
                return
            self._heartbeat_current_stage = self._heartbeat_stage
        if self._heartbeat_stage == 2:
            if self._heartbeat_current_stage == self._heartbeat_stage:
                return
            self._heartbeat_current_stage = self._heartbeat_stage
           
        pausestate : str = None
        if done:
            pausestate = "Done"
        elif self._heartbeat_stage == 0:
            pausestate = "Starting"
        elif self._heartbeat_stage == 1:
            pausestate = "Collecting HDF"
        elif self._heartbeat_stage == 2:
            pausestate = "AVS Status"
        elif  self._actions is not None and OperationActions.POPULATION in self._actions:
            if self._waitidx:
                pausestate = "Waiting"
            elif self._pausedPuts:
                pausestate = "Paused"
            elif self._remainingrecs is not None and self._remainingrecs > 0:
                pausestate = "Running"
            else:
                pausestate = "Idle"
        elif  self._actions is not None and OperationActions.QUERY in self._actions:
            pausestate = "Query"
                
        if self._query_hnswparams is None:
            queryef = '' if self._idx_hnswparams is None else str(self._idx_hnswparams.ef)
        else:
            queryef = self._query_hnswparams.ef
        
        resourceevt = ''
        if self._idx_resource_event is not None:
            if self._idx_resource_event < 0:
                resourceevt = 'Wait Idx'
            elif self._idx_resource_event > 0:
                resourceevt = f"Sleep {self._idx_resource_event} secs"
            else:
                resourceevt = 'Exception'
        concurrentevt = ''
        if self._concurrency is not None:
            if self._concurrency < 0:
                concurrentevt = 'All'
            elif self._concurrency > 0:
                concurrentevt = f'Concurrent {self._concurrency}'
            elif self._concurrency == 1:
                concurrentevt = 'Single'
            else:
                concurrentevt = 'Disabled'
        waitevt = ''
        if self._idx_nowait is not None:
            if self._idx_nowait:
                waitevt = "No Wait"
            else:
                waitevt = "Wait for Completion"
                
        self._prometheus_heartbeat_gauge.set(self.__cnthb__,
                                             {"ns": '' if self._namespace is None else self._namespace,
                                                "set": '' if self._setName is None else self._setName,
                                                "idxns":'' if self._idx_namespace is None else self._idx_namespace,
                                                "idx": '' if self._idx_name is None else self._idx_name,
                                                "idxbin":self._idx_binName,
                                                "idxdist": None if self._idx_distance is None else self._idx_distance.name,
                                                "anndist": self._ann_distance,
                                                "dims": self._dimensions,
                                                "poprecs": None if self._trainarray is None else len(self._trainarray),
                                                "queries": None if self._queryarray is None else len(self._queryarray),
                                                "querynbrlmt": self._query_nbrlimit,
                                                "queryruns": self._query_runs,
                                                "querycurrun": '' if self._query_current_run is None else self._query_current_run,
                                                "dataset":self._datasetname,
                                                "paused": pausestate,
                                                "action": None if self._actions is None else self._actions.name,
                                                "remainingRecs" : self._remainingrecs,
                                                "remainingquerynbrs" : self._remainingquerynbrs,
                                                "querymetric": '' if self._query_metric is None else self._query_metric["type"],
                                                "querymetricvalue": self._query_metric_value,
                                                "querymetricaerospikevalue": self._aerospike_metric_value,
                                                "querymetricbigvalue": self._query_metric_big_value,
                                                "hnswparams": self.hnswstr(),
                                                "queryef": queryef,
                                                "popresrcevt": resourceevt,
                                                "popconcurrent": concurrentevt,
                                                "popwait" : waitevt,
                                                "idxstate": self._idx_state,
                                                "qrydistancecalc": self._ann_distance if self._query_distancecalc is None else self._query_distancecalc
                                                })
        
    def _prometheus_heartbeat(self) -> None:
        from time import sleep
        
        self._logger.debug(f"Heartbeating Start")
        i : int = 0
        self.__cnthb__ : int = 0
        while self._prometheus_hb > 0:
            i += 1
            self.prometheus_status()
            sleep(self._prometheus_hb)
        self._logger.debug(f"Heartbeating Ended")
            
    def _start_prometheus_heartbeat(self) -> None:
        if self._heartbeat_thread is None and self._prometheus_hb > 0:
            self._logger.info(f"Starting Heartbeat at {self._prometheus_hb} secs")
            self._heartbeat_thread = Thread(target = self._prometheus_heartbeat)
            self._heartbeat_thread.start()
            
    def flush_log(self) -> None:
        if(self._logger.handlers is not None):
            for handler in self._logger.handlers:
                handler.flush()                
          
    def print_log(self, msg :str, logLevel :int = logging.INFO) -> None:
        if self._loggingEnabled:
            self._logger.log(level=logLevel, msg=msg)
            if logLevel == logging.INFO:
                print(msg + f', Time: {time.strftime("%Y-%m-%d %H:%M:%S")}')
            elif logLevel == logging.WARN or logLevel == logging.ERROR or logLevel == logging.CRITICAL:
                levelName = "" if logLevel == logging.INFO else f" {logging.getLevelName(logLevel)}: "
                print(levelName + msg + f', Time: {time.strftime("%Y-%m-%d %H:%M:%S")}')
        else:
            levelName = "" if logLevel == logging.INFO else f" {logging.getLevelName(logLevel)}: "
            print(levelName + msg + f', Time: {time.strftime("%Y-%m-%d %H:%M:%S")}')                    
    
    async def shutdown(self, waitforcompletion:bool):
        
        if waitforcompletion and self._sleepexit > 0:
            self.prometheus_status(True)            
            self.print_log(f'existing sleeping {self._sleepexit}')
            self._prometheus_meter_provider.force_flush()
            self._prometheus_metric_reader.force_flush()            
            await asyncio.sleep(self._sleepexit)
                
        self.print_log(f'done: {self}')                
        self.flush_log()
        
        if self._heartbeat_thread is not None:
            hbt = self._prometheus_hb
            self._prometheus_hb = 0
            self._heartbeat_thread.join(timeout=hbt+1)
            self._logger.info(f"Shutdown Heartbeat...")
                        
        self._prometheus_meter_provider.force_flush(1000)
        self._prometheus_metric_reader.force_flush(1000)
        self._prometheus_meter_provider.shutdown()
        #self._prometheus_metric_reader.shutdown()
        self._prometheus_http_server[0].shutdown()
        if logFileHandler is not None:
            loggerASClient.removeHandler(logFileHandler)
        
    def populate_index(self, train:  np.array) -> None:
        pass
    
    def query(self, query: np.array, limit: int) -> List[vectorTypes.Neighbor]:
        pass
    
    def hnswstr(self) -> str:
        if self._idx_hnswparams is None:
            return ''
        if self._idx_hnswparams.batching_params is None:
            batchingparams = ''
        else:
            batchingparams = f"maxrecs:{self._idx_hnswparams.batching_params.max_records}, interval:{self._idx_hnswparams.batching_params.interval}"
        return f"m:{self._idx_hnswparams.m}, efconst:{self._idx_hnswparams.ef_construction}, ef:{self._idx_hnswparams.ef}, batching:{{{batchingparams}}}"
            
    def basestring(self) -> str:
        hnswparams = self.hnswstr()
        
        if self._query_hnswparams is None:
            searchhnswparams = ""
        else:
            searchhnswparams = f", {{s_ef:{self._query_hnswparams.ef}}}"
            
        if self._idx_namespace == self._namespace:
            fullName = f"{self._namespace}.{self._setName}.{self._idx_name}"
        elif self._namespace is None:
            fullName = f"{self._idx_namespace}.{self._idx_name}"        
        else: 
            fullName = f"{self._namespace}.{self._setName}.{self._idx_namespace}.{self._idx_name}"
        
        if self._host is None:
            hosts = "NoHosts"
        else:
            hosts = ','.join(str(hp.host) + ':' + str(hp.port) for hp in self._host)
            
        return f"BaseAerospike([[{hosts}], {self._useloadbalancer}, {fullName}, {self._idx_distance}, {{{hnswparams}}}{searchhnswparams}])"

    def __str__(self):
        if self._idx_namespace == self._namespace:
            fullName = f"{self._namespace}.{self._setName}.{self._idx_name}"
        elif self._namespace is None:
            fullName = f"{self._idx_namespace}.{self._idx_name}"        
        else: 
            fullName = f"{self._namespace}.{self._setName}.{self._idx_namespace}.{self._idx_name}"
        
        return f"{fullName}({self._datasetname})"