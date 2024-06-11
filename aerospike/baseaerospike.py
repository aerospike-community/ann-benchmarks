import asyncio
import numpy as np
import time
import logging
import json
import argparse

from typing import List, Dict
from importlib.metadata import version
from logging import _nameToLevel as LogLevels

from aerospike_vector_search import types as vectorTypes

_distanceNameToAerospikeType: Dict[str, vectorTypes.VectorDistanceMetric] = {
    'angular': vectorTypes.VectorDistanceMetric.COSINE,
    'euclidean': vectorTypes.VectorDistanceMetric.SQUARED_EUCLIDEAN,
    'hamming': vectorTypes.VectorDistanceMetric.HAMMING,
    'jaccard': None,
    'dot': vectorTypes.VectorDistanceMetric.DOT_PRODUCT,
}

loggerASClient = logging.getLogger("aerospike_vector_search")
logFileHandler = None
 
class BaseAerospike():


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
            help="A list of host and optionsl ports. Example: 'hosta:5000' or 'hostb'",
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
            '-n', "--namespace",
            metavar="NS",
            help="The Aerospike Namespace",
            default="test",
        )
        parser.add_argument(
            '-s', "--setname",
            metavar="SET",
            help="The Aerospike Set Name",
            default="HDF-data",
        )
        parser.add_argument(
            '-I', "--idxname",
            metavar="IDX",
            help="The Vector Index Name. Defaults to the Set Name with the suffix of '_idx'",
            default=None,
        )
        parser.add_argument(
            '-g', "--generatedetailsetname",            
            help="Generates a Set name based on distance type, dimensions, index params, etc.",
            action='store_true'
        )
        parser.add_argument(
            '-b', "--vectorbinname",
            metavar="BIN",
            help="The Aerospike Bin Name where the Vector is stored",
            default="HDF_embedding",
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
            '-S', "--searchparams",
            metavar="PARM",
            type=json.loads,
            help="The Vector's Search Params (HnswParams)",
            default=None
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
            metavar="DLEVEL",           
            help="The Driver's Logging level",
            default="ERROR",
            choices=LogLevels.keys(),
        )
    
    def __init__(self, runtimeArgs: argparse.Namespace, logger: logging.Logger):
        
        global logFileHandler
        
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
        
        self._namespace = runtimeArgs.namespace
        self._setName = runtimeArgs.setname
        self._paramsetname = runtimeArgs.generatedetailsetname
        if runtimeArgs.idxname is None or not runtimeArgs.idxname:
            self._idx_name = f'{self._setName}_Idx'
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
        
        if runtimeArgs.searchparams is None or len(runtimeArgs.searchparams) == 0:
            self._query_hnswparams = None
        else:
            self._query_hnswparams = BaseAerospike.set_hnsw_params_attrs(
                                        vectorTypes.HnswSearchParams(),
                                        runtimeArgs.searchparams
                                    )            
        
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
                if self._asLogLevel is not None and self._asLogLevel:
                    loggerASClient.addHandler(logFileHandler)
                    loggerASClient.setLevel(logging.getLevelName(self._asLogLevel))
                self._logger.addHandler(logFileHandler)            
                self._logger.setLevel(logging.getLevelName(self._logLevel))
            self._logFileHandler = logFileHandler
            self._loggingEnabled = True
            self._logger.info(f'Start Aerospike: Metric: {self.basestring()}')
            self._logger.info(f"  aerospike-vector-search: {version('aerospike_vector_search')}")
         
    @staticmethod
    def set_hnsw_params_attrs(__obj :object, __dict: dict) -> object:
        for key in __dict: 
            if key == 'batching_params':
                setattr(
                    __obj,
                    key,
                    BaseAerospike.set_hnsw_params_attrs(
                            vectorTypes.HnswBatchingParams(),
                            __dict[key].asdict()
                    )
                )
            else:
                setattr(__obj, key, __dict[key])
        return __obj
    
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
    
    def done(self):
        self.print_log(f'done: {self}')                
        self.flush_log()
                
    def populate_index(self, train:  np.array) -> None:
        pass
    
    def query(self, query: np.array, limit: int) -> List[vectorTypes.Neighbor]:
        pass
    
    def basestring(self) -> str:
        batchingparams = f"maxrecs:{self._idx_hnswparams.batching_params.max_records}, interval:{self._idx_hnswparams.batching_params.interval}, disabled:{self._idx_hnswparams.batching_params.disabled}"
        hnswparams = f"m:{self._idx_hnswparams.m}, efconst:{self._idx_hnswparams.ef_construction}, ef:{self._idx_hnswparams.ef}, batching:{{{batchingparams}}}"
        if self._query_hnswparams is None:
            searchhnswparams = None
        else:
            searchhnswparams = f", {{s_ef:{self._query_hnswparams.ef}}}"
        
        return f"BaseAerospike([{self._host}:{self._port}, {self._useloadbalancer}, {self._namespace}.{self._setName}.{self._idx_name}, {self._idx_distance}, {{{hnswparams}}}{searchhnswparams}])"

    def __str__(self):
        return self.basestring()