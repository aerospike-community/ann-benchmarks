import asyncio
import numpy as np
import time
import json
import argparse
import logging
import aerospike

from logging import _nameToLevel as LogLevels
from typing import Iterable, List, Union, Any, Tuple
from importlib.metadata import version

from aerospike_vector_search import types as vectorTypes
from aerospike_vector_search.aio import AdminClient as vectorASyncAdminClient, Client as vectorASyncClient
from metrics import all_metrics as METRICS, DummyMetric

logger = logging.getLogger(__name__)
logFileHandler = None

class AerospikeDS():
    
    @staticmethod
    def parse_arguments(parser: argparse.ArgumentParser) -> None:
        '''
        Adds the arguments required to create a dataset from a Aerospike vector set. 
        '''
        
        parser.add_argument(
            '-a', '--hosts',
            metavar="HOST:PORT",            
            nargs='+',
            help="A list of Aerospike host and optional ports (defaults to 3000). Example: 'hosta:3000' or 'hostb'",
            default=['localhost:3000'],
        )        
        parser.add_argument(
            '--policies',
            metavar="POLICIES",            
            type=json.loads,
            help="Aerospike connection policies",
            default='{"read": {"total_timeout": 1000}}'
        )
        parser.add_argument(
            '-A', '--vectorhosts',
            metavar="HOST:PORT",            
            nargs='+',
            help="A list of Aerospike Vector host and optional ports (defaults to 5000). Example: 'hosta:5000' or 'hostb'",
            default=['localhost:5000'],
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
            "--hdf",
            metavar="HDFFILE",
            help="A HDF file that will be created in the 'data' folder by default",
            default=None,
            type=str,
            required=True,
        )
        parser.add_argument(
            '-idxns', "--indexnamespace",
            metavar="INDEXNAME",
            help="Vector's Index Namespace. Defaults to the Aerospike namepsace",
            required=False,
            default="test",
            type=str
        )
        parser.add_argument(
            '-idx', "--indexname",
            metavar="INDEXNAME",
            help="Vector's Index Name",
            required=True
        )
        parser.add_argument(
            '-S', "--searchparams",
            metavar="PARM",
            type=json.loads,
            help="The Vector's Search Params (HnswParams) used to obtained the neighbors",
            default=None
        ) 
        parser.add_argument(
            '-pk', "--pkbinname",
            metavar="BINNAME",
            type=str,
            help='''
            The Bin Name that represents the Primary Key for a record.
            If not provided the Aerospike PK will try to be used if the PK value is returned.
            If the Aerospike PK is not a value (digest), PK array will not be  part of the HDF dataset (None).
            ''',
            default="_proximus_uk_"
        )
        parser.add_argument(
            "-n", "--neighbors",
            metavar="NEIGHBORS",
            type=int,
            help="The number of neighbors to return from the query.",
            default=100,
        )
        parser.add_argument(
            "--testsize",
            metavar="VALUE",
            help='''
            If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split.
            If int, represents the absolute number of test samples.
            If None, the value is set to the complement of the train size.
            If ``trainsize`` is also None, it will be set to 0.25.
            ''',
            type=Union[float,int,None],
            default=0.1,
        )
        parser.add_argument(
            "--trainsize",
            metavar="VALUE",
            help='''
             If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the train split.
             If int, represents the absolute number of train samples.
             If None, the value is automatically set to the complement of the test size.
            ''',
            type=Union[float,int,None],
            default=None,
        )
        parser.add_argument(
            "--randomstate",
            metavar="VALUE",
            help='''
            Controls the shuffling applied to the data before applying the split.
            Pass an int for reproducible output across multiple function calls.
            See :term:`Glossary <random_state>`.
            Can be a int, RandomState instance or None.
            Using a default of 1 as defined in the ANN benchmark.
            See https://scikit-learn.org/dev/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split
            ''',
            default=1,
        )
        parser.add_argument(
            "--usetrainingds",            
            help='''
            Creates the training dataset based on the actual vectors from the DB.
            This will use the Bruteforce/k-nn method to calculate the neighbors.
            The defualt is to use all vector records in the DB and a sampling is taken to conduct the searches using the Aerospike implementation.  
            ''',
            action='store_true'
        )        
        parser.add_argument(
                "--metric",
                metavar="TYPE",
                help="Which metric to use to calculate Recall.",
                default="k-nn",
                type=str,
                choices=METRICS.keys(),
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
        
    def __init__(self, runtimeArgs: argparse.Namespace) -> None:
        from datasets import get_dataset_fn
        
        self._hosts = []
        for pos, host in enumerate(runtimeArgs.hosts):
            parts = host.split(':')
            if len(parts) == 1:
                self._hosts.append((parts[0], 3000))
            elif len(parts) == 2:
                self._hosts.append((parts[0],int(parts[1])))
        self._vector_hosts : List[vectorTypes.HostPort] = []
        for pos, host in enumerate(runtimeArgs.vectorhosts):
            parts = host.split(':')
            if len(parts) == 1:
                self._vector_hosts.append(vectorTypes.HostPort(host=host,port=5000,is_tls=bool(runtimeArgs.vectortls)))
            elif len(parts) == 2:
                self._vector_hosts.append(vectorTypes.HostPort(host=parts[0],port=int(parts[1]),is_tls=bool(runtimeArgs.vectortls)))

        self._as_namespace : str = None
        self._as_set : str = None
        self._as_pkbinname : str = runtimeArgs.pkbinname
        self._vector_lb : bool = runtimeArgs.vectorloadbalancer
        self._hdf_path, self._ann_dataset = get_dataset_fn(runtimeArgs.hdf)        
        self._vector_namespace : str = runtimeArgs.indexnamespace
        self._vector_name : str = runtimeArgs.indexname
        self._as_vectorbinname : str = None
        self._vector_searchparams = None if runtimeArgs.searchparams is None else AerospikeDS.set_hnsw_params_attrs(vectorTypes.HnswSearchParams(), runtimeArgs.searchparams)
        self._vector_distance : vectorTypes.VectorDistanceMetric = None
        self._vector_hnsw : dict = None
        self._vector_dimensions : int = None
        self._vector_neighbors : int = runtimeArgs.neighbors
        self._vector_trainsize = runtimeArgs.trainsize
        self._vector_testsize = runtimeArgs.testsize
        self._vector_randomstate = runtimeArgs.randomstate
        self._vector_usetrainingds = runtimeArgs.usetrainingds
        self._vector_metric = METRICS[runtimeArgs.metric]
        
        self._logging_init(runtimeArgs, logger)

        self._as_clientconfig = {
            'hosts':    self._hosts,
            'policies': runtimeArgs.policies,
        }
                
    def _logging_init(self, runtimeArgs: argparse.Namespace, logger: logging.Logger) -> None:
       
        global logFileHandler
        
        self._logFilePath = runtimeArgs.logfile        
        self._logLevel = runtimeArgs.loglevel
        self._logger = logger
        self._loggingEnabled = False
           
        if self._logFilePath is not None and self._logFilePath and self._logLevel != "NOTSET":
            print(f"Logging to file {self._logFilePath}")
            if logFileHandler is None:
                logFileHandler = logging.FileHandler(self._logFilePath, "w")                
                logFormatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                logFileHandler.setFormatter(logFormatter)
                self._logger.addHandler(logFileHandler)            
                self._logger.setLevel(logging.getLevelName(self._logLevel))
            self._logFileHandler = logFileHandler
            self._loggingEnabled = True
            self._logger.info(f'Start Aerospike: Metric: {self}')
            self._logger.info(f"  aerospike: {version('aerospike')}")
            self._logger.info(f"  aerospike-vector-search: {version('aerospike_vector_search')}")
            #self._logger.info(f"Prometheus HTTP Server: {self._prometheus_http_server[0].server_address}")
            #self._logger.info(f"  Metrics Name: {self._meter.name}")
            self._logger.info(f"Arguments: {runtimeArgs}")
      
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

    @staticmethod
    def _vector_hostport_str(vectorhosts : List[vectorTypes.HostPort]) -> str:
        return ",".join(f"{hp.host}:{hp.port}" for hp in vectorhosts)
    
    @staticmethod
    def set_hnsw_params_attrs(__obj :object, __dict: dict) -> object:
        for key in __dict: 
            if key == 'batching_params':
                setattr(
                    __obj,
                    key,
                    AerospikeDS.set_hnsw_params_attrs(
                            vectorTypes.HnswBatchingParams(),
                            __dict[key].asdict()
                    )
                )
            else:
                setattr(__obj, key, __dict[key])
        return __obj

    async def __aenter__(self):
        return self
 
    async def __aexit__(self, *args):
        self.flush_log()        
        
    async def index_exist(self, adminClient: vectorASyncAdminClient) -> Union[dict, None]:        
        existingIndexes = await adminClient.index_list()
        if len(existingIndexes) == 0:
            return None
        return [(index if index["id"]["namespace"] == self._vector_namespace
                            and index["id"]["name"] == self._vector_name else None)
                        for index in existingIndexes][0]
    
    async def populate_vector_info(self) -> None:
        '''
        Updates AerospikeDS instance with attributes about the vector index.
        '''
        from baseaerospike import _distanceNameToAerospikeType as DISTANCETYPES
        
        self.print_log(f'Updating instance with Aerospike Vector Idx attributes for {self._vector_namespace}.{self._vector_name} on {AerospikeDS._vector_hostport_str(self._vector_hosts)}')
        
        async with vectorASyncAdminClient(seeds=self._vector_hosts,
                                            listener_name=None,
                                            is_loadbalancer=self._vector_lb
            ) as adminClient:
           
            idxAttribs = await self.index_exist(adminClient)
            if idxAttribs is None:
                self.print_log(f'populate_vector_info: Vector Index: {self._vector_namespace}.{self._vector_name}, not found')
                raise FileNotFoundError(f"Vector Index {self._vector_namespace}.{self._vector_name} not found")

            self._as_namespace : str = idxAttribs['storage']['namespace']
            self._as_set : str = idxAttribs['setFilter']
            self._as_vectorbinname : str = idxAttribs["field"]
            self._vector_distance : vectorTypes.VectorDistanceMetric = vectorTypes.VectorDistanceMetric[idxAttribs["vectorDistanceMetric"]]
            self._vector_hnsw : dict = idxAttribs['hnsw_params']
            self._vector_dimensions : int = idxAttribs['dimensions']
            self._vector_ann_distance : str = next((anndisttype for anndisttype, disttype in DISTANCETYPES.items() if disttype == self._vector_distance))

    async def Generate_hdf_dataset(self) -> str:
        import h5py        
        from string import digits
        
        self.print_log(f"Creating HDF dataset {self._hdf_path}")
        
        pkarray = await self.get_pk_from_as_set()
        (train,
            sampletrain,
            test,
            neighbors,
            distances,
            metricresult) = await self.get_vectors(pkarray)
        
        if self._vector_usetrainingds:
            train = sampletrain
        
        with h5py.File(self._hdf_path, "w") as f:
            f.attrs["type"] = "dense"
            f.attrs["distance"] = self._vector_ann_distance
            f.attrs["dimension"] = len(train[0])
            f.attrs["point_type"] = train[0].dtype.name.rstrip(digits)
            if metricresult is not None:
                f.attrs["recall"] = metricresult
                f.attrs["recallmethod"] = self._vector_metric["type"]
            print(f"train size: {train.shape[0]} * {train.shape[1]}")
            print(f"test size:  {test.shape[0]} * {test.shape[1]}")
            f.create_dataset("train", data=train)
            f.create_dataset("test", data=test)
            f.create_dataset("neighbors", data=neighbors)
            if distances is not None:
                f.create_dataset("distances", data=distances)
            if not self._vector_usetrainingds:
                f.create_dataset("primarykeys", data=pkarray)
            hdfpath = f.filename
            self.print_log(f"Created HDF dataset '{hdfpath}'")
            
        return hdfpath

    async def get_pk_from_as_set(self) -> List:
        '''
        Read records from the Aerospike DB and return an array of the PKs.        
        '''
        self.print_log(f"Opening connecction to Aerospike DB Cluster {self._as_clientconfig}")
        
        pkarray = []
        
        client = aerospike.client(self._as_clientconfig).connect()
        try:
            self.print_log(f"Connected to Aerospike DB Cluster {client.info_random_node('version')}")
            
            self._logger.debug(f"Creating Query on {self._as_namespace}.{self._as_set} for PK { 'PK' if self._as_pkbinname == None else self._as_pkbinname}")
            
            query = client.query(self._as_namespace, self._as_set)
            if self._as_pkbinname is not None:                
                query.select(self._as_pkbinname)
            records = query.results()
            
            self.print_log(f"Aerospike Query record number: {len(records)}")
            
            for record in records:
                key, _, bins = record
                pkvalue = None
                if self._as_pkbinname is None:
                    _, _, primaryKeyValue, digest = key
                    if primaryKeyValue is not None:                        
                        pkvalue = primaryKeyValue
                    else:
                        pkvalue = digest
                elif self._as_pkbinname in bins:
                    pkvalue = bins[self._as_pkbinname]
                                
                pkarray.append(pkvalue)
        
        finally:
            self._logger.debug("closing connection to DB")
            client.close()
            
        if len(pkarray) == 0 or all(p is None for p in pkarray):
            pkarray is None
        
        self.print_log(f"Closed connecction to Aerospike DB Cluster {self._as_clientconfig} with PK array of {0 if pkarray is None else len(pkarray)}")
        
        return pkarray
    
    async def calculate_knn_neighbor_distance(self, train: np.ndarray, test: np.ndarray) -> Tuple[np.ndarray[np.ndarray], np.ndarray[np.ndarray]]:
        '''
        Determines the neighbors and associated distance from the training and test datasets
        using Bruteforce/k-nn method. 
        returns the neighbors and distances.
        '''
        from bruteforce import BruteForceBLAS
        
        self.print_log(f"Determing neighbors/distances from training and test datasets")
        bf = BruteForceBLAS(self._vector_ann_distance,
                            precision=train.dtype)
        bf.fit(train)
        
        neighbors = []
        distances = []
        
        for i, x in enumerate(test):
            if i % 1000 == 0:
                print(f"BruteForce (k-nn) {i}/{len(test)}...")

            # Query the model and sort results by distance
            res = list(bf.query_with_distances(x, self._vector_neighbors))
            res.sort(key=lambda t: t[-1])

            # Save neighbors indices and distances
            neighbors.append(np.array([idx for idx, _ in res]))
            distances.append(np.array([dist for _, dist in res]))
    
        self.print_log(f"Determined neighbors ({len(neighbors)})/distances ({len(distances)})")
        
        return np.array(neighbors), np.array(neighbors)
        
    async def get_vectors(self, pkarray : List) -> Tuple[np.ndarray[np.ndarray], np.ndarray[np.ndarray], np.ndarray[np.ndarray], np.ndarray[np.ndarray], np.ndarray[np.ndarray], Union[float, int, None]]:
        '''
        Gets the corresponding vectors based on the pkarray.
        It returns:
            training,
            calculated training (based on smapling of all vector records),
            test (used to perform searches),
            neighbors (result from the test dataset),
            distances (only provided when neighbors are calculated via k-nn method)
            metricvalue (recal) value (can be None)
        '''
        from sklearn.model_selection import train_test_split as sklearn_train_test_split
        
        self.print_log(f'get_vectors based on PKs ({len(pkarray)})')
        
        async with vectorASyncClient(seeds=self._vector_hosts,
                                        is_loadbalancer=self._vector_lb
                            ) as client:
            self.print_log(f'Opened connection to Vectors on {AerospikeDS._vector_hostport_str(self._vector_hosts)} for set {self._as_namespace}.{self._as_set}')
            vectors = []
            
            for pk in pkarray:
                record = await client.get(namespace=self._as_namespace,
                                            key=pk,
                                            field_names=[self._as_vectorbinname],
                                            set_name=self._as_set)
                vectors.append(np.array(record.fields[self._as_vectorbinname]))
                
            vectors = np.array(vectors)
            self.print_log(f"Splitting {len(vectors)}*{self._vector_dimensions} into train/test with sizes Test:{self._vector_testsize}, Train: {self._vector_trainsize}, Random State: {self._vector_randomstate}")
            
            X_train, X_test = sklearn_train_test_split(vectors,
                                                        test_size=self._vector_testsize,
                                                        train_size=self._vector_trainsize,
                                                        random_state=self._vector_randomstate)
                        
            distanceds : np.array[np.array] = None
            neighborsds : np.array[np.array] = None
            metricvalue = None
            
            if self._vector_usetrainingds:
                self.print_log(f"Split vector DS into training {X_train.shape} and test {X_test.shape} DSs. Using Brunte ", logging.WARN)
                neighborsds, distanceds = await self.calculate_knn_neighbor_distance(X_test, X_train)    
            else:
                self.print_log(f"Using vector orginal DS {vectors.shape} and test DS {X_test.shape} (Training DS, which is not used, is {X_train.shape}).")
                neighborsds = await self.conduct_search(client, X_test)
                
            metricfunc = None if self._vector_metric is None else self._vector_metric["function"]
            if metricfunc is not None:
                try:
                    metricvalue = metricfunc(X_test, X_test, DummyMetric(), 0, len(X_test[0]))
                    self.print_log(f"{self._vector_metric['type']} Recall/Metric Value: {metricvalue}")
                except Exception as e:
                    self.print_log(f"Recall/Metric caculation failed with '{e}'", logging.ERROR)
                
            zeronbrs = [len(neighbor) == 0 for neighbor in neighborsds].count(True)
            if zeronbrs > 0:
                self.print_log(f"Found {zeronbrs} neighbors in resulting Search.", logging.WARN)
                
            return  vectors, X_train, X_test, neighborsds, distanceds, metricvalue

    async def conduct_search(self, client:vectorASyncClient, testds : np.ndarray[np.ndarray]) -> np.ndarray:
        
        self.print_log(f"Performing Search using Test DS {testds.shape}")
        neighborsds = []
        for searchitem in testds:
            neighbors = await self.search_vector(client, searchitem.tolist())
            if len(neighbors) > 0:
                result_ids = [neighbor.key.key for neighbor in neighbors]
                neighborsds.append(np.array(result_ids))
            else:
                self._logger.debug(f"Found zero neighbors in resulting Search.", logging.WARN)
                neighborsds.append(np.empty())
                
        neighborsds = np.array(neighborsds)
        self.print_log(f"Search Completed resulting in a neighbors DS {neighborsds.shape}")
        
        return neighborsds

    async def search_vector(self, client:vectorASyncClient, query:List[float|bool]) -> List[vectorTypes.Neighbor]:
        
         return await client.vector_search(namespace=self._vector_namespace,
                                                index_name=self._vector_name,
                                                query=query,
                                                limit=self._vector_neighbors,
                                                search_params=self._vector_searchparams)
