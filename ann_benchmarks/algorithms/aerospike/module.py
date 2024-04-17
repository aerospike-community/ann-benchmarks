import asyncio
import os
import numpy as np
import time

from typing import Iterable, List, Any
from enum import Enum
from pythonping import ping

from aerospike_vector import vectordb_admin, vectordb_client, types

from ..base.module import BaseANN

_AerospikeIdxNames : list = []

class Aerospike(BaseANN):
    
    def __init__(self,
                    metric: str, 
                    dimension: int,
                    idx_type,
                    hnswParams: dict,
                    uniqueIdxName: bool = True,
                    dropIdx: bool = True,
                    ping: bool = False):
        self._metric = metric
        self._dims = dimension
        self._idx_type = idx_type.upper()        
        self._idx_value = types.VectorDistanceMetric[self._idx_type]
        
        if hnswParams is None or len(hnswParams) == 0:
            self._idx_hnswparams = None
        else:
            self._idx_hnswparams = Aerospike.SetHnswParamsAttrs(
                                        types.HnswParams(),
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
        #self._idx_parallism = int(os.environ.get("APP_INDEXER_PARALLELISM") or 1)
        if not uniqueIdxName or self._idx_hnswparams is None:
            self._idx_name = f'{self._setName}_{self._idx_type}_Idx'
        else:
            self._idx_name = f'{self._setName}_{self._idx_type}_{self._dims}_{self._idx_hnswparams.m}_{self._idx_hnswparams.ef_construction}_{self._idx_hnswparams.ef}_Idx'
        self._idx_binName = "ANN_embedding"
        self._idx_binKeyName = "ANN_key"
        self._query_hnswsearchparams = None
        
        if ping:
            print(f'Aerospike: Trying Connection to {self._host} {self._verifyTLS} {self._listern}')
            print(ping(self._host, verbose=True))
        print('Aerospike: Try Create Admin client')
        
        self._adminClient = vectordb_admin.VectorDbAdminClient(
                                            seeds=types.HostPort(host=self._host,
                                                                    port=self._port,
                                                                    isTls=self._verifyTLS), 
                                            listener_name=self._listern)
         
        self._client = vectordb_client.VectorDbClient(
                                    seeds=types.HostPort(host=self._host,
                                                            port=self._port,
                                                            isTls=self._verifyTLS), 
                                    listener_name=self._listern)
        self._queryLoopEvent = asyncio.get_event_loop()
        
    @staticmethod
    def SetHnswParamsAttrs(__obj :object, __dict: dict) -> object:
        for key in __dict: 
            if key == 'batching_params':
                setattr(
                    __obj,
                    key,
                    Aerospike.SetHnswParamsAttrs(
                            types.HnswBatchingParams(),
                            __dict[key].asdict()
                    )
                )
            else:
                setattr(__obj, key, __dict[key])
        return __obj
        
    def done(self) -> None:
        """Clean up BaseANN once it is finished being used."""
        print(f'done: {self}')
        
        loop = asyncio.get_event_loop()
        try:
            if self._client is not None:
                clientCloseTask = loop.create_task(self._client.close())
            if self._adminClient is not None:
                adminCloseTask = loop.create_task(self._adminClient.close())
            loop.run_until_complete(asyncio.gather(clientCloseTask, adminCloseTask))
        finally:
            loop.close
        if self._queryLoopEvent is not None:
            self._queryLoopEvent.close
                            
    async def fitAsync(self, X: np.array) -> None:
        global _AerospikeIdxNames
        
        if X.dtype != np.float32:
            X = X.astype(np.float32)
                
        print(f'Aerospike fitAsync: {self} Shape: {X.shape}')
        
        populateIdx = True
        
        #If exists, no sense to try creation...
        existingIndexes = await self._adminClient.index_list()
        if(any(index["id"]["namespace"] == self._namespace
                                and index["id"]["name"] == self._idx_name 
                        for index in existingIndexes)):
            print(f'Aerospike: Index {self._namespace}.{self._idx_name} Already Exists')
            
            #since this can be an external DB (not in a container), we need to clean up from prior runs
            #if the index name is in this list, we know it was created in this run group and don't need to drop the index.
            #If it is a fresh run, this list will not contain the index and we know it needs to be dropped.
            if self._idx_name in _AerospikeIdxNames:
                print(f'Aerospike: Index {self._namespace}.{self._idx_name} being reused (not re-populated)')
                populateIdx = False
            elif self._idx_drop:
                print(f'Aerospike: Dropping Index...')
                s = time.time()
                await self._adminClient.index_drop(namespace=self._namespace,
                                                    name=self._idx_name)
                await asyncio.sleep(10) #we have to sleep...
                t = time.time()
                print(f"Aerospike: Drop Index Time (sec) = {t - s}")                
            else:
                populateIdx = False
        else:
            print(f'Aerospike: Creating Index {self._namespace}.{self._idx_name}')
            s = time.time()
            await self._adminClient.index_create(namespace=self._namespace,
                                                    name=self._idx_name,
                                                    sets=self._setName,
                                                    vector_field=self._idx_binName,
                                                    dimensions=self._dims,
                                                    index_params= self._idx_hnswparams,
                                                    vector_distance_metric=self._idx_value
                                                    )
            await asyncio.sleep(10) #we have to sleep...
            t = time.time()
            print(f"Aerospike: Index Creation Time (sec) = {t - s}")
            _AerospikeIdxNames.append(self._idx_name)
            
        if populateIdx:
            print(f'Aerospike: Populating Index {self._namespace}.{self._idx_name}')           
            s = time.time()
            taskPuts = []
            for i, embedding in enumerate(X):
                #print(f'Item {i},{embedding.shape}  Vector:{embedding}')
                taskPuts.append(self._client.put(namespace=self._namespace,
                                                    set_name=self._setName,                            
                                                    key=i,
                                                    record_data={
                                                        self._idx_binName:embedding.tolist(),
                                                        self._idx_binKeyName:i
                                                    }
                ))
            await asyncio.gather(*taskPuts) 
            t = time.time()
            print(f"Aerospike: Index Put Time (sec) = {t - s}")
            print("Aerospike: waiting for indexing to complete")
            await self._client.wait_for_index_completion(namespace=self._namespace,
                                                                name=self._idx_name)
            await asyncio.sleep(10) #we have to sleep...
            t = time.time()
            print(f"Aerospike: Index Total Populating Time (sec) = {t - s}")
    
    def fit(self, X: np.array) -> None:              
        print(f'Aerospike fit: {self} Shape: {X.shape}')
        loop = asyncio.get_event_loop()
        try:
            t = loop.create_task(self.fitAsync(X))
            loop.run_until_complete(t)
        finally:
            loop.close
        #asyncio.run()
    
    def set_query_arguments(self, hnswParams: dict = None):        
        if hnswParams is not None and len(hnswParams) > 0:
            self._query_hnswsearchparams = Aerospike.SetHnswParamsAttrs(
                                                    types.HnswSearchParams(),
                                                    hnswParams
                                                )

    async def queryAsync(self, q, n):
        result = await self._client.vector_search(namespace=self._namespace,
                                                    index_name=self._idx_name,
                                                    query=q.tolist(),
                                                    limit=n,
                                                    search_params=self._query_hnswsearchparams,
                                                    bin_names=[self._idx_binKeyName])
        result_ids = [neighbor.bins[self._idx_binKeyName] for neighbor in result]
        return result_ids
    
    def query(self, q, n):
        queryTask = self._queryLoopEvent.create_task(self.queryAsync(q,n))
        return self._queryLoopEvent.run_until_complete(queryTask)
        
    #def get_batch_results(self):
    #    return self.batch_results

    #def get_batch_latencies(self):
    #    return self.batch_latencies

    def __str__(self):
        batchingparams = f"maxrecs:{self._idx_hnswparams.batching_params.max_records}, interval:{self._idx_hnswparams.batching_params.interval}, disabled:{self._idx_hnswparams.batching_params.disabled}"
        hnswparams = f"m:{self._idx_hnswparams.m}, efconst:{self._idx_hnswparams.ef_construction}, ef:{self._idx_hnswparams.ef}, batching:{{{batchingparams}}}"
        return f"Aerospike([{self._metric}, {self._host}, {self._port}, {self._namespace}, {self._setName}, {self._idx_name}, {self._idx_type}, {self._idx_value}, {self._dims}, {{{hnswparams}}}, {{{self._query_hnswsearchparams}}}])"
