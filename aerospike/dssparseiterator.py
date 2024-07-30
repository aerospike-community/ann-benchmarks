import h5py
import numpy
from typing import Any, Callable, Dict, Tuple, Optional, List, Union

from dsiterator import DSIterator
from distance import convert_sparse_to_list

class DSSparseIterator(DSIterator):
    
    def __init__(self, dataset : h5py.Dataset, name : str, sizes : List[int]):
    
        self._dataset : h5py.Dataset = dataset.get(name)
        if self._dataset is None:
            self._sparse_list = []            
        else:
            self._sparse_list = convert_sparse_to_list(self._dataset, sizes)
            
        if len(self._sparse_list) == 0:
            super().__init__((0, 0), name)
        else:
            super().__init__(len(self._sparse_list),
                                len(self._sparse_list[0]),
                                name)
       
    def __iter__(self):
        ''''''
        return self._sparse_list.__iter__()
    
    def __getitem__(self, i):
        ''''''
        return self._sparse_list[i]
    
    @property
    def size(self):
        ''''''
        return self._sparse_list.__sizeof__()
        
    @property
    def dataset(self) -> h5py.Dataset:
        ''''''
        return self._dataset
    
    def free(self):
        ''''''
        self._dataset = None
    
    def tolist(self) -> List[numpy.ndarray[any]]:
        return self._sparse_list
    
    def toarray(self) -> numpy.ndarray[Any, numpy.ndarray[any]]:
        return numpy.array(self._sparse_list)
    
    def index(self, value) -> int:
        return self._sparse_list.index(value)
    
    def astype(self, astype) -> "DSIterator":
        
        newtype : numpy.dtype = numpy.dtype(astype)
        
        if newtype != self.dtype:
            for idx, element in enumerate(self._sparse_list):
                self._sparse_list[idx] = element.astype(astype)
            self.dtype = newtype
            
        return self
    
    def __str__(self) -> str:
        return f"DSSparseIterator{{name:{self.name}, shape:{self.shape}}}"
        