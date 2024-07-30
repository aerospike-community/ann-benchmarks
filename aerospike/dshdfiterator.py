import h5py
import numpy
from typing import Any, Callable, Dict, Tuple, Optional, List, Union

from dsiterator import DSIterator

storage_threshold : int = 4

class DSHDFIterator(DSIterator):
    
    @staticmethod
    def set_storage_threshold(newthreshold : int):
        storage_threshold =  newthreshold
        
    @staticmethod
    def determine_iterator_large(dataset : h5py.Dataset, name : str, dtype : str) -> DSIterator:
        '''
        Will return the proper iterator that will fit into memory.
        '''
        import psutil
        from dsarrayiterator import DSArrayIterator
        
        ds : Union[h5py.Dataset, None] = dataset.get(name)
        if ds is None:
            return DSArrayIterator(dataset, name, dtype)
        
        memoryAvaiable = psutil.virtual_memory()[1]
        dstype : numpy.dtype
        if dtype is None:
            dstype = ds.dtype
        else:
            dstype = numpy.dtype(dtype)
            
        dssize = ds.size * dstype.itemsize
        
        if memoryAvaiable / dssize >= storage_threshold:
             return DSArrayIterator(dataset, name, dtype)
                 
        return DSHDFIterator(dataset, name, dtype)

    def __init__(self, dataset : h5py.Dataset, name : str, dtype : Optional[str] = None):
                
        self._dataset : h5py.Dataset = dataset[name]
        if dtype is not None and self._dataset.dtype != numpy.dtype(dtype):
            self._dataset.astype(dtype)
            
        super().__init__(self._dataset.shape,
                            name,
                            self._dataset.dtype.name)
        
        print(f"{self} will not be completely placed into memory but paged to save memory space.")
                
    def __iter__(self):
        ''''''
        return self._dataset.__iter__()
    
    def __getitem__(self, i):
        ''''''
        return self._dataset[i]
    
    @property
    def size(self):
        ''''''
        return self._dataset.size
        
    @property
    def dataset(self) -> h5py.Dataset:
        ''''''
        return self._dataset
    
    @property
    def large(self) -> bool:
        '''
        Returns true if it is a large dataset
        '''
        return True
    
    def free(self):
        ''''''        
        self._dataset = None
    
    def astype(self, astype):        
        result =  self._dataset.astype(astype)
        self.dtype = self._dataset.dtype
        return result
    
    def tolist(self) -> List[numpy.ndarray[any]]:
        return list(self._dataset)
    
    def toarray(self) -> numpy.ndarray[Any, numpy.ndarray[any]]:
        return numpy.array(self._dataset, dtype=self.dtype)
    
    def __str__(self) -> str:
        return f"DSHDFIterator{{name:{self.name}, shape:{self.shape}, large: True}}"
        