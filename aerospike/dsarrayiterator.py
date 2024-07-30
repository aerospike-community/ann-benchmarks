import h5py
import numpy
from typing import Any, Callable, Dict, Tuple, Optional, List, Union

from dsiterator import DSIterator

class DSArrayIterator(DSIterator):
    
    def __init__(self, dataset : h5py.Dataset, name : str, dtype : Optional[str] = None):
                
        self._dataset : h5py.Dataset = dataset.get(name)
        if self._dataset is None:
            super().__init__((0, 0),
                            name,
                            dtype)
            self._array : numpy.ndarray[Any, numpy.ndarray[Any]] = numpy.array([], dtype=self.dtype)
        else:
            if dtype is None:
                self._array : numpy.ndarray[Any, numpy.ndarray[Any]] = numpy.array(self._dataset, dtype=self._dataset.dtype)                
            else:
                self._array : numpy.ndarray[Any, numpy.ndarray[Any]] = numpy.array(self._dataset, dtype=dtype)
            
            super().__init__(self._dataset.shape,
                            name,
                            self._array.dtype.name)            
            
    def __iter__(self):
        ''''''
        return self._array.__iter__()
    
    def __getitem__(self, i):
        ''''''
        return self._array[i]
    
    @property
    def size(self):
        ''''''
        return self._array.size
        
    @property
    def dataset(self) -> h5py.Dataset:
        ''''''
        return self._dataset
    
    def free(self):
        ''''''
        self._dataset = None
    
    def astype(self, astype) -> "DSIterator":
        self._array = self._array.astype(astype)
        self.dtype = self._array.dtype
        return self
    
    def tolist(self) -> List[numpy.ndarray[any]]:
        return list(self._array)
    
    def toarray(self) -> numpy.ndarray[Any, numpy.ndarray[any]]:
        return self._array
    
    def __str__(self) -> str:
        return f"DSArrayIterator{{name:{self.name}, shape:{self.shape}, dtype:{self.dtype}}}"
        