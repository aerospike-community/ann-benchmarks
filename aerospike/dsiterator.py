import numpy
import h5py
from typing import Any, Tuple, Optional, Union, List
from collections.abc import Sequence

class DSIterator(Sequence):
    def __init__(self, length : int, dim : int, name : str, dtype : Optional[str] = None):
        
        self.shape : Union[Tuple[int,int], Tuple[int]]
        self.dtype : numpy.dtype = numpy.dtype(dtype)
        self._len = length
        self.shape = (length, dim)
        self.name = name
            
    def __init__(self, shape : Union[Tuple[int,int],Tuple[int],None], name : str, dtype : Optional[str] = None):
        self.dtype : Union[numpy.dtype,None] = None if dtype is None else numpy.dtype(dtype)
        
        self.shape : Union[Tuple[int,int], Tuple[int]]
        if shape is None or len(shape) == 0:
            self._len = 0
            self.shape = (0)
        elif len(shape) == 1:
            self._len = shape[0]
            self.shape = shape
        else:
            self._len = shape[0]
            self.shape = shape
            
        self.name = name
        
    def __enter__(self):
        return self
    
    def __exit__(self):
        self.free()
    
    def __iter__(self):
        ''''''
    
    def __getitem__(self, i):
        ''''''
    
    def __len__(self):
        return self._len

    def isempty(self) -> bool:
        return self._len <= 0
    
    def getorginalitem(self, key : Union[Any, int], additionalds : Optional["DSIterator"]) -> Union[List, numpy.ndarray]:
        '''
        Returns the element based on implantation and additionalds.
        It could just return the element at position key within the dataset 
        or determine the element returned.         
        '''
        return self[key]
    
    @property
    def size(self):
        ''''''
        
    @property
    def dataset(self) -> h5py.Dataset:
        ''''''
        
    @property
    def large(self) -> bool:
        '''
        Returns true if it is a large dataset
        '''
        return False
    
    def free(self):
        ''''''
        pass
    
    def astype(self, astype) -> "DSIterator":
        raise NotImplementedError(f"'astype' not supported for {self}")
    
    def index(self, value) -> int:
        raise NotImplementedError(f"'index' not supported for {self}")
   
   
    def tolist(self) -> List[numpy.ndarray[any]]:
        raise NotImplementedError(f"'tolist' not supported for {self}")
   
    def toarray(self) -> numpy.ndarray[Any, numpy.ndarray[any]]:
        raise NotImplementedError(f"'toarray' not supported for {self}")
   
    def __str__(self) -> str:
        return f"DSIterator{{name:{self.name}, shape:{self.shape}}}"

