import h5py
import numpy
from typing import Any, Callable, Dict, Tuple, Optional, List, Union

from dsiterator import DSIterator

class DSPKIterator(DSIterator):
    
    def __init__(self, dataset : h5py.Dataset, name : str):

        self._dataset : h5py.Dataset = dataset.get(name)
        if self._dataset is None:
            super().__init__((0, 0), name)
            self._pklist = []
        else:
            super().__init__(self._dataset.shape,
                             name)        
            self._pklist = list(self.dataset)
               
        if (len(self._pklist) > 1 
                and numpy.issubdtype(self._pklist[0], numpy.integer)
                and numpy.issubdtype(self._pklist[-1], numpy.integer)
                and self._pklist[0] == 0):
            numRange = list(range(0, self._pklist[-1]+1))
            self._consecutivenbrs = self._pklist == numRange
        else:
            self._consecutivenbrs = False
                
    def __iter__(self):
        ''''''
        return self._pklist.__iter__()
    
    def __getitem__(self, i):
        ''''''
        return i if self._consecutivenbrs or self._len == 0 else self._pklist[i]
    
    @property
    def size(self):
        ''''''
        return self._pklist.__sizeof__()
        
    @property
    def dataset(self) -> h5py.Dataset:
        ''''''
        return self._dataset
    
    def free(self):
        ''''''
        self._dataset = None
        
    def index(self, value) -> int:
        return self._pklist.index(value)
   
    def getorginalitem(self, key : Union[Any, int], trainingds : Optional["DSIterator"]) -> Union[List, numpy.ndarray]:
        '''
        Returns the orginal vector from the Training Dataset. 
        Note: trainingds cannot be None
        '''
        try:
            if self._consecutivenbrs:
                return trainingds[key]

            fndidx = self._pklist.index(key)
            return trainingds[fndidx]
        except IndexError as e:
            return numpy.zeros(len(trainingds[0]))
        except ValueError as e:
            return numpy.zeros(len(trainingds[0]))
    
    def tolist(self) -> List[numpy.ndarray[any]]:
        return self._pklist
    
    def toarray(self) -> numpy.ndarray[Any, numpy.ndarray[any]]:
        return numpy.array(self._pklist)
    
    def __str__(self) -> str:
        return f"DSPKIterator{{name:{self.name}, shape:{self.shape}}}"
        