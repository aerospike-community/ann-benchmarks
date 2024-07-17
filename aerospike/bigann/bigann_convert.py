import asyncio
import os
import argparse

import numpy as np
from .datasets import DatasetCompetitionFormat, BASEDIR

class BigAnnConvert():
    
    @staticmethod
    def parse_arguments(parser: argparse.ArgumentParser) -> None:
        '''
        Adds the arguments required to create an ANN HDF file. 
        '''
        
        parser.add_argument(
            "--hdf",
            metavar="HDFFILE",
            help="A HDF file name that will be created in the 'data' folder by converting a Big ANN file Format files",
            type=str,
            required=True,
        )
        
    async def __aenter__(self):
        return self
 
    async def __aexit__(self, *args):
        pass      
        
    def __init__(self, runtimeArgs: argparse.Namespace, ds : DatasetCompetitionFormat) -> None:
        
        self._bigann_ds = ds
        self._hdf_filepath : str = os.path.join(BASEDIR, runtimeArgs.hdf)
        
        self._bigann_dataset : np.ndarray
        self._bigann_query : np.ndarray
        self._bigann_neighbors : np.ndarray
        self._bigann_distances : np.ndarray
        
        if os.path.exists(self._hdf_filepath):
            print(f"Warn: ANN HDF File '{self._hdf_filepath}' exist and will be overwritten")
        
    async def _bigann_getdataset(self) -> None:
        self._bigann_dataset = self._bigann_ds.get_dataset()
        
    async def _bigann_getquery(self) -> None:
        self._bigann_query = self._bigann_ds.get_queries()
        
    async def _bigann_getnbrdists(self) -> None:
        self._bigann_neighbors, self._bigann_distances = self._bigann_ds.get_groundtruth()
        
    async def bigann_getinfo(self) -> None:
        
        self._hdf_distance = self._bigann_ds.distance()
        self._hdf_type = self._bigann_ds.data_type()
        self._hdf_dimension = self._bigann_ds.default_count()
        
        gettasks = []
        
        gettasks.append(self._bigann_getdataset())
        gettasks.append(self._bigann_getquery())
        gettasks.append(self._bigann_getnbrdists())
        
        await asyncio.gather(*gettasks)
        
        print("done")