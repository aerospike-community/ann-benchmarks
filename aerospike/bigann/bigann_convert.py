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
        
    @staticmethod
    def get_dataset_fn(dataset_name: str, folder:str = "data") -> str:
        """
        Returns the full file path for a given dataset name in the data directory.
        
        Args:
            dataset_name (str): The name of the dataset.
        
        Returns:
            str: The full file path of the dataset.
        """
        if not os.path.exists(folder):
            os.mkdir(folder)
            
        filename, fileext = os.path.splitext(dataset_name)
        filenamewext : str = dataset_name
        
        if fileext is None or not fileext:
            filenamewext = f"{filename}.hdf5"
            
        if (filenamewext[0] == os.path.sep
                or filenamewext.startswith(f"{folder}{os.path.sep}")
                or filenamewext.startswith(f".{os.path.sep}")):
            splitpath = os.path.split(filename)
            return filenamewext, splitpath[1]
    
        return os.path.join(folder, filenamewext)
    
    def __init__(self, runtimeArgs: argparse.Namespace, ds : DatasetCompetitionFormat) -> None:
        
        self._bigann_ds = ds
        self._hdf_filepath : str = BigAnnConvert.get_dataset_fn(runtimeArgs.hdf, BASEDIR)
        
        self._bigann_dataset : np.ndarray
        self._bigann_query : np.ndarray
        self._bigann_neighbors : np.ndarray
        self._bigann_distances : np.ndarray
        self._bigann_searchtype : str
        self._bigann_default_search_count : int
        
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
        
        gettasks = []
        
        gettasks.append(self._bigann_getdataset())
        gettasks.append(self._bigann_getquery())
        gettasks.append(self._bigann_getnbrdists())
        
        await asyncio.gather(*gettasks)
        
        self._hdf_dimension = self._bigann_dataset.shape[1]
        self._bigann_searchtype = str(self._bigann_ds.search_type())
        self._bigann_default_search_count = int(self._bigann_ds.default_count())
        
    async def create_hdf(self) -> None:
        import h5py        
        from string import digits
        
        with h5py.File(self._hdf_filepath, "w") as f:
            f.attrs["type"] = self._hdf_type
            f.attrs["sourcedataset"] = self._bigann_ds.short_name()
            f.attrs["distance"] = self._hdf_distance
            f.attrs["dimension"] = self._hdf_dimension
            f.attrs["metrictype"] = self._bigann_searchtype
            f.attrs["point_type"] = self._bigann_dataset[0].dtype.name.rstrip(digits)
            f.attrs["defaultsearchcount"] = self._bigann_default_search_count
            print(f"train size: {self._bigann_dataset.shape[0]} * {self._bigann_dataset.shape[1]}")
            print(f"test size:  {self._bigann_query.shape[0]} * {self._bigann_query.shape[1]}")
            f.create_dataset("train", data=self._bigann_dataset)
            f.create_dataset("test", data=self._bigann_query)
            f.create_dataset("neighbors", data=self._bigann_neighbors)
            f.create_dataset("distances", data=self._bigann_distances)
            hdfpath = f.filename
            print(f"Created HDF dataset '{hdfpath}'")