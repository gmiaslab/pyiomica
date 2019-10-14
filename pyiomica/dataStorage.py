'''Data storage functions'''


import h5py
import json
import pickle

from .globalVariables import *

from .extendedDataFrame import DataFrame
from . import utilityFunctions


def read(fileName, withPKLZextension = True, hdf5fileName = None, jsonFormat = False):

    """Read object from a file recorded by function "write". Pandas and Numpy objects are
    read from HDF5 file when provided, otherwise attempt to read from PKLZ file.

    Parameters:
        fileName: str
            Path of directories ending with the file name

        withPKLZextension: boolean, Default True
            Add ".pklz" to a pickle file

        hdf5fileName: str, Default None
            Path of directories ending with the file name. 
            If None then data is read from a pickle file

        jsonFormat: boolean, Default False
            Save data into compressed json file 
    
    Returns:
        data 
            Data object to write into a file

    Usage:
        exampleDataFrame = read('/dir1/exampleDataFrame', hdf5fileName='/dir2/data.h5')
    """

   
    if jsonFormat:
        utilityFunctions.createDirectories("/".join(fileName.split("/")[:-1]))

        with gzip.GzipFile(fileName, 'r') as tempFile:
            data = json.loads(tempFile.read().decode('utf-8'))

        return data

    if hdf5fileName!=None:
        if not os.path.isfile(hdf5fileName):
            print(hdf5fileName, 'not found.')
            return None

        hdf5file = h5py.File(hdf5fileName, 'r')
        
        key = os.path.basename(fileName)
        if key in hdf5file:
            if hdf5file[key].attrs['gtype']=='pd':
                return DataFrame(pd.read_hdf(hdf5fileName, key=key, mode='r'))

        key = 'arrays/' + os.path.basename(fileName)
        if key in hdf5file:
            if hdf5file[key].attrs['gtype']=='np':
                return hdf5file[key].value
        
        searchPickled = print(os.path.basename(fileName), 'not found in', hdf5fileName)

    if hdf5fileName==None or ('searchPickled' in locals()):  
        if not os.path.isfile(fileName + ('.pklz' if withPKLZextension else '')):
            print(fileName + ('.pklz' if withPKLZextension else ''), 'not found.')
            return None

        with gzip.open(fileName + ('.pklz' if withPKLZextension else ''),'rb') as temp_file:
            data = pickle.load(temp_file)

    return data

def write(data, fileName, withPKLZextension = True, hdf5fileName = None, jsonFormat = False):

    """Write object into a file. Pandas and Numpy objects are recorded in HDF5 format
    when 'hdf5fileName' is provided otherwise pickled into a new file.

    Parameters:
        data: any type
            Data object to write into a file

        fileName: str
            Path of directories ending with the file name

        withPKLZextension: boolean, Default True
            Add ".pklz" to a pickle file

        hdf5fileName: str, Default None
            Path of directories ending with the file name. If None then data is pickled

        jsonFormat: boolean, Default False
            Save data into compressed json file 

    Returns:
        None

    Usage:
        write(exampleDataFrame, '/dir1/exampleDataFrame', hdf5fileName='/dir2/data.h5')
    """

    if jsonFormat:
        utilityFunctions.createDirectories("/".join(fileName.split("/")[:-1]))

        with gzip.GzipFile(fileName, 'w') as tempFile:
            tempFile.write(json.dumps(data).encode('utf-8'))

        return None

    if hdf5fileName!=None and type(data) in [pd.DataFrame, DataFrame]:
        utilityFunctions.createDirectories("/".join(hdf5fileName.split("/")[:-1]))
        key=os.path.basename(fileName)

        pd.DataFrame(data).to_hdf(hdf5fileName, key=key, mode='a', complevel=6, complib='zlib')

        hdf5file = h5py.File(hdf5fileName, 'a')
        hdf5file[key].attrs['gtype'] = 'pd'
    elif hdf5fileName!=None and type(data) is np.ndarray:
        utilityFunctions.createDirectories(hdf5fileName)
        hdf5file = h5py.File(hdf5fileName, 'a')
        key = 'arrays/' + fileName.split("/")[-1]
        data = data.astype(float)
        if not key in hdf5file:
            hdf5file.create_dataset(key, data=data, maxshape=tuple([None]*len(data.shape)), dtype=data.dtype,
                                    compression='gzip', compression_opts=6)
        else:
            dataset = hdf5file[key]
            if dataset.shape!=data.shape:
                dataset.resize(data.shape)
            dataset[...] = data
        hdf5file[key].attrs['gtype'] = 'np'
    else:
        if hdf5fileName!=None:
            print('HDF5 format is not supported for data type:', type(data))
            print('Recording data to a pickle file.')

        utilityFunctions.createDirectories("/".join(fileName.split("/")[:-1]))

        with gzip.open(fileName + ('.pklz' if withPKLZextension else ''),'wb') as temp_file:
            pickle.dump(data, temp_file, protocol=4)

    return None

