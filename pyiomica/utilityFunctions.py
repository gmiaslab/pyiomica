'''Utility functions'''


from .globalVariables import *

def readMathIOmicaData(fileName):

    '''Read text files exported by MathIOmica and convert to Python data

    Parameters:
        fileName: str
            Path of directories and name of the file containing data

    Returns:
        data
            Python data

    Usage:
        data = readMathIOmicaData("../../MathIOmica/MathIOmica/MathIOmicaData/ExampleData/rnaExample")
    '''

    if os.path.isfile(fileName):
        with open(fileName, 'r') as tempFile:
            data = tempFile.read()

        data = data.replace('\n','').replace('{','(').replace('}',')').replace('->',':').replace('|>','}')
        data = data.replace('<|','{').replace('^','*').replace('`','*').replace('Missing[]','"Missing[]"')
        data = data.replace("\\",'')
    else:
        print('File not found (%s)'%(fileName))

    returning = None

    try:
        returning = eval(data)
    except:
        print('Error occured while converting data (%s)'%(fileName))

    return returning

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
        createDirectories("/".join(fileName.split("/")[:-1]))

        with gzip.GzipFile(fileName, 'r') as tempFile:
            data = json.loads(tempFile.read().decode('utf-8'))

        return data

    if hdf5fileName!=None:
        if not os.path.isfile(hdf5fileName):
            print(hdf5fileName, 'not found.')
            return None

        hdf5file = h5py.File(hdf5fileName, 'r')
        
        key = fileName.split("/")[-1]
        if key in hdf5file:
            if hdf5file[key].attrs['gtype']=='pd':
                return pd.read_hdf(hdf5fileName, key=key, mode='r')

        key = 'arrays/' + fileName.split("/")[-1]
        if key in hdf5file:
            if hdf5file[key].attrs['gtype']=='np':
                return hdf5file[key].value
        
        searchPickled = print(fileName.split("/")[-1], 'not found in', hdf5fileName)

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
        createDirectories("/".join(fileName.split("/")[:-1]))

        with gzip.GzipFile(fileName, 'w') as tempFile:
            tempFile.write(json.dumps(data).encode('utf-8'))

        return None

    if hdf5fileName!=None and type(data) in [pd.DataFrame, pd.Series]:
        createDirectories("/".join(hdf5fileName.split("/")[:-1]))
        key=fileName.split("/")[-1]
        
        pd.DataFrame(data=data.values.copy().astype(float), 
                     index=data.index, 
                     columns=data.columns).to_hdf(hdf5fileName, key=key, mode='a', complevel=6, complib='zlib')

        hdf5file = h5py.File(hdf5fileName, 'a')
        hdf5file[key].attrs['gtype'] = 'pd'
    elif hdf5fileName!=None and type(data) is np.ndarray:
        createDirectories(hdf5fileName)
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

        createDirectories("/".join(fileName.split("/")[:-1]))

        with gzip.open(fileName + ('.pklz' if withPKLZextension else ''),'wb') as temp_file:
            pickle.dump(data, temp_file, protocol=4)

    return None

def runCPUs(NumberOfAvailableCPUs, func, list_of_tuples_of_func_params):

    """Parallelize function call with multiprocessing.Pool.

    Parameters:
        NumberOfAvailableCPUs: int
            Number of processes to create

        func: function
            Function to apply, must take at most one argument

        list_of_tuples_of_func_params: list
            Function parameters

    Returns:
        2d numpy.array
            Results of func in a numpy array

    Usage:
        results = runCPUs(4, pAutocorrelation, [(times[i], data[i], allTimes) for i in range(10)])
    """

    instPool = multiprocessing.Pool(processes = NumberOfAvailableCPUs)
    return_values = instPool.map(func, list_of_tuples_of_func_params)
    instPool.close()
    instPool.join()

    return np.vstack(return_values)

def createReverseDictionary(inputDictionary):

    """Efficient way to create a reverse dictionary from a dictionary.
    Utilizes Pandas.Dataframe.groupby and Numpy arrays indexing.
    
    Parameters: 
        inputDictionary: dictionary
            Dictionary to reverse

    Returns:
        dictionary
            Reversed dictionary

    Usage:
        revDict = createReverseDictionary(Dict)
    """

    keys, values = np.array(list(inputDictionary.keys())), np.array(list(inputDictionary.values()))
    df = pd.DataFrame(np.array([[keys[i], value] for i in range(len(keys)) for value in values[i]]))
    dfGrouped = df.groupby(df.columns[1])
    keys, values = list(dfGrouped.indices.keys()), list(dfGrouped.indices.values())
    GOs = df.values.T[0]

    return dict(zip(keys, [GOs[value].tolist() for value in values]))

def createDirectories(path):

    """Create a path of directories, unless the path already exists.

    Parameters:
        path: str
            Path directory

    Returns:
        None

    Usage:
        createDirectories("/pathToFolder1/pathToSubFolder2")
    """

    if path=='':
        return None

    if not os.path.exists(path):
        os.makedirs(path)

    return None

