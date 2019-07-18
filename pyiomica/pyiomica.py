"""
PyIOmica is a general omics package with multiple tools for analyzing omics data.

Usage:
    from pyiomica import pyiomica
Notes:
    For additional information visit: https://github.com/gmiaslab/pyiomica and https://mathiomica.org by G. Mias Lab
"""


print("Loading PyIOmica (https://github.com/gmiaslab/pyiomica by G. Mias Lab)")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import matplotlib.patches
import matplotlib.collections
from matplotlib import cm

import numpy as np
import pandas as pd
import networkx as nx

import appdirs

import zipfile
import os
import pickle
import gzip
import copy
import multiprocessing
import urllib.request
import shutil
import h5py
import pymysql
import datetime
import json
import numba

import scipy
import scipy.signal
import scipy.stats
import scipy.cluster.hierarchy as hierarchy
from scipy.spatial.distance import pdist, squareform
from scipy.interpolate import UnivariateSpline

import sklearn
from sklearn.cluster import AgglomerativeClustering, KMeans, MiniBatchKMeans
from sklearn.preprocessing import quantile_transform
from sklearn.metrics import silhouette_score, silhouette_samples, pairwise_distances, adjusted_rand_score
from sklearn.manifold import TSNE

from importlib import resources

numba.config.NUMBA_DEFAULT_NUM_THREADS = 1

### Global constants ##############################################################################
"""Package directory"""
PackageDirectory = os.path.split(__file__)[0]

try:
    with resources.path('pyiomica.data','__init__.py') as readIn:
        ConstantPyIOmicaDataDirectory = os.path.dirname(readIn)

    UserDataDirectory = os.path.join(PackageDirectory, "data")
except:
    print('Cannot access default site package directory, will instead use ', ConstantPyIOmicaDataDirectory)

    UserDataDirectory = appdirs.user_data_dir('pyiomica', 'gmiaslab')

"""ConstantPyIOmicaDataDirectory is a global variable pointing to the PyIOmica data directory."""
ConstantPyIOmicaDataDirectory = UserDataDirectory

"""ConstantPyIOmicaExamplesDirectory is a global variable pointing to the PyIOmica example data directory."""
ConstantPyIOmicaExamplesDirectory = os.path.join(UserDataDirectory, "ExampleData")

"""ConstantPyIOmicaExampleVideosDirectory is a global variable pointing to the PyIOmica example videos directory."""
ConstantPyIOmicaExampleVideosDirectory = os.path.join(UserDataDirectory, "ExampleVideos")

for path in [ConstantPyIOmicaDataDirectory, ConstantPyIOmicaExamplesDirectory, ConstantPyIOmicaExampleVideosDirectory]:
    if not os.path.exists(path):
        os.makedirs(path)

"""ConstantGeneDictionary is a global gene/protein dictionary variable typically created by GetGeneDictionary."""
ConstantGeneDictionary = None

###################################################################################################

### Utility functions #############################################################################
def createDirectories(path):

    """Create a path of directories, unless the path already exists.

    Args:
        path: path directory

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


def runCPUs(NumberOfAvailableCPUs, func, list_of_tuples_of_func_params):

    """Parallelize function call with multiprocessing.Pool.

    Args:
        NumberOfAvailableCPUs: number of processes to create
        func: function to apply, must take at most one argument
        list_of_tuples_of_func_params: function parameters

    Returns:
        Results of func in a numpy array

    Usage:
        results = runCPUs(4, pAutocorrelation, [(times[i], data[i], allTimes) for i in range(10)])
    """

    instPool = multiprocessing.Pool(processes = NumberOfAvailableCPUs)
    return_values = instPool.map(func, list_of_tuples_of_func_params)
    instPool.close()
    instPool.join()

    return np.vstack(return_values)


def write(data, fileName, withPKLZextension = True, hdf5fileName = None, jsonFormat = False):

    """Write object into a file. Pandas and Numpy objects are recorded in HDF5 format
    when 'hdf5fileName' is provided otherwise pickled into a new file.

    Args:
        data: data object to write into a file
        fileName: path of directories ending with the file name
        withPKLZextension: add ".pklz" to a pickle file
        hdf5fileName: path of directories ending with the file name. If None then data is pickled.
        jsonFormat: save data into compressed json file 

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


def read(fileName, withPKLZextension = True, hdf5fileName = None, jsonFormat = False):

    """Read object from a file recorded by function "write". Pandas and Numpy objects are
    read from HDF5 file when provided, otherwise attempt to read from PKLZ file.

    Args:
        fileName: path of directories ending with the file name
        withPKLZextension: add ".pklz" to a pickle file
        hdf5fileName: path of directories ending with the file name. If None then data is pickled
        jsonFormat: save data into compressed json file 
    
    Returns:
        data: data object to write into a file

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


def createReverseDictionary(inputDictionary):

    """Efficient way to create a reverse dictionary from a dictionary.
    Utilizes Pandas.Dataframe.groupby and Numpy arrays indexing.
    
    Args: 
        inputDictionary: a dictionary to reverse

    Returns:
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


def readMathIOmicaData(fileName):

    '''Read text files exported by MathIOmica and convert to Python data

    Args:
        fileName: path of directories and name of the file containing data

    Returns:
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

###################################################################################################



### Annotations and Enumerations ##################################################################

def internalAnalysisFunction(data, multiCorr, MultipleList,  OutputID, InputID, Species, totalMembers,
                            pValueCutoff, ReportFilterFunction, ReportFilter, TestFunction, HypothesisFunction, FilterSignificant,
                            AssignmentForwardDictionary, AssignmentReverseDictionary, prefix, infoDict):

    """Analysis for Multi-Omics or Single-Omics input list
    The function is used internally and not intended to be used directly by user.
    
    Usage:
        Intended for internal use
    """
    
    listData = data[list(data.keys())[0]]

    #If input data was a list of genes, convert it to pairs with label "Generic"
    if not type(listData[0]) is list:
        listData = [[item, 'Unknown'] for item in listData]
    else:
        if len(listData[0])==1:
            listData = [[item[0], 'Unknown'] for item in listData]

    #Get IDs for each gene
    dataForGeneTranslation = [item[0] if type(item) is list else item for item in listData]
    IDs = GeneTranslation(dataForGeneTranslation, OutputID, ConstantGeneDictionary, InputID = InputID,  Species = Species)[OutputID]

    #Remove Missing IDs
    [item.remove('Missing') if 'Missing' in item else None for item in IDs]

    #Might have put in mixed IDs correspocting to different omics types still, give option to user;
    #Extract the translations, match the ones that intersect for different modes (e.g. RNA/Protein label (last element) must be same)
    membersWithAssociations = {}
    for gene, geneIDs in zip(listData, IDs):

        if len(gene)==4:
            geneKey, _, _, geneOmi = gene
        elif len(gene)==3:
            geneKey, geneOmi, _ = gene
        else:
            geneKey, geneOmi = gene

        if MultipleList:
            geneKey += '_' + geneOmi

        if geneKey in membersWithAssociations.keys():
            labels = membersWithAssociations[geneKey][0]
            if not geneOmi in labels:
                labels.append(geneOmi)
        else:
            membersWithAssociations[geneKey] = [[geneOmi],[]]

        for ID in geneIDs:
            if prefix + ID in list(AssignmentForwardDictionary.keys()):
                for AssignmentID in list(AssignmentForwardDictionary[prefix + ID]):
                    if not AssignmentID in membersWithAssociations[geneKey][1]:
                        membersWithAssociations[geneKey][1].append(AssignmentID)
        if len(membersWithAssociations[geneKey][1])==0:
            membersWithAssociations.pop(geneKey)

    allAssignmentIDs = []
    for thisGeneGOlist in [item[1] for item in list(membersWithAssociations.values())]:
        for AssignmentID in thisGeneGOlist:
            if not AssignmentID in allAssignmentIDs:
                allAssignmentIDs.append(AssignmentID)

    testCats = {}
    for AssignmentID in allAssignmentIDs:
        countsInList = len(membersWithAssociations.keys())
        countsInFamily = multiCorr*len(AssignmentReverseDictionary[AssignmentID])
        countsInMembers = np.sum([AssignmentID in item[1] for item in membersWithAssociations.values()])
        whereGeneHits = [AssignmentID in item[1] for item in membersWithAssociations.values()]
        listOfGenesHit = [[item, membersWithAssociations[item][0]] for item in np.array(list(membersWithAssociations.keys()))[whereGeneHits]]
        testValue = TestFunction(countsInList, countsInFamily, multiCorr*totalMembers, countsInMembers)

        testCats[AssignmentID] = [testValue, [countsInList, countsInFamily, multiCorr*totalMembers, countsInMembers], infoDict[AssignmentID], listOfGenesHit]

    correctedpValues = dict(zip(allAssignmentIDs, HypothesisFunction([item[0] for item in list(testCats.values())], pValueCutoff).T))

    for AssignmentID in allAssignmentIDs:
        testCats[AssignmentID][0] = [testCats[AssignmentID][0], correctedpValues[AssignmentID][1], correctedpValues[AssignmentID][2]]

    ResultsHCct = testCats

    #Length filter
    whatIsFilteredLength = ReportFilterFunction(np.array([item[1][3] for item in list(ResultsHCct.values())]), ReportFilter)

    #Significance filter
    whatIsFilteredSignif = np.array([(item[0][2] if FilterSignificant else True) for item in list(ResultsHCct.values())]).astype(bool)

    #Combined filter
    whatIsFiltered = whatIsFilteredLength * whatIsFilteredSignif

    returning = dict(zip(list(np.array(list(ResultsHCct.keys()))[whatIsFiltered]),list(np.array(list(ResultsHCct.values()))[whatIsFiltered])))

    return {list(data.keys())[0]: returning}


def OBOGODictionary(FileURL="http://purl.obolibrary.org/obo/go/go-basic.obo", ImportDirectly=False, PyIOmicaDataDirectory=None, OBOFile="goBasicObo.txt"):

    """Generate Open Biomedical Ontologies (OBO) Gene Ontology (GO) vocabulary dictionary.
    
    Args: 
        FileURL: provides the location of the Open Biomedical Ontologies (OBO) Gene Ontology (GO) 
    file in case this will be downloaded from the web
        ImportDirectly: import from URL regardles is the file already exists
        PyIOmicaDataDirectory: path of directories to data storage
        OBOFile: name of file to store data in (file will be zipped)

    Returns:
        Dictionary of definitions

    Usage:
        OBODict = OBOGODictionary()
    """

    global ConstantPyIOmicaDataDirectory

    PyIOmicaDataDirectory = ConstantPyIOmicaDataDirectory if PyIOmicaDataDirectory==None else PyIOmicaDataDirectory
    fileGOOBO = os.path.join(PyIOmicaDataDirectory, OBOFile)
    fileGOOBOgz = fileGOOBO + '.gz'

    #import the GO OBO file: we check if the OBO file Exist, if not, attempt to download and create it
    if not os.path.isfile(fileGOOBOgz):
        print("Did Not Find Annotation Files, Attempting to Download...")
        ImportDirectly = True

    if os.path.isfile(fileGOOBO):
        os.remove(fileGOOBO)

    if ImportDirectly:
        if os.path.isfile(fileGOOBOgz):
            os.remove(fileGOOBOgz)

        urllib.request.urlretrieve(FileURL.strip('"'), fileGOOBO)

        if os.path.isfile(fileGOOBO):
            print("Created Annotation Files at ", fileGOOBO)
        else:
            print("Did Not Find Annotation Files, Aborting Process")
            return

        with open(fileGOOBO, 'rb') as fileIn:
            with gzip.open(fileGOOBOgz, 'wb') as fileOut:
                shutil.copyfileobj(fileIn, fileOut)

        print("Compressed local file with GZIP.")

        os.remove(fileGOOBO)

    with gzip.open(fileGOOBOgz, 'r') as tempFile:
        inputFile = tempFile.readlines()

    inputFile = [item.decode() for item in inputFile]

    #Find keys "accessions (id):" and "name:" and "namespace" but extract their corresponding values in a list and map them to their corresponding [Term] positions, 
    #Once the "accessions (id):" and its corresponding "name:" in a list, make an association between them,
    #so you can search this association using the key "accessions (id):" to get the value "name:" and "namespace"
    outDictionary = {}

    for position in np.where([item=='[Term]\n'for item in inputFile])[0]:
        
        def getValue(index):

            return inputFile[position + index].strip(['id:', 'name:', 'namespace:'][index - 1]).strip('\n').strip()

        outDictionary[getValue(1)] = [getValue(2), getValue(3)]
    
    return outDictionary


def GetGeneDictionary(geneUCSCTable = None, UCSCSQLString = None, UCSCSQLSelectLabels = None,
                    ImportDirectly = False, Species = "human", KEGGUCSCSplit = [True,"KEGG Gene ID"]):
    
    """Create an ID/accession dictionary from a UCSC search - typically of gene annotations.
    
    Args: 
        geneUCSCTable: path to a geneUCSCTable file
        UCSCSQLString: an association to be used to obtain data from the UCSC Browser tables. The key of the association must 
    match the Species option value used (default: human). The value for the species corresponds to the actual MySQL command used
        UCSCSQLSelectLabels: an association to be used to assign key labels for the data improted from the UCSC Browser tables. 
    The key of the association must match the Species option value used (default: human). The value is a multi component string 
    list corresponding to the matrices in the data file, or the tables used in the MySQL query provided by UCSCSQLString
        ImportDirectly: import from URL regardles is the file already exists
        Species: species considered in the calculation, by default corresponding to human
        KEGGUCSCSplit: a two component list, {True/False, label}. If the first component is set to True the initially imported KEGG IDs, 
    identified by the second component label,  are split on + string to fix nomenclature issues, retaining the string following +

    Returns:
        Dictionary

    Usage:
        geneDict = GetGeneDictionary()
    """
       
    UCSCSQLSelectLabels = {"human": ["UCSC ID", "UniProt ID", "Gene Symbol", 
        "RefSeq ID", "NCBI Protein Accession", "Ensembl ID", 
        "KEGG Gene ID", "HGU133Plus2 Affymetrix ID"]}

    #Update these to match any change in query
    #Query for UCSC SQL server
    UCSCSQLString = {"human": 
       "SELECT hg19.kgXref.kgID, hg19.kgXref.spID, \
        hg19.kgXref.geneSymbol, hg19.kgXref.refseq, hg19.kgXref.protAcc, \
        hg19.knownToEnsembl.value, hg19.knownToKeggEntrez.keggEntrez, \
        hg19.knownToU133Plus2.value FROM hg19.kgXref LEFT JOIN \
        hg19.knownToEnsembl ON hg19.kgXref.kgID = hg19.knownToEnsembl.name \
        LEFT JOIN hg19.knownToKeggEntrez ON hg19.kgXref.kgID = \
        hg19.knownToKeggEntrez.name LEFT JOIN hg19.knownToU133Plus2 ON \
        hg19.kgXref.kgID = hg19.knownToU133Plus2.name"}

    global ConstantPyIOmicaDataDirectory

    PyIOmicaDataDirectory = ConstantPyIOmicaDataDirectory

    if geneUCSCTable is None:
        geneUCSCTable = os.path.join(PyIOmicaDataDirectory, Species + "GeneUCSCTable" + ".json.gz")
       
    #If the user asked us to import directly, import directly with SQL, otherwise, get it from a directory they specify
    if not os.path.isfile(geneUCSCTable):
        print("Did Not Find Gene Translation Files, Attempting to Download from UCSC...")
        ImportDirectly = True
    else:
        termTable = read(geneUCSCTable, jsonFormat=True)[1]
        termTable = np.array(termTable)

    if ImportDirectly:
        #Connect to the database from UCSC
        ucscDatabase = pymysql.connect("genome-mysql.cse.ucsc.edu","genomep","password")

        if ucscDatabase==None:
            print("Could not establish connection to UCSC. Please try again or add the dictionary manually at ", geneUCSCTable)
            return

        #Prepare a cursor object using cursor() method
        ucscDatabaseCursor = ucscDatabase.cursor()

        try:
            #Execute the SQL command
            ucscDatabaseCursor.execute(UCSCSQLString[Species])

            #Fetch all the rows in a list of lists.
            termTable = ucscDatabaseCursor.fetchall()

        except:
            print ("Error: unable to fetch data")

        termTable = np.array(termTable).T
        termTable[np.where(termTable=="")] = None

        #Get all the terms we are going to need, import with SQL the combined tables,and export with a time stamp
        write((datetime.datetime.now().isoformat(), termTable.tolist()), geneUCSCTable, jsonFormat=True)

        #Close SQL connection
        ucscDatabase.close()

        if os.path.isfile(geneUCSCTable):
            print("Created Annotation Files at ", geneUCSCTable)
        else:
            print("Did Not Find Annotation Files, Aborting Process")
            return

    returning = {Species : dict(zip(UCSCSQLSelectLabels[Species],termTable))}
    
    if KEGGUCSCSplit[0]:
        returning[Species][KEGGUCSCSplit[1]] = np.array([item.split("+")[1] if item!=None else item for item in returning[Species][KEGGUCSCSplit[1]]])
    
    return returning


def GOAnalysisAssigner(PyIOmicaDataDirectory = None, ImportDirectly = False, BackgroundSet = [], Species = "human",
                        LengthFilter = None, LengthFilterFunction = np.greater_equal, GOFileName = None, GOFileColumns = [2, 5], 
                        GOURL = "http://current.geneontology.org/annotations/"):
    
    """Download and create gene associations and restrict to required background set.

    Args: 
        PyIOmicaDataDirectory: the directory where the default package data is stored
        ImportDirectly: import from URL regardles is the file already exists
        BackgroundSet: background list to create annotation projection to limited background space, involves
    considering pathways/groups/sets and that provides a list of IDs (e.g. gene accessions) that should 
    be considered as the background for the calculation
        Species: species considered in the calculation, by default corresponding to human
        LengthFilterFunction: performs computations of membership in pathways/ontologies/groups/sets, 
    that specifies which function to use to filter the number of members a reported category has 
    compared to the number typically provided by LengthFilter 
        LengthFilter: argument for LengthFilterFunction
        GOFileName: the name for the specific GO file to download from the GOURL if option ImportDirectly is set to True
        GOFileColumns: columns to use for IDs and GO:accessions respectively from the downloaded GO annotation file, 
    used when ImportDirectly is set to True to obtain a new GO association file
        GOURL: the location (base URL) where the GO association annotation files are downloaded from

    Returns:
        IDToGO and GOToID dictionary

    Usage:
        GOassignment = GOAnalysisAssigner()
    """

    global ConstantPyIOmicaDataDirectory

    PyIOmicaDataDirectory = ConstantPyIOmicaDataDirectory if PyIOmicaDataDirectory==None else PyIOmicaDataDirectory

    #If the user asked us to import PyIOmicaDataDirectoryectly, import PyIOmicaDataDirectoryectly from GO website, otherwise, get it from a PyIOmicaDataDirectoryectory they specify
    file = "goa_" + Species + ".gaf.gz" if GOFileName==None else GOFileName
    localFile =  os.path.join(PyIOmicaDataDirectory, "goa_" + Species + ".gaf")
    localZipFile =  os.path.join(PyIOmicaDataDirectory, "goa_" + Species + ".gaf.gz")
    fileGOAssociations = [os.path.join(PyIOmicaDataDirectory, Species + item + ".json.gz") for item in ["GeneOntAssoc", "IdentifierAssoc"]]

    #We check if the Annotations exist, if not, attempt to download and create them
    if not np.array(list(map(os.path.isfile, fileGOAssociations))).all():
        print("Did Not Find Annotation Files, Attempting to Download...")
        ImportDirectly = True

    if ImportDirectly:
        #Delete existing file
        if os.path.isfile(localFile):
            os.remove(localFile)
        
        urllib.request.urlretrieve(GOURL + file, "\\".join([localZipFile]))
        
        with gzip.open(localZipFile, 'rb') as fileIn:
            with open(localFile, 'wb') as fileOut:
                shutil.copyfileobj(fileIn, fileOut)
        
        #Clean up archive
        os.remove(localZipFile)
        
        with open(localFile, 'r') as tempFile:
            goData = tempFile.readlines()

        #Remove comments by "!" lines
        goData = np.array(goData[np.where(np.array([line[0]!='!' for line in goData]))[0][0]:])
        goData = pd.DataFrame([item.strip('\n').split('\t') for item in goData]).values

        #Remove all entries with missing
        df = pd.DataFrame(goData.T[np.array(GOFileColumns)-1].T)
        df = df[np.count_nonzero(df.values!='', axis=1)==2]

        ##Mark missing values with NaN
        #df.iloc[np.where(df.values=='')] = np.nan

        dfGrouped = df.groupby(df.columns[1])

        keys, values = list(dfGrouped.indices.keys()), list(dfGrouped.indices.values())
        IDs = df.values.T[0]
        geneOntAssoc = dict(zip(keys, [np.unique(IDs[value]).tolist() for value in values]))

        identifierAssoc = createReverseDictionary(geneOntAssoc)

        #Save created annotations geneOntAssoc, identifierAssoc
        write((datetime.datetime.now().isoformat(), geneOntAssoc), fileGOAssociations[0], jsonFormat=True)
        write((datetime.datetime.now().isoformat(), identifierAssoc), fileGOAssociations[1], jsonFormat=True)
        
        os.remove(localFile)

        if np.array(list(map(os.path.isfile, fileGOAssociations))).all():
            print("Created Annotation Files at ", fileGOAssociations)
        else:
            print("Did Not Find Annotation Files, Aborting Process")
            return
    else:
        #Otherwise we get from the user specified PyIOmicaDataDirectoryectory
        geneOntAssoc = read(fileGOAssociations[0], jsonFormat=True)[-1]
        identifierAssoc = read(fileGOAssociations[1], jsonFormat=True)[-1]

    if BackgroundSet!=[]:
        #Using provided background list to create annotation projection to limited background space, also remove entries with only one and missing value
        keys, values = np.array(list(identifierAssoc.keys())), np.array(list(identifierAssoc.values()))
        index = np.where([(((len(values[i])==True)*(values[i][0]!=values[i][0]))==False)*(keys[i] in BackgroundSet) for i in range(len(keys))])[0]
        identifierAssoc = dict(zip(keys[index],values[index]))

        #Create corresponding geneOntAssoc
        geneOntAssoc = createReverseDictionary(identifierAssoc)

    if LengthFilter!=None:
        keys, values = np.array(list(geneOntAssoc.keys())), np.array(list(geneOntAssoc.values()))
        index = np.where(LengthFilterFunction(np.array([len(value) for value in values]), LengthFilter))[0]
        geneOntAssoc = dict(zip(keys[index],values[index]))

        #Create corresponding identifierAssoc
        identifierAssoc = createReverseDictionary(geneOntAssoc)

    return {Species : {"IDToGO": identifierAssoc, "GOToID": geneOntAssoc}}


def obtainConstantGeneDictionary(GeneDictionary, GetGeneDictionaryOptions, AugmentDictionary):
    
    """Obtain gene dictionary - if it exists can either augment with new information or Species or create new, 
    if not exist then create variable.

    Args:
        GeneDictionary: an existing variable to use as a gene dictionary in annotations. 
    If set to None the default ConstantGeneDictionary will be used
        GetGeneDictionaryOptions: a list of options that will be passed to this internal GetGeneDictionary function
        AugmentDictionary: a choice whether or not to augment the current ConstantGeneDictionary global variable or create a new one

    Returns:
        None

    Usage:
        obtainConstantGeneDictionary(None, {}, False)
    """

    global ConstantGeneDictionary
    
    if ConstantGeneDictionary!=None:
        #ConstantGeneDictionary exists
        if AugmentDictionary:
            #Augment ConstantGeneDictionary
            ConstantGeneDictionary = {**ConstantGeneDictionary, **(GetGeneDictionary(**GetGeneDictionaryOptions) if GeneDictionary==None else GeneDictionary)}
        else:
            #Replace ConstantGeneDictionary
            ConstantGeneDictionary = GetGeneDictionary(**GetGeneDictionaryOptions) if GeneDictionary==None else GeneDictionary
    else:
        #Create/load UCSC based translation dictionary - NB global variable or use specified variable
        ConstantGeneDictionary = GetGeneDictionary(**GetGeneDictionaryOptions) if GeneDictionary==None else GeneDictionary

    return None


def GOAnalysis(data, GetGeneDictionaryOptions={}, AugmentDictionary=True, InputID=["UniProt ID","Gene Symbol"], OutputID="UniProt ID",
                 GOAnalysisAssignerOptions={}, BackgroundSet=[], Species="human", OntologyLengthFilter=2, ReportFilter=1, ReportFilterFunction=np.greater_equal,
                 pValueCutoff=0.05, TestFunction=lambda n, N, M, x: 1. - scipy.stats.hypergeom.cdf(x-1, M, n, N), 
                 HypothesisFunction=lambda data, SignificanceLevel: BenjaminiHochbergFDR(data, SignificanceLevel=SignificanceLevel)["Results"], 
                 FilterSignificant=True, OBODictionaryVariable=None,
                 OBOGODictionaryOptions={}, MultipleListCorrection=None, MultipleList=False, GeneDictionary=None):

    """Calculate input data over-representation analysis for Gene Ontology (GO) categories.

    Args:
        data: data to analyze
        GetGeneDictionaryOptions: a list of options that will be passed to this internal GetGeneDictionary function
        AugmentDictionary: a choice whether or not to augment the current ConstantGeneDictionary global variable or create a new one
        InputID:  kind of identifiers/accessions used as input
        OutputID: kind of IDs/accessions to convert the input IDs/accession numbers in the function's analysis
        GOAnalysisAssignerOptions: a list of options that will be passed to the internal GOAnalysisAssigner function
        BackgroundSet: background list to create annotation projection to limited background space, involves
    considering pathways/groups/sets and that provides a list of IDs (e.g. gene accessions) that should be 
    considered as the background for the calculation
        Species: the species considered in the calculation, by default corresponding to human
        OntologyLengthFilter: function that can be used to set the value for which terms to consider in the computation, 
    by excluding GO terms that have fewer items compared to the OntologyLengthFilter value. It is used by the internal
    GOAnalysisAssigner function
        ReportFilter: functions that use pathways/ontologies/groups, and provides a cutoff for membership in ontologies/pathways/groups
    in selecting which terms/categories to return. It is typically used in conjunction with ReportFilterFunction
        ReportFilterFunction: specifies what operator form will be used to compare against ReportFilter option value in 
    selecting which terms/categories to return
        HypothesisFunction: allows the choice of function for implementing multiple hypothesis testing considerations
        FilterSignificant: can be set to True to filter data based on whether the analysis result is statistically significant, 
    or if set to False to return all membership computations
        OBODictionaryVariable: a GO annotation variable. If set to None, OBOGODictionary will be used internally to 
    automatically generate the default GO annotation
        OBOGODictionaryOptions: a list of options to be passed to the internal OBOGODictionary function that provides the GO annotations
        MultipleListCorrection: specifies whether or not to correct for multi-omics analysis. The choices are None, Automatic, 
    or a custom number, e.g protein+RNA
        MultipleList: specifies whether the input accessions list constituted a multi-omics list input that is annotated so
        GeneDictionary: points to an existing variable to use as a gene dictionary in annotations. If set to None 
    the default ConstantGeneDictionary will be used

    Returns:
        Enrichment dictionary

    Usage:
        goExample1 = GOAnalysis(["TAB1", "TNFSF13B", "MALT1", "TIRAP", "CHUK", 
                                "TNFRSF13C", "PARP1", "CSNK2A1", "CSNK2A2", "CSNK2B", "LTBR", 
                                "LYN", "MYD88", "GADD45B", "ATM", "NFKB1", "NFKB2", "NFKBIA", 
                                "IRAK4", "PIAS4", "PLAU"])
    """

    global ConstantGeneDictionary

    #Obtain OBO dictionary with OBOGODictionaryOptions if any. If externally defined use user definition for OBODict Variable
    OBODict = OBOGODictionary(**OBOGODictionaryOptions) if OBODictionaryVariable==None else OBODictionaryVariable

    #Obtain gene dictionary - if it exists can either augment with new information or Species or create new, if not exist then create variable
    obtainConstantGeneDictionary(GeneDictionary, GetGeneDictionaryOptions, AugmentDictionary)
    
    #Get the right GO terms for the BackgroundSet requested and correct Species
    Assignment = GOAnalysisAssigner(BackgroundSet=BackgroundSet, Species=Species , LengthFilter=OntologyLengthFilter) if GOAnalysisAssignerOptions=={} else GOAnalysisAssigner(**GOAnalysisAssignerOptions)
    
    #The data may be a subgroup from a clustering object, i.e. a pd.DataFrame
    if type(data) is pd.DataFrame:
        id = list(data.index.get_level_values('id'))
        source = list(data.index.get_level_values('source'))
        data = [[id[i], source[i]] for i in range(len(data))]

    #If the input is simply a list
    listToggle = True if type(data) is list else False
    data = {'dummy': data} if listToggle else data

    returning = {}

    #Check if a clustering object
    if "linkage" in data.keys():
        if MultipleListCorrection==None:
            multiCorr = 1
        elif MultipleListCorrection=='Automatic':
            multiCorr = 1
            for keyGroup in sorted([item for item in list(data.keys()) if not item=='linkage']):
                for keySubGroup in sorted([item for item in list(data[keyGroup].keys()) if not item=='linkage']):
                    multiCorr = max(max(np.unique(data[keyGroup][keySubGroup]['data'].index.get_level_values('id'), return_counts=True)[1]), multiCorr)
        else:
            multiCorr = MultipleListCorrection

        #Loop through the clustering object, calculate GO for each SubGroup
        for keyGroup in sorted([item for item in list(data.keys()) if not item=='linkage']):
            returning[keyGroup] = {}
            for keySubGroup in sorted([item for item in list(data[keyGroup].keys()) if not item=='linkage']):
                SubGroupMultiIndex = data[keyGroup][keySubGroup]['data'].index
                SubGroupGenes = list(SubGroupMultiIndex.get_level_values('id'))
                SubGroupMeta = list(SubGroupMultiIndex.get_level_values('source'))
                SubGroupList = [[SubGroupGenes[i], SubGroupMeta[i]] for i in range(len(SubGroupMultiIndex))]

                returning[keyGroup][keySubGroup] = internalAnalysisFunction({keySubGroup:SubGroupList},
                                                                     multiCorr, MultipleList,  OutputID, InputID, Species, len(Assignment[Species]["IDToGO"]),
                                                                     pValueCutoff, ReportFilterFunction, ReportFilter, TestFunction, HypothesisFunction, FilterSignificant,
                                                                     AssignmentForwardDictionary=Assignment[Species]['IDToGO'],
                                                                     AssignmentReverseDictionary=Assignment[Species]['GOToID'],
                                                                     prefix='', infoDict=OBODict)[keySubGroup]

    #The data is a dictionary of type {'Name1': [data1], 'Name2': [data2], ...}
    else:
        for key in list(data.keys()):
            if MultipleListCorrection==None:
                multiCorr = 1
            elif MultipleList and MultipleListCorrection=='Automatic':
                multiCorr = max(np.unique([item[0] for item in data[key]], return_counts=True)[1])
            else:
                multiCorr = MultipleListCorrection

            returning.update(internalAnalysisFunction({key:data[key]}, multiCorr, MultipleList,  OutputID, InputID, Species, len(Assignment[Species]["IDToGO"]),
                                                pValueCutoff, ReportFilterFunction, ReportFilter, TestFunction, HypothesisFunction, FilterSignificant,
                                                AssignmentForwardDictionary=Assignment[Species]['IDToGO'],
                                                AssignmentReverseDictionary=Assignment[Species]['GOToID'],
                                                prefix='', infoDict=OBODict))

        #If a single list was provided, return the association for Gene Ontologies
        returning = returning['dummy'] if listToggle else returning

    return returning


def GeneTranslation(InputList, TargetIDList, GeneDictionary, InputID = None, Species = "human"):

    """Use geneDictionary to convert inputList IDs to different annotations as indicated by targetIDList.
    
    Args:
        InputList: list of names
        TargetIDList: target ID list
        GeneDictionary: an existing variable to use as a gene dictionary in annotations. 
    If set to None the default ConstantGeneDictionary will be used
        InputID: the kind of identifiers/accessions used as input
        Species: the species considered in the calculation, by default corresponding to human
    
    Returns:
        Dictionary

    Usage:
        GenDict = GeneTranslation(data, "UniProt ID", ConstantGeneDictionary, InputID = ["UniProt ID","Gene Symbol"],  Species = "human")
    
    """

    if InputID!=None:
        listOfKeysToUse = []
        if type(InputID) is list:
            for key in InputID:
                if key in list(GeneDictionary[Species].keys()):
                    listOfKeysToUse.append(key)
        elif type(InputID) is str:
            listOfKeysToUse.append(InputID)
    else:
        listOfKeysToUse = list(GeneDictionary[Species].keys())
    
    returning = {}

    for TargetID in ([TargetIDList] if type(TargetIDList) is str else TargetIDList):
        returning[TargetID] = {}
        for key in listOfKeysToUse:
            returning[TargetID][key] = []
            for item in InputList:
                allEntries = np.array(GeneDictionary[Species][TargetID])[np.where(np.array(GeneDictionary[Species][key])==item)[0]]
                returning[TargetID][key].append(list(np.unique(allEntries[np.where(allEntries!=None)[0]]) if InputID!=None else allEntries))
                
        #Merge all found lists into one list
        if InputID!=None:
            returningCopy = returning.copy()
            returning[TargetID] = []
            for iitem, item in enumerate(InputList):
                tempList = []
                for key in listOfKeysToUse:
                    tempList.extend(returningCopy[TargetID][key][iitem])
                returning[TargetID].append(tempList)
                    
    return returning


def KEGGAnalysisAssigner(PyIOmicaDataDirectory = None, ImportDirectly = False, BackgroundSet = [], KEGGQuery1 = "pathway", KEGGQuery2 = "hsa",
                        LengthFilter = None, LengthFilterFunction = np.greater_equal, Labels = ["IDToPath", "PathToID"]):

    """Create KEGG: Kyoto Encyclopedia of Genes and Genomes pathway associations, 
    restricted to required background set, downloading the data if necessary.

    Args: 
        PyIOmicaDataDirectory: directory where the default package data is stored
        ImportDirectly: import from URL regardles is the file already exists
        BackgroundSet: a list of IDs (e.g. gene accessions) that should be considered as the background for the calculation
        KEGGQuery1: make KEGG API calls, and sets string query1 in http://rest.kegg.jp/link/<> query1 <> / <> query2. 
    Typically this will be used as the target database to find related entries by using database cross-references
        KEGGQuery2: KEGG API calls, and sets string query2 in http://rest.kegg.jp/link/<> query1 <> / <> query2. 
    Typically this will be used as the source database to find related entries by using database cross-references
        LengthFilterFunction: option for functions that perform computations of membership in 
    pathways/ontologies/groups/sets, that specifies which function to use to filter the number of members a reported 
    category has compared to the number typically provided by LengthFilter
        LengthFilter: allows the selection of how many members each category can have, as typically 
    restricted by the LengthFilterFunction
        Labels: a string list for how keys in a created association will be named

    Returns:
        IDToPath and PathToID dictionary

    Usage:
        KEGGassignment = KEGGAnalysisAssigner()
    """
    
    global ConstantPyIOmicaDataDirectory
    
    PyIOmicaDataDirectory = ConstantPyIOmicaDataDirectory if PyIOmicaDataDirectory==None else PyIOmicaDataDirectory

    ##if the user asked us to import directly, import directly from KEGG website, otherwise, get it from a directory they specify
    fileAssociations = [os.path.join(PyIOmicaDataDirectory, item) for item in [KEGGQuery1 + "_" + KEGGQuery2 + "KEGGMemberToPathAssociation.json.gz", 
                                                                                KEGGQuery1 + "_" + KEGGQuery2 + "KEGGPathToMemberAssociation.json.gz"]]

    if not np.array(list(map(os.path.isfile, fileAssociations))).all():
        print("Did Not Find Annotation Files, Attempting to Download...")
        ImportDirectly = True

    if ImportDirectly:
        localFile = os.path.join(PyIOmicaDataDirectory, KEGGQuery1 + "_" + KEGGQuery2 + ".tsv")
        #Delete existing file
        if os.path.isfile(localFile):
            os.remove(localFile)

        urllib.request.urlretrieve("http://rest.kegg.jp/link/" + KEGGQuery1 + ("" if KEGGQuery2=="" else "/" + KEGGQuery2), localFile)

        with open(localFile, 'r') as tempFile:
            tempLines = tempFile.readlines()
            
        df = pd.DataFrame([line.strip('\n').split('\t') for line in tempLines])

        ##Remove all entries with missing
        #df = pd.DataFrame(goData.T[np.array(GOFileColumns)-1].T)
        #df = df[np.count_nonzero(df.values!='', axis=1)==2]

        dfGrouped = df.groupby(df.columns[1])
        keys, values = list(dfGrouped.indices.keys()), list(dfGrouped.indices.values())
        IDs = df.values.T[0]
        pathToID = dict(zip(keys, [np.unique(IDs[value]).tolist() for value in values]))
        idToPath = createReverseDictionary(pathToID)

        write((datetime.datetime.now().isoformat(), idToPath), fileAssociations[0], jsonFormat=True)
        write((datetime.datetime.now().isoformat(), pathToID), fileAssociations[1], jsonFormat=True)

        os.remove(localFile)

        if np.array(list(map(os.path.isfile, fileAssociations))).all():
            print("Created Annotation Files at ", fileAssociations)
        else:
            print("Did Not Find Annotation Files, Aborting Process")
            return
    else:
        #otherwise import the necessary associations from PyIOmicaDataDirectoryectory
        idToPath = read(fileAssociations[0], jsonFormat=True)[1]
        pathToID = read(fileAssociations[1], jsonFormat=True)[1]

    if BackgroundSet!=[]:
        #Using provided background list to create annotation projection to limited background space
        keys, values = np.array(list(idToPath.keys())), np.array(list(idToPath.values()))
        index = np.where([(((len(values[i])==True)*(values[i][0]!=values[i][0]))==False)*(keys[i] in BackgroundSet) for i in range(len(keys))])[0]
        idToPath = dict(zip(keys[index],values[index]))

        #Create corresponding reverse dictionary
        pathToID = createReverseDictionary(idToPath)

    if LengthFilter!=None:
        keys, values = np.array(list(pathToID.keys())), np.array(list(pathToID.values()))
        index = np.where(LengthFilterFunction(np.array([len(value) for value in values]), LengthFilter))[0]
        pathToID = dict(zip(keys[index],values[index]))

        #Create corresponding reverse dictionary
        idToPath = createReverseDictionary(pathToID)

    return {KEGGQuery2 : {Labels[0]: idToPath, Labels[1]: pathToID}}


def KEGGDictionary(PyIOmicaDataDirectory = None, ImportDirectly = False, KEGGQuery1 = "pathway", KEGGQuery2 = "hsa"):

    """Create a dictionary from KEGG: Kyoto Encyclopedia of Genes and Genomes terms - 
    typically association of pathways and members therein.
    
    Args: 
        PyIOmicaDataDirectory: directory where the default package data is stored
        ImportDirectly: import from URL regardles is the file already exists
        KEGGQuery1: make KEGG API calls, and sets string query1 in http://rest.kegg.jp/link/<> query1 <> / <> query2. 
    Typically this will be used as the target database to find related entries by using database cross-references
        KEGGQuery2: KEGG API calls, and sets string query2 in http://rest.kegg.jp/link/<> query1 <> / <> query2. 
    Typically this will be used as the source database to find related entries by using database cross-references

    Returns:
        Dictionary of definitions

    Usage:
        KEGGDict = KEGGDictionary()
    """
    
    global ConstantPyIOmicaDataDirectory
    
    PyIOmicaDataDirectory = ConstantPyIOmicaDataDirectory if PyIOmicaDataDirectory==None else PyIOmicaDataDirectory

    #if the user asked us to import directly, import directly from KEGG website, otherwise, get it from a directory they specify
    fileKEGGDict = os.path.join(PyIOmicaDataDirectory, KEGGQuery1 + "_" + KEGGQuery2 + "_KEGGDictionary.json.gz")

    if os.path.isfile(fileKEGGDict):
        associationKEGG = read(fileKEGGDict, jsonFormat=True)[1]
    else:
        print("Did Not Find Annotation Files, Attempting to Download...")
        ImportDirectly = True

    if ImportDirectly:
        queryFile = os.path.join(PyIOmicaDataDirectory, KEGGQuery1 + "_" + KEGGQuery2 + ".tsv")

        if os.path.isfile(queryFile): 
            os.remove(queryFile)

        urllib.request.urlretrieve("http://rest.kegg.jp/list/" + KEGGQuery1 + ("" if KEGGQuery2=="" else "/" + KEGGQuery2), queryFile)

        with open(queryFile, 'r') as tempFile:
            tempLines = tempFile.readlines()
            
        os.remove(queryFile)
        
        associationKEGG = dict([line.strip('\n').split('\t') for line in tempLines])

        write((datetime.datetime.now().isoformat(), associationKEGG), fileKEGGDict, jsonFormat=True)

        if os.path.isfile(fileKEGGDict):
            print("Created Annotation Files at ", fileKEGGDict)
        else:
            print("Did Not Find Annotation Files, Aborting Process")
            return

    return associationKEGG


def KEGGAnalysis(data, AnalysisType = "Genomic", GetGeneDictionaryOptions = {}, AugmentDictionary = True, InputID = ["UniProt ID", "Gene Symbol"],
                OutputID = "KEGG Gene ID", MolecularInputID = ["cpd"], MolecularOutputID = "cpd", KEGGAnalysisAssignerOptions = {}, BackgroundSet = [], 
                KEGGOrganism = "hsa", KEGGMolecular = "cpd", KEGGDatabase = "pathway", PathwayLengthFilter = 2, ReportFilter = 1, 
                ReportFilterFunction = np.greater_equal, pValueCutoff = 0.05, TestFunction = lambda n, N, M, x: 1. - scipy.stats.hypergeom.cdf(x-1, M, n, N), 
                HypothesisFunction = lambda data, SignificanceLevel: BenjaminiHochbergFDR(data, SignificanceLevel=SignificanceLevel)["Results"],
                FilterSignificant = True, KEGGDictionaryVariable = None, KEGGDictionaryOptions = {}, MultipleListCorrection = None, MultipleList = False, 
                GeneDictionary = None, Species = "human", MolecularSpecies = "compound", NonUCSC = False, PyIOmicaDataDirectory = None):

    """Calculate input data over-representation analysis for KEGG: Kyoto Encyclopedia of Genes and Genomes pathways.
    Input can be a list, a dictionary of lists or a clustering object.

    Args:
        data: data to analyze
        AnalysisType: analysis methods that may be used, "Genomic", "Molecular" or "All"
        GetGeneDictionaryOptions: a list of options that will be passed to this internal GetGeneDictionary function
        AugmentDictionary: a choice whether or not to augment the current ConstantGeneDictionary global variable or create a new one
        InputID: the kind of identifiers/accessions used as input
        OutputID: a string value that specifies what kind of IDs/accessions to convert the input IDs/accession 
    numbers in the function's analysis
        MolecularInputID: a string list to indicate the kind of ID to use for the input molecule entries
        KEGGAnalysisAssignerOptions: a list of options that will be passed to this internal KEGGAnalysisAssigner function
        BackgroundSet: a list of IDs (e.g. gene accessions) that should be considered as the background for the calculation
        KEGGOrganism: indicates which organism (org) to use for \"Genomic\" type of analysis (default is human analysis: org=\"hsa\")
        KEGGMolecular: which database to use for molecular analysis (default is the compound database: cpd)
        KEGGDatabase: KEGG database to use as the target database
        PathwayLengthFilter: pathways to consider in the computation, by excluding pathways that have fewer items 
    compared to the PathwayLengthFilter value
        ReportFilter: provides a cutoff for membership in ontologies/pathways/groups in selecting which terms/categories 
    to return. It is typically used in conjunction with ReportFilterFunction
        ReportFilterFunction: operator form will be used to compare against ReportFilter option value in selecting 
    which terms/categories to return
        pValueCutoff: a cutoff p-value for (adjusted) p-values to assess statistical significance 
        TestFunction: a function used to calculate p-values
        HypothesisFunction: allows the choice of function for implementing multiple hypothesis testing considerations
        FilterSignificant: can be set to True to filter data based on whether the analysis result is statistically significant, 
    or if set to False to return all membership computations
        KEGGDictionaryVariable: KEGG dictionary, and provides a KEGG annotation variable. If set to None, KEGGDictionary 
    will be used internally to automatically generate the default KEGG annotation
        KEGGDictionaryOptions: a list of options to be passed to the internal KEGGDictionary function that provides the KEGG annotations
        MultipleListCorrection: specifies whether or not to correct for multi-omics analysis. 
    The choices are None, Automatic, or a custom number
        MultipleList: whether the input accessions list constituted a multi-omics list input that is annotated so
        GeneDictionary: existing variable to use as a gene dictionary in annotations. If set to None the default ConstantGeneDictionary will be used
        Species: the species considered in the calculation, by default corresponding to human
        MolecularSpecies: the kind of molecular input
        NonUCSC: if UCSC browser was used in determining an internal GeneDictionary used in ID translations,
    where the KEGG identifiers for genes are number strings (e.g. 4790).The NonUCSC option can be set to True 
    if standard KEGG accessions are used in a user provided GeneDictionary variable, 
    in the form OptionValue[KEGGOrganism] <>:<>numberString, e.g. hsa:4790
        PyIOmicaDataDirectory: directory where the default package data is stored

    Returns:
        Enrichment dictionary

    Usage:
        keggExample1 = KEGGAnalysis(["TAB1", "TNFSF13B", "MALT1", "TIRAP", "CHUK", "TNFRSF13C", "PARP1", "CSNK2A1", "CSNK2A2", "CSNK2B", "LTBR", "LYN", "MYD88", 
                                            "GADD45B", "ATM", "NFKB1", "NFKB2", "NFKBIA", "IRAK4", "PIAS4", "PLAU", "POLR3B", "NME1", "CTPS1", "POLR3A"])
    """

    argsLocal = locals().copy()

    global ConstantPyIOmicaDataDirectory

    obtainConstantGeneDictionary(None, {}, True)
    
    PyIOmicaDataDirectory = ConstantPyIOmicaDataDirectory if PyIOmicaDataDirectory==None else PyIOmicaDataDirectory

    #Gene Identifier based analysis
    if AnalysisType=="Genomic":
        #Obtain OBO dictionary. If externally defined use user definition for OBODict Var
        keggDict = KEGGDictionary(**KEGGDictionaryOptions) if KEGGDictionaryVariable==None else KEGGDictionaryVariable

        #Obtain gene dictionary - if it exists can either augment with new information or Species or create new, if not exist then create variable
        obtainConstantGeneDictionary(GeneDictionary, GetGeneDictionaryOptions, AugmentDictionary)

        #get the right KEGG terms for the BackgroundSet requested and correct Species
        Assignment = KEGGAnalysisAssigner(BackgroundSet=BackgroundSet, KEGGQuery1=KEGGDatabase, KEGGQuery2=KEGGOrganism, LengthFilter=PathwayLengthFilter) if KEGGAnalysisAssignerOptions=={} else KEGGAnalysisAssigner(**KEGGAnalysisAssignerOptions)

    #Molecular based analysis
    elif AnalysisType=="Molecular":
        InputID = MolecularInputID
        OutputID = MolecularOutputID
        Species = MolecularSpecies
        NonUCSC = True
        KEGGOrganism = KEGGMolecular
        MultipleListCorrection = None

        keggDict = KEGGDictionary(**({"KEGGQuery1": "pathway", "KEGGQuery2": ""} if KEGGDictionaryOptions=={} else KEGGDictionaryOptions)) if KEGGDictionaryVariable==None else KEGGDictionaryVariable

        #Obtain gene dictionary - if it exists can either augment with new information or Species or create new, if not exist then create variable
        fileMolDict = os.path.join(PyIOmicaDataDirectory, "PyIOmicaMolecularDictionary.json.gz")

        if os.path.isfile(fileMolDict):
            GeneDictionary = read(fileMolDict, jsonFormat=True)[1]
        else:
            fileCSV = os.path.join(PackageDirectory, "data", "MathIOmicaMolecularDictionary.csv")

            print('Attempting to read:', fileCSV)

            if os.path.isfile(fileCSV):
                with open(fileCSV, 'r') as tempFile:
                    tempLines = tempFile.readlines()
            
                tempData = np.array([line.strip('\n').replace('"', '').split(',') for line in tempLines]).T
                tempData = {'compound': {'pumchem': tempData[0].tolist(), 'cpd': tempData[1].tolist()}}
                write((datetime.datetime.now().isoformat(), tempData), fileMolDict, jsonFormat=True)
            else:
                print("Could not find annotation file at " + fileMolDict + " Please either obtain an annotation file from mathiomica.org or provide a GeneDictionary option variable.")
                return

            GeneDictionary = read(fileMolDict, jsonFormat=True)[1]

        obtainConstantGeneDictionary(GeneDictionary, {}, AugmentDictionary)

        #Get the right KEGG terms for the BackgroundSet requested and correct Species
        #If no specific options for function use BackgroundSet, Species request, length request
        Assignment = KEGGAnalysisAssigner(BackgroundSet=BackgroundSet, KEGGQuery1=KEGGDatabase, KEGGQuery2=KEGGOrganism , LengthFilter=PathwayLengthFilter) if KEGGAnalysisAssignerOptions=={} else KEGGAnalysisAssigner(**KEGGAnalysisAssignerOptions)
    
    #Gene Identifier and Molecular based analysis done concurrently
    elif AnalysisType=='All':
        argsMolecular = argsLocal.copy()
        argsMolecular['AnalysisType'] = 'Molecular'
        argsGenomic = argsLocal.copy()
        argsGenomic['AnalysisType'] = 'Genomic'

        return {"Molecular": KEGGAnalysis(**argsMolecular), "Genomic": KEGGAnalysis(**argsGenomic)}
    
    #Abort
    else:
        print("AnalysisType %s is not a valid choice."%AnalysisType)

        return

    #The data may be a subgroup from a clustering object, i.e. a pd.DataFrame
    if type(data) is pd.DataFrame:
        id = list(data.index.get_level_values('id'))
        source = list(data.index.get_level_values('source'))
        data = [[id[i], source[i]] for i in range(len(data))]

    #If the input is simply a list
    listToggle = True if type(data) is list else False
    data = {'dummy': data} if listToggle else data

    returning = {}

    #Check if a clustering object
    if "linkage" in data.keys():
        if MultipleListCorrection==None:
            multiCorr = 1
        elif MultipleListCorrection=='Automatic':
            multiCorr = 1
            for keyGroup in sorted([item for item in list(data.keys()) if not item=='linkage']):
                for keySubGroup in sorted([item for item in list(data[keyGroup].keys()) if not item=='linkage']):
                    multiCorr = max(max(np.unique(data[keyGroup][keySubGroup]['data'].index.get_level_values('id'), return_counts=True)[1]), multiCorr)
        else:
            multiCorr = MultipleListCorrection

        #Loop through the clustering object, calculate GO for each SubGroup
        for keyGroup in sorted([item for item in list(data.keys()) if not item=='linkage']):
            returning[keyGroup] = {}
            for keySubGroup in sorted([item for item in list(data[keyGroup].keys()) if not item=='linkage']):
                SubGroupMultiIndex = data[keyGroup][keySubGroup]['data'].index
                SubGroupGenes = list(SubGroupMultiIndex.get_level_values('id'))
                SubGroupMeta = list(SubGroupMultiIndex.get_level_values('source'))
                SubGroupList = [[SubGroupGenes[i], SubGroupMeta[i]] for i in range(len(SubGroupMultiIndex))]

                returning[keyGroup][keySubGroup] = internalAnalysisFunction({keySubGroup:SubGroupList},
                                                                             multiCorr, MultipleList,  OutputID, InputID, Species, len(Assignment[KEGGOrganism]["IDToPath"]),
                                                                             pValueCutoff, ReportFilterFunction, ReportFilter, TestFunction, HypothesisFunction, FilterSignificant,
                                                                             AssignmentForwardDictionary=Assignment[KEGGOrganism]['IDToPath'],
                                                                             AssignmentReverseDictionary=Assignment[KEGGOrganism]['PathToID'],
                                                                             prefix='hsa:' if AnalysisType=='Genomic' else '', infoDict=keggDict)[keySubGroup]

    #The data is a dictionary of type {'Name1': [data1], 'Name2': [data2], ...}
    else:
        for key in list(data.keys()):
            if MultipleListCorrection==None:
                multiCorr = 1
            elif MultipleList and MultipleListCorrection=='Automatic':
                multiCorr = max(np.unique([item[0] for item in data[key]], return_counts=True)[1])
            else:
                multiCorr = MultipleListCorrection

            returning.update(internalAnalysisFunction({key:data[key]}, multiCorr, MultipleList,  OutputID, InputID, Species, len(Assignment[KEGGOrganism]["IDToPath"]),
                                                pValueCutoff, ReportFilterFunction, ReportFilter, TestFunction, HypothesisFunction, FilterSignificant,
                                                AssignmentForwardDictionary=Assignment[KEGGOrganism]['IDToPath'],
                                                AssignmentReverseDictionary=Assignment[KEGGOrganism]['PathToID'],
                                                prefix='hsa:' if AnalysisType=='Genomic' else '', infoDict=keggDict))

        #If a single list was provided
        returning = returning['dummy'] if listToggle else returning

    return returning


def MassMatcher(data, accuracy, MassDictionaryVariable = None, MolecularSpecies = "cpd"):

    """Assign putative mass identification to input data based on monoisotopic mass 
    (using PyIOmica's mass dictionary). The accuracy in parts per million. 
    
    Args: 
        data: input data
        accuracy: accuracy
        MassDictionaryVariable: mass dictionary variable. If set to None, inbuilt 
    mass dictionary (MassDictionary) will be loaded and used
        MolecularSpecies: the kind of molecular input

    Returns:
        List of IDs 

    Usage:
       result = MassMatcher(18.010565, 2)
    """
    
    ppm = accuracy*(10**-6)

    MassDict = MassDictionary() if MassDictionaryVariable==None else MassDictionaryVariable
    keys, values = np.array(list(MassDict[MolecularSpecies].keys())), np.array(list(MassDict[MolecularSpecies].values()))

    return keys[np.where((values > data*(1 - ppm)) * (values < data*(1 + ppm)))[0]]


def MassDictionary(PyIOmicaDataDirectory=None):

    """Load PyIOmica's current mass dictionary.
    
    Args:
        PyIOmicaDataDirectory: directory where the default package data is stored

    Returns:
        Mass dictionary

    Usage:
        MassDict = MassDictionary()
    """

    global ConstantPyIOmicaDataDirectory

    PyIOmicaDataDirectory = ConstantPyIOmicaDataDirectory if PyIOmicaDataDirectory==None else PyIOmicaDataDirectory

    fileMassDict = os.path.join(PyIOmicaDataDirectory, "PyIOmicaMassDictionary.json.gz")

    if os.path.isfile(fileMassDict):
        MassDict = read(fileMassDict, jsonFormat=True)[1]
    else:
        fileCSV = os.path.join(PackageDirectory, "data", "MathIOmicaMassDictionary" +  ".csv")

        if False:
            with open("PyIOmicaMassDictionary", 'r') as tempFile:
                mathDictData = tempFile.readlines()

            mathDict = ''.join([line.strip('\n') for line in mathDictData]).replace('"','').replace(' ','').replace('->',' ').split(',')
            mathDict = np.array([line.split(' ') for line in mathDict])
            np.savetxt(fileCSV, mathDictData, delimiter=',', fmt='%s')

        if os.path.isfile(fileCSV):
            print('Reading:', fileCSV)

            fileMassDictData = np.loadtxt(fileCSV, delimiter=',', dtype=str)
            MassDict = {fileMassDictData[0][0].split(':')[0]: dict(zip(fileMassDictData.T[0],fileMassDictData.T[1].astype(float)))}
            write((datetime.datetime.now().isoformat(), MassDict), fileMassDict, jsonFormat=True)

            print("Created mass dictionary at ", fileMassDict)
        else:
            print("Could not find mass dictionary at ", fileMassDict, 
                    "Please either obtain a mass dictionary file from mathiomica.org or provide a custom file at the above location.")

            return None

    return MassDict


def ExportEnrichmentReport(data, AppendString="", OutputDirectory=None):

    """Export results from enrichment analysis to Excel spreadsheets.
    
    Args:
        data: enrichment results
        AppendString: custom report name, if empty then time stamp will be used
        OutputDirectory: path of directories where the report will be saved

    Returns:
        None

    Usage:
        ExportEnrichmentReport(goExample1, AppendString='goExample1', OutputDirectory=None)
    """

    def FlattenDataForExport(data):

        returning = {}

        if (type(data) is dict):
            if len(data)==0:
                print('The result is empty.')
                returning['List'] = data
                return 
            idata = data[list(data.keys())[0]]
            if not type(idata) is dict:
                returning['List'] = data
            elif type(idata) is dict:
                idata = idata[list(idata.keys())[0]]
                if not type(idata) is dict:
                    returning = data
                elif type(idata) is dict:
                    idata = idata[list(idata.keys())[0]]
                    if not type(idata) is dict:
                        #Loop through the clustering object
                        for keyClass in list(data.keys()):
                            for keySubClass in list(data[keyClass].keys()):
                                returning[str(keyClass)+' '+str(keySubClass)] = data[keyClass][keySubClass]
                    elif type(idata) is dict:
                        for keyAnalysisType in list(data.keys()):
                            #Loop through the clustering object
                            for keyClass in list(data[keyAnalysisType].keys()):
                                for keySubClass in list(data[keyAnalysisType][keyClass].keys()):
                                    returning[str(keyAnalysisType)+' '+str(keyClass)+' '+str(keySubClass)] = data[keyAnalysisType][keyClass][keySubClass]
        else:
            print('Results type is not supported...')

        return returning

    def ExportToFile(fileName, data):

        writer = pd.ExcelWriter(fileName)

        for key in list(data.keys()):
            keys, values = list(data[key].keys()), list(data[key].values())

            listNum = [[item for sublist in list(value)[:2] for item in sublist] for value in values]
            listNon = [list(value)[2:] for value in values]

            dataDF = [listNum[i] + listNon[i] for i in range(len(keys))]
            columns = ['p-Value', 'BH-corrected p-Value', 'Significant', 'Counts in list', 'Counts in family', 'Total members', 'Counts in members', 'Description', 'List of gene hits']

            df = pd.DataFrame(data=dataDF, index=keys, columns=columns)

            df['Significant'] = df['Significant'].map(bool)

            cleanup = lambda value: value.replace("']], ['", ' | ').replace("[", '').replace("]", '').replace("'", '').replace(", Unknown", '')
            df['List of gene hits'] = df['List of gene hits'].map(str).apply(cleanup)
            df['Description'] = df['Description'].map(str).apply(cleanup)

            df.sort_values(by='BH-corrected p-Value', inplace=True)

            df.to_excel(writer, str(key))

            writer.sheets[str(key)].set_column('A:A', df.index.astype(str).map(len).max()+2)
             
            format = writer.book.add_format({'text_wrap': True,
                                             'valign': 'top'})

            for idx, column in enumerate(df.columns):
                max_len = max((df[column].astype(str).map(len).max(),  # len of largest item
                            len(str(df[column].name)))) + 1            # len of column name/header adding a little extra space

                width = 50 if column=='Description' else min(180, max_len)

                writer.sheets[str(key)].set_column(idx+1, idx+1, width, format)  # set column width

        writer.save()

        print('Saved:', fileName)

        return None
    
    saveDir = os.path.join(os.getcwd(), "Enrichment reports") if OutputDirectory==None else OutputDirectory

    createDirectories(saveDir)

    if AppendString=="":
        AppendString=(datetime.datetime.now().isoformat().replace(' ', '_').replace(':', '_').split('.')[0])

    ExportToFile(saveDir + AppendString + '.xlsx', FlattenDataForExport(data))

    return None

###################################################################################################



### Core functions ################################################################################
def chop(expr, tolerance=1e-10):

    """Equivalent of Mathematica.Chop Function.

    Args:
        expr: a number or a pyhton sequence of numbers
        tolerance: default is the same as in Mathematica

    Returns:
        Chopped data

    Usage
        data = chop(data)
    """
        
    if isinstance(expr, (list, tuple, np.ndarray)):

        expr_copy = np.copy(expr)
        expr_copy[np.abs(expr) < tolerance] = 0

    else:
        expr_copy = 0 if expr < tolerance else expr

    return expr_copy


def modifiedZScore(subset):

    """Calculate modified z-score of a 1D array based on "Median absolute deviation".
    Use on 1-D arrays only.

    Args:
        subset: data to transform

    Returns:
        Transformed subset

    Usage:
        data = modifiedZScore(data)
    """

    def medianAbsoluteDeviation(expr, axis=None):

        """1D, 2D Median absolute deviation of a sequence of numbers or pd.Series.

        Args:
            expr: data for analysis
            axis: default is None: multidimentional arrays are flattened, 0: use if data in columns, 1: use if data in rows

        Returns:
            Median absolute deviation (M.A.D.)

        Usage:
            MedianAD = medianAbsoluteDeviation(data, axis=None)
        """

        data = None

        if isinstance(expr, np.ndarray):

            data = expr

        elif isinstance(expr, (pd.Series, pd.DataFrame)):

            data = expr.values

        try:

            if len(data) > 1:

                if axis == None or axis == 0:

                    return np.median(np.abs(data - np.median(data,axis)),axis)

                elif axis == 1:

                    if len(data.shape) < 2:

                        print('Warning: axis = %s option is invalid for 1-D array...' % (axis))

                    else:

                        return np.median((np.abs(data.transpose() - np.median(data,axis)).transpose()),axis)

        except :

            print('Unsupported data type: ', type(expr))
        
        return None

    values = subset[~np.isnan(subset.values)].values

    MedianAD = medianAbsoluteDeviation(values, axis=None)

    if MedianAD == 0.:
        MeanAD = np.sum(np.abs(values - np.mean(values))) / len(values)
        print('MeanAD:', MeanAD, '\tMedian:', np.median(values))
        coefficient = 0.7978846 / MeanAD
    else:
        print('MedianAD:', MedianAD, '\tMedian:', np.median(values))
        coefficient = 0.6744897 / MedianAD
        
    subset.iloc[~np.isnan(subset.values)] = coefficient * (values - np.median(values))

    return subset


def boxCoxTransform(subset, lmbda=None, giveLmbda=False):

    """Power transform from scipy.stats

    Args:
        subset: pandas Series.
        lmbda: Lambda parameter, if not specified optimal value will be determined
        giveLmbda: also return Lambda value

    Returns:
        Transformed subset and Lambda parameter

    Usage:
        myData = boxCoxTransform(myData)
    """

    where_negative = np.where(subset < 0)
    if len(where_negative>0):
        errMsg = 'Warning: negative values are present in the data. Review the sequence of the data processing steps.'
        print(errMsg)

    where_positive = np.where(subset > 0)

    if lmbda == None:
        transformed_data = scipy.stats.boxcox(subset.values[where_positive])
    else:
        transformed_data = (scipy.stats.boxcox(subset.values[where_positive], lmbda=lmbda),lmbda)

    subset.iloc[where_positive] = transformed_data[0]

    lmbda = transformed_data[1]

    if giveLmbda:

        return subset, lmbda

    print('Fitted lambda:', lmbda)

    return subset


def ampSquaredNormed(func, freq, times, data):

    """Lomb-Scargle core function
    Calculate the different frequency components of our spectrum: project the cosine/sine component and normalize it:

    Args:
        func: Sin or Cos
        freq: frequencies (1D array of floats)
        times: input times (starting point adjusted w.r.t.dataset times), Zero-padded
        data: input Data with the mean subtracted from it, before zero-padding.

    Returns:
        Squared amplitude normalized.

    Usage:
        coef = ampSquaredNormed(np.cos, freguency, inputTimesNormed, inputDataCentered)
        Intended for internal use only.
    """

    omega_freq = 2. * (np.pi) * freq
    theta_freq = 0.5 * np.arctan2(np.sum(np.sin(4. * (np.pi) * freq * times)), np.sum(np.cos(4. * (np.pi) * freq * times) + 10 ** -20))
    
    ampSum = np.sum(data * func(omega_freq * times - theta_freq)) ** 2
    ampNorm = np.sum(func(omega_freq * times - theta_freq) ** 2)

    return chop(ampSum) / ampNorm


def autocorrelation(inputTimes, inputData, inputSetTimes, UpperFrequencyFactor=1):
    
    """Autocorrelation function

    Args:
        inputTimes: times corresponding to provided data points (1D array of floats)
        inputData: data points (1D array of floats)
        inputSetTimes: a complete set of all possible N times during which data could have been collected.

    Returns:
        Array of time lags with corresponding autocorrelations

    Usage:
        result = autocorrelation(inputTimes, inputData, inputSetTimes)
    """

    def InverseAutocovariance(inputTimes, inputData, inputSetTimes, UpperFrequencyFactor=1):

        #adjust inputTimes starting point w.r.t.dataset times, AND ZERO-PAD THEM
        inputTimesNormed = np.concatenate((inputTimes, inputSetTimes + inputSetTimes[-1])) - inputSetTimes[0]

        #calculate the number of timepoints in the overall set-since we cut
        #freqStep to half f0, we should compensate n by multiplication by two
        n = 2 * len(inputSetTimes)

        #calculate the time window of observation
        window = np.max(inputSetTimes) - np.min(inputSetTimes)

        #invert this window to get the fundamental frequency,with the n/n-1
        #correction to account for the difference between the time window and
        #time period of the signal (WE REMOVE THIS FOR NOW)
        f0 = 1.0 / window

        #subtract the mean from the inputData,BEFORE YOU ZERO-PAD IT!
        inputDataCentered = np.concatenate((inputData - np.mean(inputData), np.zeros(len(inputSetTimes))))

        #calculate a variance for the centered data
        varianceInputPoints = np.var(inputDataCentered, ddof=1)

        #define the frequency step as HALF the fundamental frequency in order
        #to zero-pad and get an evenly spaced mesh
        freqStep = 0.5 * f0

        #get the list of frequencies
        freq = np.linspace(0.5 * f0, n * UpperFrequencyFactor * 0.5 * f0, n * UpperFrequencyFactor)

        #calculate the inverse autocorrelation
        inverseAuto = 1.0 / (2.0 * varianceInputPoints) * np.array(tuple(map(lambda f: ampSquaredNormed(np.cos, f, inputTimesNormed, inputDataCentered) + ampSquaredNormed(np.sin, f, inputTimesNormed, inputDataCentered), list(freq))))
    
        #return: 1) the list of frequencies, 2) the correspoinding list of inverse autocovariances
        return np.transpose(np.vstack((freq, inverseAuto)))

    inputInverseAuto = InverseAutocovariance(inputTimes[np.isnan(inputData) == False], inputData[np.isnan(inputData) == False], inputSetTimes, UpperFrequencyFactor = UpperFrequencyFactor)

    #create the amplitude spectrum from the input data:
    #add a zero at the first element to make the DFT work, sample only half the
    #points because we have oversampled by 2 in the inverseAutocovariance
    inverseAmplitudes = np.concatenate(([0], inputInverseAuto[:np.int(inputInverseAuto.shape[0] / 2), 1]))

    #do the DCT-III transform:
    autoCorrs = scipy.fftpack.dct(inverseAmplitudes, type=3, norm='ortho')

    #divide everything by a normalization factor so that the autocorrelation at lag 0 = 1
    #make sure we are only returning autocorrelations for the lags we can rely on, i.e.  for up to N/2 time points
    values = autoCorrs[:np.int(np.floor(0.5 * len(autoCorrs)))] / autoCorrs[0]

    return np.vstack((inputSetTimes[:len(values)], values))


def pAutocorrelation(args):

    """Wrapper of Autocorrelation function for use with Multiprocessing.

    Args:
        args: a tuple of arguments in the form (inputTimes, inputData, inputSetTimes)

    Returns:
        Array of time lags with corresponding autocorrelations

    Usage:
        result = pAutocorrelation((inputTimes, inputData, inputSetTimes))
    """

    inputTimes, inputData, inputSetTimes = args
    
    return autocorrelation(inputTimes, inputData, inputSetTimes)


def getSpikes(inputData, func, cutoffs):

    """Get sorted index of signals with statistically significant spikes,
    i.e. those that pass the provided cutoff.

    Args:
        inputData: data points (2D array of floats) where rows are normalized signals
        func: np.max or np.min
        cutoffs: a dictionary of cutoff values

    Returns:
        Index of data with statistically significant spikes

    Usage:
        index = getSpikes(inputData, np.max, cutoffs)
    """

    data = inputData.copy()
    counts_non_missing = np.sum(~np.isnan(data), axis=1)
    data[np.isnan(data)] = 0.

    spikesIndex = []

    for i in list(range(data.shape[1]+1)):
        ipos = np.where(counts_non_missing==i)[0]
        if len(data[ipos])>0:
            points = func(data[ipos], axis=1)
            spikesIndex.extend(ipos[np.where((points>cutoffs[i][0]) | (points<cutoffs[i][1]))[0]])

    return sorted(spikesIndex)


def getSpikesCutoffs(df_data, p_cutoff, NumberOfRandomSamples=10**3):

    """Calculate spikes cuttoffs from a bootstrap of provided data,
    gived the significance cutoff p_cutoff.

    Args:
        df_data: pandas DataFrame where rows are normalized signals
        p_cutoff: p-Value cutoff, e.g. 0.01
        NumberOfRandomSamples: size of the bootstrap distribution

    Returns:
        Dictionary of spike cutoffs.

    Usage:
        cutoffs = getSpikesCutoffs(df_data, 0.01)
    """

    data = np.vstack([np.random.choice(df_data.values[:,i], size=NumberOfRandomSamples, replace=True) for i in range(len(df_data.columns.values))]).T

    df_data_random = pd.DataFrame(data=data, index=range(NumberOfRandomSamples), columns=df_data.columns)
    df_data_random = filterOutFractionZeroSignalsDataframe(df_data_random, 0.75)
    df_data_random = normalizeSignalsToUnityDataframe(df_data_random)
    df_data_random = removeConstantSignalsDataframe(df_data_random, 0.)

    data = df_data_random.values
    counts_non_missing = np.sum(~np.isnan(data), axis=1)
    data[np.isnan(data)] = 0.

    cutoffs = {}

    for i in list(range(data.shape[1]+1)):
        idata = data[counts_non_missing==i]
        if len(idata)>0:
            cutoffs.update({i : (np.quantile(np.max(idata, axis=1), 1.-p_cutoff, interpolation='lower'),
            np.quantile(np.min(idata, axis=1), p_cutoff, interpolation='lower'))} )

    return cutoffs


def LombScargle(inputTimes, inputData, inputSetTimes, FrequenciesOnly=False,NormalizeIntensities=False,OversamplingRate=1,UpperFrequencyFactor=1):

    """Calculate Lomb-Scargle periodogram.

    Args:
        inputTimes: times corresponding to provided data points (1D array of floats)
        inputData: data points (1D array of floats)
        inputSetTimes: a complete set of all possible N times during which data could have been collected
        FrequenciesOnly: return frequencies only
        NormalizeIntensities: normalize intensities to unity
        OversamplingRate: oversampling rate
        UpperFrequencyFactor: upper frequency factor

    Returns:
        Periodogram with a list of frequencies.

    Usage:
        pgram = LombScargle(inputTimes, inputData, inputSetTimes)
    """

    #adjust inputTimes starting point w.r.t.dataset times
    inputTimesNormed = inputTimes - inputSetTimes[0]

    #calculate the number of timepoints in the overall set
    n = len(inputSetTimes)

    #calculate the time window of observation
    window = np.max(inputSetTimes) - np.min(inputSetTimes)

    #invert this window to get the fundamental frequency, with the n/n-1
    #correction to account for the difference between the time window and time
    #period of the signal (WE ARE,FOR NOW,NOT INCLUDING THIS!)
    f0 = n / ((n - 1) * window)

    #subtract the mean from the inputData
    inputDataCentered = inputData - np.mean(inputData)

    #calculate a variance for the centered data
    varianceInputPoints = np.var(inputDataCentered, ddof=1)

    #define a frequency step
    freqStep = 1 / (OversamplingRate * (np.floor(n / 2) - 1)) * (n / 2 * UpperFrequencyFactor - 1) * f0

    #get the list of frequencies, adjusting both the lower frequency (to equal
    #f0 0-effectively a lowpass filter) and the upper cutoff Nyquist by the upper factor specified
    freq = np.linspace(f0, n / 2 * UpperFrequencyFactor * f0, f0 * (n / 2 * UpperFrequencyFactor) / freqStep)

    if FrequenciesOnly:
        return freq

    #get the periodogram
    periodogram = 1.0 / (2.0 * varianceInputPoints) * np.array(tuple(map(lambda f: chop(ampSquaredNormed(np.cos, f, inputTimesNormed, inputDataCentered)) + chop(ampSquaredNormed(np.sin, f, inputTimesNormed, inputDataCentered)), list(freq))))
    
    #the function finally returns:1) the list of frequencies, 2) the
    #corresponding list of Lomb-Scargle spectra
    if NormalizeIntensities:
        periodogram = periodogram / np.sqrt(np.dot(periodogram,periodogram))

    returning = np.vstack((freq, periodogram))

    return returning


def pLombScargle(args):

    """Wrapper of LombScargle function for use with Multiprocessing.

    Args:
        args: a tuple of arguments in the form (inputTimes, inputData, inputSetTimes)

    Returns:
        Array of frequencies with corresponding intensities

    Usage:
        result = pLombScargle((inputTimes, inputData, inputSetTimes))
    """

    inputTimes, inputData, inputSetTimes = args
    
    return LombScargle(inputTimes, inputData, inputSetTimes)


def getAutocorrelationsOfData(params):

    """Calculate autocorrelation using Lomb-Scargle Autocorrelation.
    NOTE: there should be already no missing or non-numeric points in the input Series or Dataframe

    Args:
        params: a tuple of parameters in the form (df_data, setAllInputTimes), where
        df_data is a pandas Series or Dataframe, 
        setAllInputTimes is a complete set of all possible N times during which data could have been collected.

    Returns:
        Array of autocorrelations of data.

    Usage:
        result  = autocorrelation(df_data, setAllInputTimes)
    """

    df, setAllInputTimes = params

    if isinstance(df, pd.Series):

        return autocorrelation(df.index.values, df.values, setAllInputTimes)

    elif isinstance(df, pd.DataFrame):
        listOfAutocorrelations = []

        for timeSeriesIndex in df.index:
            listOfAutocorrelations.append(autocorrelation(df.loc[timeSeriesIndex].index.values, df.loc[timeSeriesIndex].values, setAllInputTimes))

        return np.vstack(listOfAutocorrelations)

    print('Warning: Input data type unrecognized: use <pandas.Series> or <pandas.DataFrame>')

    return None


def getRandomAutocorrelations(df_data, NumberOfRandomSamples=10**5, NumberOfCPUs=4):

    """Generate autocorrelation null-distribution from permutated data using Lomb-Scargle Autocorrelation.
    NOTE: there should be already no missing or non-numeric points in the input Series or Dataframe

    Args:
        df_data: pandas Series or Dataframe
        NumberOfRandomSamples: size of the distribution to generate
        NumberOfCPUs: number of processes to run simultaneously

    Returns:
        DataFrame containing autocorrelations of null-distribution of data.

    Usage:
        result = getRandomAutocorrelations(df_data)
    """

    data = np.vstack([np.random.choice(df_data.values[:,i], size=NumberOfRandomSamples, replace=True) for i in range(len(df_data.columns.values))]).T

    df_data_random = pd.DataFrame(data=data, index=range(NumberOfRandomSamples), columns=df_data.columns)
    df_data_random = filterOutFractionZeroSignalsDataframe(df_data_random, 0.75)
    df_data_random = normalizeSignalsToUnityDataframe(df_data_random)
    df_data_random = removeConstantSignalsDataframe(df_data_random, 0.)

    print('\nCalculating autocorrelations of %s random samples (sampled with replacement)...'%(df_data_random.shape[0]))

    results = runCPUs(NumberOfCPUs, pAutocorrelation, [(df_data_random.iloc[i].index.values, df_data_random.iloc[i].values, df_data.columns.values) for i in range(df_data_random.shape[0])])
    
    return pd.DataFrame(data=results[1::2], columns=results[0])


def getRandomPeriodograms(df_data, NumberOfRandomSamples=10**5, NumberOfCPUs=4):

    """Generate periodograms null-distribution from permutated data using Lomb-Scargle function.

    Args:
        df_data: pandas Series or Dataframe
        NumberOfRandomSamples: size of the distribution to generate
        NumberOfCPUs: number of processes to run simultaneously

    Returns:
        New Pandas DataFrame containing periodograms

    Usage:
        result = getRandomPeriodograms(df_data)
    """

    data = np.vstack([np.random.choice(df_data.values[:,i], size=NumberOfRandomSamples, replace=True) for i in range(len(df_data.columns.values))]).T

    df_data_random = pd.DataFrame(data=data, index=range(NumberOfRandomSamples), columns=df_data.columns)
    df_data_random = filterOutFractionZeroSignalsDataframe(df_data_random, 0.75)
    df_data_random = normalizeSignalsToUnityDataframe(df_data_random)
    df_data_random = removeConstantSignalsDataframe(df_data_random, 0.)

    print('\nCalculating periodograms of %s random samples (sampled with replacement)...'%(df_data_random.shape[0]))

    return getLobmScarglePeriodogramOfDataframe(df_data_random, NumberOfCPUs=NumberOfCPUs)


def BenjaminiHochbergFDR(pValues, SignificanceLevel=0.05):

    """HypothesisTesting BenjaminiHochbergFDR correction

    Args:
        pValues: p-values (1D array of floats)
        SignificanceLevel: default = 0.05.

    Returns:
        Corrected p-Values, p- and q-Value cuttoffs

    Usage:
        result = BenjaminiHochbergFDR(pValues)
    """

    pValues = np.round(pValues,6)
      
    #count number of hypotheses tested
    nTests = len(pValues)

    #sort the pValues in order
    sortedpVals = np.sort(pValues)

    #generate a sorting ID by ordering
    sortingIDs = np.argsort(np.argsort(pValues))

    #adjust p values to weighted p values
    weightedpVals = sortedpVals * nTests / (1 + np.arange(nTests))

    #flatten the weighted p-values to smooth out any local maxima and get adjusted p-vals
    adjustedpVals = np.array([np.min(weightedpVals[i:]) for i in range(nTests)])

    #finally,generate the qVals by reordering
    qVals = adjustedpVals[sortingIDs]

    ##create an association from which to identify correspondence between
    ##p-values and q-values#Print[{qVals,pValues}];
    pValqValAssociation = dict(zip(qVals, pValues))

    #get the cutoff adjusted q-value
    tempValues = np.flip(adjustedpVals)[np.flip(adjustedpVals) <= SignificanceLevel]
    cutoffqValue = tempValues[0] if len(tempValues) > 0 else np.nan

    #identify corresponding cutoff p-value
    if np.isnan(cutoffqValue):
        cutoffqValue = 0.
        pValCutoff = 0.
    else:
        pValCutoff = pValqValAssociation[cutoffqValue]

    #get q-vals and q-val cutoff, test the qVals for being above or below
    #significance level -- return "true" if enriched and "false" if not
    returning = {"Results": np.vstack((pValues, qVals, qVals <= cutoffqValue)),
                    "p-Value Cutoff": pValCutoff,
                    "q-Value Cutoff": cutoffqValue}

    return returning


def metricCommonEuclidean(u,v):

    """Metric to calculate 'euclidean' distance between vectors u and v 
    using only common non-missing points (not NaNs).

    Args:
        u: Numpy 1-D array
        v: Numpy 1-D array

    Returns:
        Measure of the distance between u and v

    Usage:
        dist = metricCommonEuclidean(u,v)
    """

    where_common = (~np.isnan(u)) * (~np.isnan(v))

    return np.sqrt(((u[where_common] - v[where_common]) ** 2).sum())

###################################################################################################



### Clustering functions ##########################################################################
def getEstimatedNumberOfClusters(data, cluster_num_min, cluster_num_max, trials_to_do, numberOfAvailableCPUs=4, plotID=None, printScores=False):

    """ Get estimated number of clusters using ARI with KMeans

    Args:
        data: data to analyze
        cluster_num_min: minimum possible number of clusters
        cluster_num_max: maximum possible number of clusters
        trials_to_do: number of trials to do in ARI function
        numberOfAvailableCPUs: number of processes to run in parallel
        plotID: label for the plot of peaks
        printScores: print all scores

    Returns: 
        Largest peak, other possible peaks.

    Usage:
        n_clusters = getEstimatedNumberOfClusters(data, 1, 20, 25)
    """

    def getPeakPosition(scores, makePlot=False, plotID=None):

        print()

        spline = UnivariateSpline(scores.T[0], scores.T[1])
        spline.set_smoothing_factor(0.005)
        xs = np.linspace(scores.T[0][0], scores.T[0][-1], 1000)
        data = np.vstack((xs, spline(xs))).T

        data_all = data.copy()
        data = data[data.T[0] > 4.]
        peaks = scipy.signal.find_peaks(data.T[1])[0]

        if len(peaks) == 0:
            selected_peak = 5
            print('WARNING: no peak found')
        else:
            selected_peak = np.round(data.T[0][peaks[np.argmax(data.T[1][peaks])]],0).astype(int)

        selected_peak_value = scores.T[1][np.argwhere(scores.T[0] == selected_peak)[0][0]]
        peaks = np.round(data.T[0][peaks],0).astype(int) if len(peaks) != 0 else peaks

        if makePlot:
            fig, ax = plt.subplots()

            ax.plot(data_all.T[0], data_all.T[1], 'g', lw=3)
            ax.plot(scores.T[0], scores.T[1], 'ro', ms=5)
            ax.plot(selected_peak, selected_peak_value, 'bo', alpha=0.5, ms=10)

            fig.savefig('spline_%s.png' % ('' if plotID == None else str(plotID)), dpi=300)
            plt.close(fig)

        print(selected_peak, peaks)

        return selected_peak, peaks

    print('Testing data clustering in a range of %s-%s clusters' % (cluster_num_min,cluster_num_max))
                
    scores = runCPUs(numberOfAvailableCPUs, runForClusterNum, [(cluster_num, copy.deepcopy(data), trials_to_do) for cluster_num in range(cluster_num_min, cluster_num_max + 1)])

    if printScores: 
        print(scores)
                
    return getPeakPosition(scores, makePlot=True, plotID=plotID)[0]


def get_optimal_number_clusters_from_linkage_Elbow(Y):

    """ Get optimal number clusters from linkage.
    A point of the highest accelleration of the fusion coefficient of the given linkage.

    Args:
        Y: linkage matrix

    Returns:
        Optimal number of clusters

    Usage:
        n_clusters = get_optimal_number_clusters_from_linkage_Elbow(Y)
    """

    return np.diff(np.array([[nc, Y[-nc + 1][2]] for nc in range(2,min(50,len(Y)))]).T[1], 2).argmax() + 1 if len(Y) >= 5 else 1


def get_optimal_number_clusters_from_linkage_Silhouette(Y, data, metric):

    """Determine the optimal number of cluster in data maximizing the Silhouette score.

    Args:
        Y: linkage matrix
        data: data to analyze
        metric: distance measure

    Returns:
        Optimal number of clusters

    Usage:
        n_clusters = get_optimal_number_clusters_from_linkage_Elbow(Y, data, 'euclidean')
    """

    max_score = 0
    n_clusters = 1

    distmatrix = squareform(pdist(data, metric=metric))

    for temp_n_clusters in range(2,10):
        print(temp_n_clusters, end='a, ', flush=True)
        temp_clusters = scipy.cluster.hierarchy.fcluster(Y, t=temp_n_clusters, criterion='maxclust')

        print(temp_n_clusters, end='b, ', flush=True)
        temp_score = sklearn.metrics.silhouette_score(distmatrix, temp_clusters, metric=metric)

        if temp_score>max_score:
            max_score = temp_score
            n_clusters = temp_n_clusters

    return n_clusters - 1


def runForClusterNum(arguments):
    
    """Calculate Adjusted Rand Index of the data for a range of cluster numbers.

    Args:
        arguments: a tuple of three parameters int the form
        (cluster_num, data_array, trials_to_do), where
        cluster_num: maximum number of clusters
        data_array: data to test
        trials_to_do: number of trials for each cluster number

    Returns:
        Numpy array

    Usage:
        instPool = multiprocessing.Pool(processes = NumberOfAvailableCPUs)
        scores = instPool.map(runForClusterNum, [(cluster_num, copy.deepcopy(data), trials_to_do) for cluster_num in range(cluster_num_min, cluster_num_max + 1)])
        instPool.close()
        instPool.join()
    """

    np.random.seed()

    cluster_num, data_array, trials_to_do = arguments

    print(cluster_num, end=', ', flush=True)

    labels = [KMeans(n_clusters=cluster_num).fit(data_array).labels_ for i in range(trials_to_do)]

    agreement_matrix = np.zeros((trials_to_do,trials_to_do))

    for i in range(trials_to_do):
        for j in range(trials_to_do):
            agreement_matrix[i, j] = adjusted_rand_score(labels[i], labels[j]) if agreement_matrix[j, i] == 0 else agreement_matrix[j, i]

    selected_data = agreement_matrix[np.triu_indices(agreement_matrix.shape[0],1)]

    return np.array((cluster_num, np.mean(selected_data), np.std(selected_data)))


def getGroupingIndex(data, n_groups=None, method='weighted', metric='correlation', significance='Elbow'):

    """Cluster data into N groups, if N is provided, else determine N
    return: linkage matrix, cluster labels, possible cluster labels.

    Args:
        data: data to analyze
        n_groups: number of groups to split data into
        method: linkage calculation method
        metric: distance measure
        significance: method for determining optimal number of groups and subgroups

    Returns:
        Linkage matrix, cluster index, possible groups

    Usage:
        x, y, z = getGroupingIndex(data, method='weighted', metric='correlation', significance='Elbow')
    """

    Y = hierarchy.linkage(data, method=method, metric=metric, optimal_ordering=False)

    if n_groups == None:
        if significance=='Elbow':
            n_groups = get_optimal_number_clusters_from_linkage_Elbow(Y)
        elif significance=='Silhouette':
            n_groups = get_optimal_number_clusters_from_linkage_Silhouette(Y, data, metric)

    print('n_groups:', n_groups)

    labelsClusterIndex = scipy.cluster.hierarchy.fcluster(Y, t=n_groups, criterion='maxclust')

    groups = np.sort(np.unique(labelsClusterIndex))

    print([np.sum(labelsClusterIndex == group) for group in groups])

    return Y, labelsClusterIndex, groups


def makeClusteringObject(df_data, df_data_autocorr, significance='Elbow'):

    """Make a clustering Groups-Subgroups dictionary object.

    Args:
        df_data: data to analyze in DataFrame format
        df_data_autocorr: autocorrelations or periodograms in DataFrame format
        significance: method for determining optimal number of groups and subgroups

    Returns:
        Clustering object

    Usage:
        myObj = makeClusteringObject(df_data, df_data_autocorr, significance='Elbow')
    """

    def getSubgroups(df_data, metric, significance):

        Y = hierarchy.linkage(df_data.values, method='weighted', metric=metric, optimal_ordering=True)
        leaves = hierarchy.dendrogram(Y, no_plot=True)['leaves']

        if significance=='Elbow':
            n_clusters = get_optimal_number_clusters_from_linkage_Elbow(Y)
        elif significance=='Silhouette':
            n_clusters = get_optimal_number_clusters_from_linkage_Silhouette(Y, df_data.values, metric)

        print('n_subgroups:', n_clusters)

        clusters = scipy.cluster.hierarchy.fcluster(Y, t=n_clusters, criterion='maxclust')[leaves]

        return {cluster:df_data.index[leaves][clusters==cluster] for cluster in np.unique(clusters)}, Y

    ClusteringObject = {}

    ClusteringObject['linkage'], labelsClusterIndex, groups = getGroupingIndex(df_data_autocorr.values, method='weighted', metric='correlation', significance=significance)

    for group in groups:
        signals = df_data.index[labelsClusterIndex==group]

        ClusteringObject[group], ClusteringObject[group]['linkage'] = ({1: signals}, None) if len(signals)==1 else getSubgroups(df_data.loc[signals], metricCommonEuclidean, significance)

        for subgroup in sorted([item for item in list(ClusteringObject[group].keys()) if not item=='linkage']):
            ClusteringObject[group][subgroup] = {'order':[np.where([temp==signal for temp in df_data.index.values])[0][0] for signal in list(ClusteringObject[group][subgroup])],
                                                 'data':df_data.loc[ClusteringObject[group][subgroup]], 
                                                 'dataAutocorr':df_data_autocorr.loc[ClusteringObject[group][subgroup]]}

    return ClusteringObject


def exportClusteringObject(ClusteringObject, saveDir, dataName, includeData=True, includeAutocorr=True):

    """Export a clustering Groups-Subgroups dictionary object to a SpreadSheet.
    Linkage data is not exported.

    Args:
        ClusteringObject: clustering object
        saveDir: path of directories to save the object to
        dataName: label to include in the file name
        includeData: export data 
        includeAutocorr: export autocorrelations of data

    Returns:
        File name of the exported clustering object

    Usage:
        exportClusteringObject(myObj, '/dir1', 'myObj')
    """

    if not os.path.exists(saveDir):
        os.makedirs(saveDir)

    fileName = saveDir + dataName + '_GroupsSubgroups.xlsx'

    writer = pd.ExcelWriter(fileName)

    for group in sorted([item for item in list(ClusteringObject.keys()) if not item=='linkage']):
        for subgroup in sorted([item for item in list(ClusteringObject[group].keys()) if not item=='linkage']):

            df_data = ClusteringObject[group][subgroup]['data']
            df_dataAutocorr = ClusteringObject[group][subgroup]['dataAutocorr']

            if includeData==True and includeAutocorr==True:
                df = pd.concat((df_data,df_dataAutocorr), sort = False, axis=1)
            elif includeData==True and includeAutocorr==False:
                df = df_data
            elif includeData==False and includeAutocorr==True:
                df = df_dataAutocorr
            else:
                df = pd.DataFrame(index=df_data.index)

            df.index.name = 'Index'
            df.to_excel(writer, 'G%sS%s'%(group, subgroup))

    writer.save()

    print('Saved clustering object to:', fileName)

    return fileName

###################################################################################################



### Visibility graph auxilary functions ###########################################################
@numba.jit(cache=True)
def getAdjacencyMatrixOfVisibilityGraph(data, times):

    """Calculate adjacency matrix of visibility graph.
    JIT-accelerated version (a bit faster than NumPy-accelerated version).
    Allows use of Multiple CPUs.

    Args:
        data: Numpy 2-D array of floats
        times: Numpy 1-D array of floats

    Returns:
        Adjacency matrix

    Usage:
        A = getAdjacencyMatrixOfVisibilityGraph_serial(data, times)
    """

    dimension = len(data)

    V = np.zeros((dimension,dimension))

    for i in range(dimension):
        for j in range(i + 1, dimension):
            V[i,j] = V[j,i] = (data[i] - data[j]) / (times[i] - times[j])

    A = np.zeros((dimension,dimension))

    for i in range(dimension):
        for j in range(i + 1, dimension):
            no_conflict = True

            for a in list(range(i+1,j)):
                if V[a,i] > V[j,i]:
                    no_conflict = False
                    break

            if no_conflict:
                A[i,j] = A[j,i] = 1

    return A


def getAdjacencyMatrixOfVisibilityGraph_NUMPY(data, times):

    """Calculate adjacency matrix of visibility graph.
    NumPy-accelerated version. Somewhat slower than JIT-accelerated version.
    Use in serial applications.

    Args:
        data: Numpy 2-D array of floats
        times: Numpy 1-D array of floats

    Returns:
        Adjacency matrix

    Usage:
        A = getAdjacencyMatrixOfVisibilityGraph_serial(data, times)
    """

    dimension = len(data)

    V = (np.subtract.outer(data, data))/(np.subtract.outer(times, times) + np.identity(dimension))

    A = np.zeros((dimension,dimension))

    for i in range(dimension):
        if i<dimension-1:
            A[i,i+1] = A[i+1,i] = 1

        for j in range(i + 2, dimension):
            if np.max(V[i+1:j,i])<=V[j,i]:
                A[i,j] = A[j,i] = 1

    return A


@numba.jit(cache=True)
def getAdjacencyMatrixOfHorizontalVisibilityGraph(data):

    """Calculate adjacency matrix of horizontal visibility graph.
    JIT-accelerated version (a bit faster than NumPy-accelerated version).
    Single-threaded beats NumPy up to 2k data sizes.
    Allows use of Multiple CPUs.

    Args:
        data: Numpy 2-D array of floats

    Returns:
        Adjacency matrix

    Usage:
        A = getAdjacencyMatrixOfHorizontalVisibilityGraph(data)
    """

    A = np.zeros((len(data),len(data)))

    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            no_conflict = True

            for a in list(range(i+1,j)):
                if data[a] > data[i] or data[a] > data[j]:
                    no_conflict = False
                    break

            if no_conflict:
                A[i,j] = A[j,i] = 1

    return A


def getAdjacencyMatrixOfHorizontalVisibilityGraph_NUMPY(data):

    """Calculate adjacency matrix of horizontal visibility graph.
    NumPy-accelerated version.
    Use with datasets larger than 2k.
    Use in serial applications.

    Args:
        data: Numpy 2-D array of floats

    Returns:
        Adjacency matrix

    Usage:
        A = getAdjacencyMatrixOfHorizontalVisibilityGraph_NUMPY(data)
    """

    dimension = len(data)

    A = np.zeros((dimension,dimension))

    for i in range(dimension):
        if i<dimension-1:
            A[i,i+1] = A[i+1,i] = 1

        for j in range(i + 2, dimension):
            if np.max(data[i+1:j])<=min(data[i], data[j]):
                A[i,j] = A[j,i] = 1

    return A

###################################################################################################



### Visualization functions #######################################################################
def makeDataHistograms(df, saveDir, dataName):

    """Make a histogram for each pandas Series (time point) in a pandas Dataframe.

    Args:
        df: DataFrame containing data to visualize
        saveDir: path of directories to save the object to
        dataName: label to include in the file name

    Returns:
        None

    Usage:
        makeDataHistograms(df, '/dir1', 'myData')
    """

    if not os.path.exists(saveDir):
        os.makedirs(saveDir)

    for timePoint in df.columns[:]:

        timeLabel = (str(np.round(timePoint,3)) + '000000')[:5]

        subset = df[timePoint]
        subset = subset[~np.isnan(subset.values)].values

        N_bins = 100

        range_min = np.min(subset)
        range_max = np.max(subset)

        hist_of_subset = scipy.stats.rv_histogram(np.histogram(subset, bins=N_bins, range=(range_min,range_max)))
        hist_data = hist_of_subset._hpdf / N_bins
        hist_bins = hist_of_subset._hbins

        fig, ax = plt.subplots(figsize=(8,8))

        bar_bin_width = range_max / N_bins

        ax.bar(hist_bins, hist_data[:-1], width=0.9 * bar_bin_width, color='b', align='center')

        ax.set_title('Data @ timePoint: ' + timeLabel, fontdict={'color': 'b'})
        ax.set_xlabel('Gene expression', fontsize=8)
        ax.set_ylabel('Density', fontsize=8)

        ax.set_xlim(range_min - 0.5 * bar_bin_width, range_max + 0.5 * bar_bin_width)

        fig.tight_layout()
        fig.savefig(saveDir + dataName + '_' + timeLabel + '_histogram_of_expression.png', dpi=600)

        plt.close(fig)

    return None


def makeLombScarglePeriodograms(df, saveDir, dataName):
        
    """Make a combined plot of the signal and its Lomb-Scargle periodogram
    for each pandas Series (time point) in a pandas Dataframe.

    Args:
        df: DataFrame containing data to visualize
        saveDir: path of directories to save the object to
        dataName: label to include in the file name

    Returns:
        None

    Usage:
        makeLombScarglePeriodograms(df, '/dir1', 'myData')
    """

    if not os.path.exists(saveDir):
        os.makedirs(saveDir)

    for geneIndex in range(len(df.index[:])):
        
        geneName = df.index[geneIndex].replace(':', '_')

        subset = df.iloc[geneIndex]
        subset = subset[subset > 0.]

        setTimes, setValues, inputSetTimes = subset.index.values, subset.values, df.columns.values

        if len(subset) < 5:

            print(geneName, ' skipped (only %s non-zero point%s), ' % (len(subset), 's' if len(subset) != 1 else ''), end=' ', flush=True)

            continue

        pgram = LombScargle(setTimes, setValues, inputSetTimes, OversamplingRate=100)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 5))

        ax1.plot(setTimes, setValues, 'bo', linewidth=3, markersize=5, markeredgecolor='k', markeredgewidth=2)

        zero_points = np.array(list(set(inputSetTimes) - set(setTimes)))
        ax1.plot(zero_points, np.zeros((len(zero_points),)), 'ro', linewidth=3, markersize=3, markeredgecolor='k', markeredgewidth=0)

        ax1.set_aspect('auto')

        minTime = np.min(inputSetTimes)
        maxTime = np.max(inputSetTimes)
        extraTime = (maxTime - minTime) / 10

        ax1.set_xlim(minTime - extraTime, maxTime + extraTime)
        ax1.set_title('TimeSeries Data')
    
        ax2.plot(2 * np.pi * pgram[0], pgram[1], 'r-', linewidth=1)

        ax2.set_aspect('auto')
        ax2.set_title('Lomb-Scargle periodogram')

        fig.tight_layout()

        fig.savefig(saveDir + dataName + '_' + geneName + '_Lomb_Scargle_periodogram.png', dpi=600)

        plt.close(fig)

    return None


def addVisibilityGraph(data, times, dataName='G1S1', coords=[0.05,0.95,0.05,0.95], 
                       numberOfVGs=1, groups_ac_colors=['b'], fig=None, numberOfCommunities=6, printCommunities=False, 
                       fontsize=None, nodesize=None, level=0.55, commLineWidth=0.5, lineWidth=1.0,
                       withLabel=True, withTitle=False, layout='circle', radius=0.07, noplot=False):

    """Draw a Visibility graph of data on a provided Matplotlib figure.

    Args:
        data: array of data to visualize
        times: times corresponding to each data point, used for labels
        dataName: label to include in file name
        coords: coordinates of location of the plot on the figure
        numberOfVGs: number of plots to add to this figure
        groups_ac_colors: colors corresponding to different groups of graphs
        fig: figure object
        printCommunities: print communities details to screen
        fontsize: size of labels
        nodesize: size of nodes
        level: distance of the community lines to nodes
        commLineWidth: width of the community lines
        lineWidth: width of the edges between nodes
        withLabel: include label on plot
        withTitle: include title on plot

    Returns:
        None

    Usage:
        addVisibilityGraph(exampleData, exampleTimes, fig=fig, fontsize=16, nodesize=700, 
                            level=0.85, commLineWidth=3.0, lineWidth=2.0, withLabel=False)
    """

    def imputeWithMedian(data):

        data[np.isnan(data)] = np.median(data[np.isnan(data) == False])

        return data

    if len(data.shape)>1:
        data = pd.DataFrame(data=data).apply(imputeWithMedian, axis=1).apply(lambda data: np.sum(data[data > 0.0]) / len(data), axis=0).values

    graph_nx = nx.from_numpy_matrix(getAdjacencyMatrixOfVisibilityGraph(data, times))
    
    def find_and_remove_node(graph_nx):
        bc = nx.betweenness_centrality(graph_nx)
        node_to_remove = list(bc.keys())[np.argmax(list(bc.values()))]
        graph_nx.remove_node(node_to_remove)
        return graph_nx, node_to_remove

    list_of_nodes = []
    graph_nx_inv = nx.from_numpy_matrix(getAdjacencyMatrixOfVisibilityGraph(-data, times))
    for i in range(numberOfCommunities):
        graph_nx_inv, node = find_and_remove_node(graph_nx_inv)
        list_of_nodes.append(node)
        
    if not 0 in list_of_nodes:
        list_of_nodes.append(0)

    list_of_nodes.append(list(graph_nx.nodes)[-1] + 1)
    list_of_nodes.sort()

    communities = [list(range(list_of_nodes[i],list_of_nodes[i + 1])) for i in range(len(list_of_nodes) - 1)]

    if printCommunities:
        print(list_of_nodes, '\n')
        [print(community) for community in communities]
        print()

    if noplot:
        return graph_nx, data, communities

    group = int(dataName[:dataName.find('S')].strip('G'))

    if fontsize is None:
        fontsize = 4. * (8. + 5.) / (numberOfVGs + 5.)
    
    if nodesize is None:
        nodesize = 30. * (8. + 5.) / (numberOfVGs + 5.)

    (x1,x2,y1,y2) = coords
    
    axisVG = fig.add_axes([x1,y1,x2 - x1,y2 - y1])
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('GR', [(0, 1, 0), (1, 0, 0)], N=1000)

    if layout=='line':
        pos = {i:[float(i)/float(len(graph_nx)), 0.5] for i in range(len(graph_nx))}
    else:
        pos = nx.circular_layout(graph_nx)
        keys = np.array(list(pos.keys())[::-1])
        values = np.array(list(pos.values()))
        values = (values - np.min(values, axis=0))/(np.max(values, axis=0)-np.min(values, axis=0))
        keys = np.roll(keys, np.argmax(values.T[1]) - np.argmin(keys))
        pos = dict(zip(keys, values))

    keys = np.array(list(pos.keys()))
    values = np.array(list(pos.values()))

    shortest_path = nx.shortest_path(graph_nx, source=min(keys), target=max(keys))
    shortest_path_edges = [(shortest_path[i],shortest_path[i + 1]) for i in range(len(shortest_path) - 1)]

    if layout=='line':
        for edge in graph_nx.edges:
            l = np.array(pos[edge[0]])
            r = np.array(pos[edge[1]])

            if edge in shortest_path_edges:
                axisVG.add_artist(matplotlib.patches.Wedge((l+r)/2., 0.5*np.sqrt((l-r)[0]*(l-r)[0]+(l-r)[1]*(l-r)[1]), 0, 180, fill=False, edgecolor='y', linewidth=0.5*3.*lineWidth, alpha=0.7, width=0.001))

            axisVG.add_artist(matplotlib.patches.Wedge((l+r)/2., 0.5*np.sqrt((l-r)[0]*(l-r)[0]+(l-r)[1]*(l-r)[1]), 0, 180, fill=False, edgecolor='k', linewidth=0.5*lineWidth, alpha=0.7, width=0.001))

        nx.draw_networkx(graph_nx, pos=pos, ax=axisVG, node_color='y', edge_color='y', node_size=nodesize * 1.7, width=0., nodelist=shortest_path, edgelist=shortest_path_edges, with_labels=False)
        nx.draw_networkx(graph_nx, pos=pos, ax=axisVG, node_color=data, cmap=cmap, alpha=1.0, font_size=fontsize,  width=0., font_color='k', node_size=nodesize)
    else:
        nx.draw_networkx(graph_nx, pos=pos, ax=axisVG, node_color='y', edge_color='y', node_size=nodesize * 1.7, width=3.0*lineWidth, nodelist=shortest_path, edgelist=shortest_path_edges, with_labels=False)
        nx.draw_networkx(graph_nx, pos=pos, ax=axisVG, node_color=data, cmap=cmap, alpha=1.0, font_size=fontsize,  width=lineWidth, font_color='k', node_size=nodesize)

    if layout=='line':
        xmin, xmax = (-1.,1.)
        ymin, ymax = (-1.,1.)
    else:
        xmin, xmax = axisVG.get_xlim()
        ymin, ymax = axisVG.get_ylim()

    X, Y = np.meshgrid(np.arange(xmin, xmax, (xmax - xmin) / 300.), np.arange(ymin, ymax, (ymax - ymin) / 300.))

    def smooth(Z, N=7.):
        for ix in range(1,Z.shape[0]-1,1):
            Z[ix] = ((N-1.)*Z[ix] + (Z[ix-1] + Z[ix+1])/2.)/N
        return Z

    for icommunity, community in enumerate(communities):
        Z = np.exp(X ** 2 - Y ** 2) * 0.
        nX, nY = tuple(np.array([pos[node] for node in community]).T)
        for i in range(len(community)-1):
            p1, p2 = np.array([nX[i], nY[i]]), np.array([nX[i+1], nY[i+1]])

            for j in range(-2, 32):
                pm = p1 + (p2-p1)*float(j)/30.
                Z[np.where((X-pm[0])**2+(Y-pm[1])**2<=radius**2)] = 1.
        
        for _ in range(20):
            Z = smooth(smooth(Z).T).T

        CS = axisVG.contour(X, Y, Z, [level], linewidths=commLineWidth, alpha=0.8, colors=groups_ac_colors[group - 1])
        #axisVG.clabel(CS, inline=True,fontsize=4,colors=group_colors[group-1], fmt ={level:'C%s'%icommunity})

    if layout=='line':
        axisVG.set_xlim(-0.1,1.)
        axisVG.set_ylim(-0.1,1.)

    axisVG.spines['left'].set_visible(False)
    axisVG.spines['right'].set_visible(False)
    axisVG.spines['top'].set_visible(False)
    axisVG.spines['bottom'].set_visible(False)
    axisVG.set_xticklabels([])
    axisVG.set_yticklabels([])
    axisVG.set_xticks([])
    axisVG.set_yticks([])

    if withLabel:
        axisVG.text(axisVG.get_xlim()[1], (axisVG.get_ylim()[1] + axisVG.get_ylim()[0]) * 0.5, dataName, ha='left', va='center',
                    fontsize=8).set_path_effects([path_effects.Stroke(linewidth=0.4, foreground=groups_ac_colors[group - 1]),path_effects.Normal()])

    if withTitle:
        titleText = dataName + ' (size: ' + str(data.shape[0]) + ')' + ' min=%s max=%s' % (np.round(min(data),2), np.round(max(data),2))
        axisVG.set_title(titleText, fontsize=10)

    return graph_nx, data, communities


def makeDendrogramHeatmap(ClusteringObject, saveDir, dataName, AutocorrNotPeriodogr=True, textScale=1.0, vectorImage=True):

    """Make Dendrogram-Heatmap plot along with VIsibility graphs.

    Args:
        ClusteringObject: clustering object
        saveDir: path of directories to save the object to
        dataName: label to include in the file name
        AutocorrNotPeriodogr: export data
        textScale: scaling of text size
        vectorImage: Boolean for exporting vector graphics or PNG format 

    Returns:
        None

    Usage:
        makeDendrogramHeatmap(myObj, '/dir1', 'myData', AutocorrNotPeriodogr=True)
    """

    def addAutocorrelationDendrogramAndHeatmap(ClusteringObject, groupColors, fig, AutocorrNotPeriodogr=AutocorrNotPeriodogr):

        axisDendro = fig.add_axes([0.68,0.1,0.17,0.8], frame_on=False)

        n_clusters = len(ClusteringObject.keys()) - 1
        hierarchy.set_link_color_palette(groupColors[:n_clusters]) #gist_ncar #nipy_spectral #hsv
        origLineWidth = matplotlib.rcParams['lines.linewidth']
        matplotlib.rcParams['lines.linewidth'] = 0.5
        Y_ac = ClusteringObject['linkage']
        Z_ac = hierarchy.dendrogram(Y_ac, orientation='left',color_threshold=Y_ac[-n_clusters + 1][2])
        hierarchy.set_link_color_palette(None)
        matplotlib.rcParams['lines.linewidth'] = origLineWidth
        axisDendro.set_xticks([])
        axisDendro.set_yticks([])

        posA = ((axisDendro.get_xlim()[0] if n_clusters == 1 else Y_ac[-n_clusters + 1][2]) + Y_ac[-n_clusters][2]) / 2
        axisDendro.plot([posA, posA], [axisDendro.get_ylim()[0], axisDendro.get_ylim()[1]], 'k--', linewidth = 1)

        clusters = []
        order = []
        tempData = None
        for group in sorted([item for item in list(ClusteringObject.keys()) if not item=='linkage']):
            for subgroup in sorted([item for item in list(ClusteringObject[group].keys()) if not item=='linkage']):
                subgroupData = ClusteringObject[group][subgroup]['dataAutocorr'].values
                tempData = subgroupData if tempData is None else np.vstack((tempData, subgroupData))
                clusters.extend([group for _ in range(subgroupData.shape[0])])
                order.extend(ClusteringObject[group][subgroup]['order'])

        tempData = tempData[np.argsort(order),:][Z_ac['leaves'],:].T[1:].T
        clusters = np.array(clusters)
        cluster_line_positions = np.where(clusters - np.roll(clusters,1) != 0)[0]

        for i in range(n_clusters - 1):
            posB = cluster_line_positions[i + 1] * (axisDendro.get_ylim()[1] / len(Z_ac['leaves'])) - 5. * 0
            axisDendro.plot([posA, axisDendro.get_xlim()[1]], [posB, posB], '--', color='black', linewidth = 1.0)
        
        axisDendro.plot([posA, axisDendro.get_xlim()[1]], [0., 0.], '--', color='k', linewidth = 1.0)
        axisDendro.plot([posA, axisDendro.get_xlim()[1]], [-0. + axisDendro.get_ylim()[1], -0. + axisDendro.get_ylim()[1]], '--', color='k', linewidth = 1.0)


        axisMatrixAC = fig.add_axes([0.78 + 0.07,0.1,0.18 - 0.075,0.8])

        cmap = plt.cm.bwr
        imAC = axisMatrixAC.imshow(tempData, aspect='auto', vmin=np.min(tempData), vmax=np.max(tempData), origin='lower', cmap=cmap)
        for i in range(n_clusters - 1):
            axisMatrixAC.plot([-0.5, tempData.shape[1] - 0.5], [cluster_line_positions[i + 1] - 0.5, cluster_line_positions[i + 1] - 0.5], '--', color='black', linewidth = 1.0)

        axisMatrixAC.set_xticks([i for i in range(tempData.shape[1] - 1)])
        axisMatrixAC.set_xticklabels([i + 1 for i in range(tempData.shape[1] - 1)], fontsize=6*textScale)
        axisMatrixAC.set_yticks([])
        axisMatrixAC.set_xlabel('Lag' if AutocorrNotPeriodogr else 'Frequency', fontsize=axisMatrixAC.xaxis.label._fontproperties._size*textScale)
        axisMatrixAC.set_title('Autocorrelation' if AutocorrNotPeriodogr else 'Periodogram', fontsize=axisMatrixAC.title._fontproperties._size*textScale)

        axisColorAC = fig.add_axes([0.9 + 0.065,0.55,0.01,0.35])

        axisColorAC.tick_params(labelsize=6*textScale)
        plt.colorbar(imAC, cax=axisColorAC, ticks=[np.round(np.min(tempData),2),np.round(np.max(tempData),2)])

        return

    def addGroupDendrogramAndShowSubgroups(ClusteringObject, groupSize, bottom, top, group, groupColors, fig):

        Y = ClusteringObject[group]['linkage']

        n_clusters = len(sorted([item for item in list(ClusteringObject[group].keys()) if not item=='linkage']))
        print("Number of subgroups:", n_clusters)

        axisDendro = fig.add_axes([left, bottom, dx + 0.005, top - bottom], frame_on=False)

        hierarchy.set_link_color_palette([matplotlib.colors.rgb2hex(rgb[:3]) for rgb in cm.nipy_spectral(np.linspace(0, 0.5, n_clusters + 1))]) #gist_ncar #nipy_spectral #hsv
        origLineWidth = matplotlib.rcParams['lines.linewidth']
        matplotlib.rcParams['lines.linewidth'] = 0.5
        Z = hierarchy.dendrogram(Y, orientation='left',color_threshold=Y[-n_clusters + 1][2])
        hierarchy.set_link_color_palette(None)
        matplotlib.rcParams['lines.linewidth'] = origLineWidth
        axisDendro.set_xticks([])
        axisDendro.set_yticks([])

        posA = axisDendro.get_xlim()[0]/2 if n_clusters == groupSize else (Y[-n_clusters + 1][2] + Y[-n_clusters][2])/2 
        axisDendro.plot([posA, posA], [axisDendro.get_ylim()[0], axisDendro.get_ylim()[1]], 'k--', linewidth = 1)

        clusters = []
        for subgroup in sorted([item for item in list(ClusteringObject[group].keys()) if not item=='linkage']):
            clusters.extend([subgroup for _ in range(ClusteringObject[group][subgroup]['data'].values.shape[0])])

        clusters = np.array(clusters)

        cluster_line_positions = np.where(clusters - np.roll(clusters,1) != 0)[0]

        for i in range(n_clusters - 1):
            posB = cluster_line_positions[i + 1] * (axisDendro.get_ylim()[1] / len(Z['leaves'])) - 5. * 0
            axisDendro.plot([posA, axisDendro.get_xlim()[1]], [posB, posB], '--', color='black', linewidth = 1.0)
        
        axisDendro.plot([posA, axisDendro.get_xlim()[1]], [0., 0.], '--', color='k', linewidth = 1.0)
        axisDendro.plot([posA, axisDendro.get_xlim()[1]], [-0. + axisDendro.get_ylim()[1], -0. + axisDendro.get_ylim()[1]], '--', color='k', linewidth = 1.0)

        axisDendro.text(axisDendro.get_xlim()[0], 0.5 * axisDendro.get_ylim()[1], 
                        'G%s:' % group + str(groupSize), fontsize=14*textScale).set_path_effects([path_effects.Stroke(linewidth=1, foreground=groupColors[group - 1]),path_effects.Normal()])

        return n_clusters, clusters, cluster_line_positions

    def addGroupHeatmapAndColorbar(data_loc, n_clusters, clusters, cluster_line_positions, bottom, top, group, groupColors, fig):

        axisMatrix = fig.add_axes([left + 0.205, bottom, dx + 0.025 + 0.075, top - bottom])

        masked_array = np.ma.array(data_loc, mask=np.isnan(data_loc))

        cmap = matplotlib.colors.LinearSegmentedColormap.from_list('GR', [(0, 1, 0), (1, 0, 0)], N=1000) #plt.cm.prism #plt.cm.hsv_r #plt.cm.RdYlGn_r
        cmap.set_bad('grey')
        im = axisMatrix.imshow(masked_array, aspect='auto', origin='lower', vmin=np.min(data_loc[np.isnan(data_loc) == False]), vmax=np.max(data_loc[np.isnan(data_loc) == False]), cmap=cmap)

        for i in range(n_clusters - 1):
            posB = cluster_line_positions[i + 1]
            posBm = cluster_line_positions[i]
            axisMatrix.plot([-0.5, data_loc.shape[1] - 0.5], [posB - 0.5, posB - 0.5], '--', color='black', linewidth = 1.0)

        def add_label(pos, labelText):
            return axisMatrix.text(-1., pos, labelText, ha='right', va='center', fontsize=12.*textScale).set_path_effects([path_effects.Stroke(linewidth=0.4, foreground=groupColors[group - 1]),path_effects.Normal()])

        order = clusters[np.sort(np.unique(clusters,return_index=True)[1])] - 1


        for i in range(n_clusters - 1):
            if len(data_loc[clusters == i + 1]) >= 5.:
                try:
                    add_label((cluster_line_positions[np.where(order == i)[0][0]] + cluster_line_positions[np.where(order == i)[0][0] + 1]) * 0.5, 'G%sS%s:%s' % (group,i + 1,len(data_loc[clusters == i + 1])))
                except:
                    print('Label printing error!')
        if len(data_loc[clusters == n_clusters]) >= 5.:
            posC = axisMatrix.get_ylim()[0] if n_clusters == 1 else cluster_line_positions[n_clusters - 1]
            add_label((posC + axisMatrix.get_ylim()[1]) * 0.5, 'G%sS%s:%s' % (group,n_clusters,len(data_loc[clusters == n_clusters])))

        axisMatrix.set_xticks([])
        axisMatrix.set_yticks([])

        times = ClusteringObject[group][subgroup]['data'].columns.values

        if group == 1:
            axisMatrix.set_xticks(range(data_loc.shape[1]))
            axisMatrix.set_xticklabels([('' if (i%2==1 and textScale>1.3) else np.int(time)) for i, time in enumerate(np.round(times,1))], rotation=0, fontsize=6*textScale)
            axisMatrix.set_xlabel('Time (hours)', fontsize=axisMatrix.xaxis.label._fontproperties._size*textScale)

        if group == sorted([item for item in list(ClusteringObject.keys()) if not item=='linkage'])[-1]:
            axisMatrix.set_title('Transformed gene expression', fontsize=axisMatrix.title._fontproperties._size*textScale)

        axisColor = fig.add_axes([0.635 - 0.075 - 0.1 + 0.075,current_bottom + 0.01,0.01, max(0.01,(current_top - current_bottom) - 0.02)])
        plt.colorbar(im, cax=axisColor, ticks=[np.max(im._A),np.min(im._A)])
        axisColor.tick_params(labelsize=6*textScale)
        axisColor.set_yticklabels([np.round(np.max(im._A),2),np.round(np.min(im._A),2)])

        return

    signalsInClusteringObject = 0
    for group in sorted([item for item in list(ClusteringObject.keys()) if not item=='linkage']):
        for subgroup in sorted([item for item in list(ClusteringObject[group].keys()) if not item=='linkage']):
            signalsInClusteringObject += ClusteringObject[group][subgroup]['data'].shape[0]


    fig = plt.figure(figsize=(12,8))

    left = 0.02
    bottom = 0.1
    current_top = bottom
    dx = 0.2
    dy = 0.8

    groupColors = ['b','g','r','c','m','y','k']
    [groupColors.extend(groupColors) for _ in range(10)]

    addAutocorrelationDendrogramAndHeatmap(ClusteringObject, groupColors, fig)

    for group in sorted([item for item in list(ClusteringObject.keys()) if not item=='linkage']):

        tempData = None
        for subgroup in sorted([item for item in list(ClusteringObject[group].keys()) if not item=='linkage']):
            subgroupData = ClusteringObject[group][subgroup]['data'].values
            tempData = subgroupData if tempData is None else np.vstack((tempData, subgroupData))

        current_bottom = current_top
        current_top += dy * float(len(tempData)) / float(signalsInClusteringObject)

        if len(tempData)==1:
            n_clusters, clusters, cluster_line_positions = 1, np.array([]), np.array([])
        else:
            n_clusters, clusters, cluster_line_positions = addGroupDendrogramAndShowSubgroups(ClusteringObject, len(tempData), current_bottom, current_top, group, groupColors, fig)

        addGroupHeatmapAndColorbar(tempData, n_clusters, clusters, cluster_line_positions, current_bottom, current_top, group, groupColors, fig)

    data_list = []
    data_names_list = []
    for group in sorted([item for item in list(ClusteringObject.keys()) if not item=='linkage']):
        for subgroup in sorted([item for item in list(ClusteringObject[group].keys()) if not item=='linkage']):
            if ClusteringObject[group][subgroup]['data'].shape[0] >= 5:
                data_list.append(ClusteringObject[group][subgroup]['data'].values)
                data_names_list.append('G%sS%s' % (group, subgroup))

    times = ClusteringObject[group][subgroup]['data'].columns.values

    for indVG, (dataVG, dataNameVG) in enumerate(zip(data_list, data_names_list)):
        
        x_min = 0.57
        x_max = 0.66
        y_min = 0.1
        y_max = 0.9

        numberOfVGs = len(data_list)
        height = min((y_max - y_min) / numberOfVGs, (x_max - x_min) * (12. / 8.))
        x_displacement = (x_max - x_min - height / 1.5) * 0.5
        y_displacement = (y_max - y_min - numberOfVGs * height) / numberOfVGs

        coords = [x_min + x_displacement, x_min + x_displacement + height / (12. / 8.), y_min + indVG * height + (0.5 + indVG) * y_displacement, y_min + (indVG + 1) * height + (0.5 + indVG) * y_displacement]

        addVisibilityGraph(dataVG, times, dataNameVG, coords, numberOfVGs, groupColors, fig)
    
    if vectorImage:
        fig.savefig(saveDir + dataName + '_DendrogramHeatmap.eps')
        fig.savefig(saveDir + dataName + '_DendrogramHeatmap.svg')
    else:
        fig.savefig(saveDir + dataName + '_DendrogramHeatmap.png', dpi=300)

    return None


def PlotVisibilityGraph(A, data, times, fileName, id):

    """Bar-plot style visibility graph.

    Args:
        A: Adjacency matrix
        data: Numpy 2-D array of floats
        times: Numpy 1-D array of floats
        fileName: name of the figure file to save
        id: label to add to the figure title

    Returns:
        None

    Usage:
        PlotVisibilityGraph(A, data, times, 'FIgure.png', 'Test Data')
    """

    fig, ax = plt.subplots(figsize=(8,4))
    ax.bar(times, data, width = 0.03, color='r', align='center', zorder=-np.inf)
    ax.scatter(times, data, color='b')

    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if i>j and A[i,j] == 1:
                ax.annotate(s='', xy=(times[i],data[i]), xytext=(times[j],data[j]), 
                            arrowprops=dict(arrowstyle='<->', shrinkA=0, shrinkB=0,linestyle='--'))

    ax.set_title('%s Time Series'%(id), fontdict={'color': 'k'})
    ax.set_xlabel('Times', fontsize=8)
    ax.set_ylabel('Signal intensity', fontsize=8)
    ax.set_xticks(times)
    ax.set_xticklabels([str(item)[:-2]+' hr' for item in np.round(times,0)],fontsize=10, rotation=90)
    ax.set_yticks([])

    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(True)

    fig.tight_layout()
    fig.savefig(fileName, dpi=600)
    plt.close(fig)

    return None


def PlotHorizontalVisibilityGraph(A, data, times, fileName, id):
    
    """Bar-plot style horizontal visibility graph.

    Args:
        A: Adjacency matrix
        data: Numpy 2-D array of floats
        times: Numpy 1-D array of floats
        fileName: name of the figure file to save
        id: label to add to the figure title

    Returns:
        None

    Usage:
        PlotHorizontalVisibilityGraph(A, data, times, 'FIgure.png', 'Test Data')
    """

    fig, ax = plt.subplots(figsize=(8,4))
    ax.bar(times, data, width = 0.03, color='r', align='center', zorder=-np.inf)
    ax.scatter(times, data, color='b')

    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if i>j and A[i,j] == 1:
                level = np.min([data[i],data[j]])
                ax.annotate(s='', xy=(times[i],level), xytext=(times[j],level), 
                            arrowprops=dict(arrowstyle='<->', shrinkA=0, shrinkB=0,linestyle='--'))

    ax.set_title('%s Time Series'%(id), fontdict={'color': 'k'})
    ax.set_xlabel('Times', fontsize=8)
    ax.set_ylabel('Signal intensity', fontsize=8)
    ax.set_xticks(times)
    ax.set_xticklabels([str(item)[:-2]+' hr' for item in np.round(times,0)],fontsize=10, rotation=90)
    ax.set_yticks([])

    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(True)

    fig.tight_layout()
    fig.savefig(fileName, dpi=600)
    plt.close(fig)
    
    return None

###################################################################################################



### Dataframe functions ###########################################################################
def prepareDataframe(dataDir, dataFileName, AlltimesFileName):

    """Make a DataFrame from CSV files.
    
    Args:
        dataDir: path of directories pointing to data
        dataFileName: file name in dataDir
        AlltimesFileName: file name in dataDir

    Returns:
        Pandas Dataframe

    Usage:
        df_data = prepareDataframe(dataDir, dataFileName, AlltimesFileName)
        df_data.index = pd.MultiIndex.from_tuples([(item.split(':')[1], item.split(':')[0].split('_')[0],
                                                    (' '.join(item.split(':')[0].split('_')[1:]),)) for item in df_data.index.values], 
                                                    names=['source', 'id', 'metadata'])
    """

    df = pd.read_csv(os.path.join(dataDir, dataFileName), delimiter=',', header=None)

    df = df.set_index(df[df.columns[0]]).drop(columns=[df.columns[0]])

    df.columns = list(pd.read_csv(os.path.join(dataDir, AlltimesFileName), delimiter=',', header=None).values.T[0])

    return df


def filterOutAllZeroSignalsDataframe(df):

    """Filter out all-zero signals from a DataFrame.
    
    Args:
        df: pandas DataFrame

    Returns:
        Processed pandas Dataframe

    Usage:
        df_data = filterOutAllZeroSignalsDataframe(df_data)
   """

    print('Filtering out all-zero signals...')

    init = df.shape[0]

    df = df.loc[df.index[np.count_nonzero(df, axis=1) > 0]]

    print('Removed ', init - df.shape[0], 'signals out of %s.' % init) 
    print('Remaining ', df.shape[0], 'signals!')

    return df


def filterOutFractionZeroSignalsDataframe(df, max_fraction_of_allowed_zeros):
       
    """Filter out fraction-zero signals from a DataFrame.
    
    Args:
        df: pandas DataFrame
        max_fraction_of_allowed_zeros: maximum fraction of allowed zeros

    Returns:
        Processed pandas Dataframe

    Usage:
        df_data = filterOutFractionZeroSignalsDataframe(df_data, 0.75)
   """

    print('Filtering out low-quality signals (with more than %s%% missing points)...' %(100.*(1.-max_fraction_of_allowed_zeros)))

    init = df.shape[0]

    min_number_of_non_zero_points = np.int(np.round(max_fraction_of_allowed_zeros * df.shape[1],0))
    df = df.loc[df.index[np.count_nonzero(df, axis=1) >= min_number_of_non_zero_points]]

    if (init - df.shape[0]) > 0:
        print('Removed ', init - df.shape[0], 'signals out of %s.' % init) 
        print('Remaining ', df.shape[0], 'signals!')

    return df


def filterOutFirstPointZeroSignalsDataframe(df):

    """Filter out out first time point zeros signals from a DataFrame.
    
    Args:
        df: pandas DataFrame

    Returns:
        Processed pandas Dataframe

    Usage:
        df_data = filterOutFirstPointZeroSignalsDataframe(df_data)
   """

    print('Filtering out first time point zeros signals...')

    init = df.shape[0]

    df = df.loc[~(df.iloc[:,0] == 0.0)]

    if (init - df.shape[0]) > 0:
        print('Removed ', init - df.shape[0], 'signals out of %s.' % init) 
        print('Remaining ', df.shape[0], 'signals!')

    return df


def tagMissingValuesDataframe(df):

    """Tag missing (i.e. zero) values with NaN.
    
    Args:
        df: pandas DataFrame

    Returns:
        Processed pandas Dataframe

    Usage:
        df_data = tagMissingValuesDataframe(df_data)
    """

    print('Tagging missing (i.e. zero) values with NaN...')

    df[df == 0.] = np.NaN

    return df


def tagLowValuesDataframe(df, cutoff, replacement):

    """Tag low values with replacement value.
    
    Args:
        df: pandas DataFrame
        cutoff: values below the "cutoff" are replaced with "replacement" value
        replacement: replacement value

    Returns:
        Processed pandas Dataframe

    Usage:
        df_data = tagLowValuesDataframe(df_data, 1., 1.)
    """

    print('Tagging low values (<=%s) with %s...'%(cutoff, replacement))

    df[df <= cutoff] = replacement

    return df


def removeConstantSignalsDataframe(df, theta_cutoff):

    """Remove constant signals.
    
    Args:
        df: pandas DataFrame
        theta_cutoff: parameter for filtering the signals

    Returns:
        Processed pandas Dataframe

    Usage:
        df_data = removeConstantSignalsDataframe(df_data, 0.3)
    """

    print('\nRemoving constant genes. Cutoff value is %s' % (theta_cutoff))

    init = df.shape[0]

    df = df.iloc[np.where(np.std(df,axis=1) / np.mean(np.std(df,axis=1)) > theta_cutoff)[0]]

    print('Removed ', init - df.shape[0], 'signals out of %s.' % init)
    print('Remaining ', df.shape[0], 'signals!')

    return df


def boxCoxTransformDataframe(df):

    """Box-cox transform data.
    
    Args:
        df: pandas DataFrame

    Returns:
        Processed pandas Dataframe

    Usage:
        df_data = boxCoxTransformDataframe(df_data)
    """
    
    print('Box-cox transforming raw data...', end='\t', flush=True)
            
    df = df.apply(boxCoxTransform, axis=0)

    print('Done')

    return df


def modifiedZScoreDataframe(df):

    """Z-score (Median-based) transform data.
    
    Args:
        df: pandas DataFrame

    Returns:
        Processed pandas Dataframe

    Usage:
        df_data = modifiedZScoreDataframe(df_data)
    """
            
    print('Z-score (Median-based) transforming box-cox transformed data...', end='\t', flush=True)

    df = df.apply(modifiedZScore, axis=0)

    print('Done')

    return df


def normalizeSignalsToUnityDataframe(df):

    """Normalize signals to unity.
    
    Args:
        df: pandas DataFrame

    Returns:
        Processed pandas Dataframe

    Usage:
        df_data = normalizeSignalsToUnityDataframe(df_data)
    """

    print('Normalizing signals to unity...')

    #Subtract 0-time-point value from all time-points
    df = compareTimeSeriesToPointDataframe(df, point='first')
    
    where_nan = np.isnan(df.values.astype(float))
    df[where_nan] = 0.0
    df = df.apply(lambda data: data / np.sqrt(np.dot(data,data)),axis=1)
    df[where_nan] = np.nan

    return df


def quantileNormalizeDataframe(df):

    """Quantile Normalize signals to normal distribution.
    
    Args:
        df: pandas DataFrame

    Returns:
        Processed pandas Dataframe

    Usage:
        df_data = quantileNormalizeDataframe(df_data)
    """

    print('Quantile normalizing signals...')

    df.iloc[:] = quantile_transform(df.values, output_distribution='normal', n_quantiles=min(df.shape[0],1000), copy=False)

    return df


def compareTimeSeriesToPointDataframe(df, point='first'):

    """Subtract a particular point of each time series (row) of a Dataframe.
    
    Args:
        df: pandas DataFrame
        point: 'first', 'last', 0, 1, ... , 10, or a value.

    Returns:
        Processed pandas Dataframe

    Usage:
        df_data = compareTimeSeriesToPointDataframe(df_data)
    """

    independent = True

    if point == 'first':
        idx = 0
    elif point == 'last':
        idx = len(df.columns) - 1
    elif type(point) is int:
        idx = point
    elif type(point) is float:
        independent = False
    else:
        print("Specify a valid comparison point: 'first', 'last', 0, 1, ..., 10, or a value")
        return

    if independent:
        df.iloc[:] = (df.values.T - df.values.T[idx]).T
    else:
        df.iloc[:] = (df.values.T - point).T

    return df


def compareTwoTimeSeriesDataframe(df1, df2, function=np.subtract, compareAllLevelsInIndex=True, mergeFunction=np.mean):

    """Create a new Dataframe based on comparison of two existing Dataframes.
    
    Args:
        df1: pandas DataFrame
        df2: pandas DataFrame
        function: np.subtract (default), np.add, np.divide, or another <ufunc>.
        compareAllLevelsInIndex: True (default), if False only "source" and "id" will be compared,
        mergeFunction: input Dataframes are merged with this function, i.e. np.mean (default), np.median, np.max, or another <ufunc>.

    Returns:
        New merged pandas Dataframe

    Usage:
        df_data = compareTwoTimeSeriesDataframe(df_dataH2, df_dataH1, function=np.subtract, compareAllLevelsInIndex=False, mergeFunction=np.median)
    """

    if df1.index.names!=df2.index.names:
        errMsg = 'Index of Dataframe 1 is not of the same shape as index of Dataframe 2!'
        print(errMsg)
        return errMsg

    if compareAllLevelsInIndex:
        df1_grouped, df2_grouped = df1, df2
    else:
        def aggregate(df):
            return df.groupby(level=['source', 'id']).agg(mergeFunction)

        df1_grouped, df2_grouped = aggregate(df1), aggregate(df2)

    index = pd.MultiIndex.from_tuples(list(set(df1_grouped.index.values).intersection(set(df2_grouped.index.values))), 
                                      names=df1_grouped.index.names)

    return function(df1_grouped.loc[index], df2_grouped.loc[index])


def mergeDataframes(listOfDataframes):

    """Merge a list of Dataframes (outer join).
    
    Args:
        listOfDataframes: list of pandas DataFrames

    Returns:
        New pandas Dataframe

    Usage:
        df_data = mergeDataframes([df_data1, df_data2])
    """

    if len(listOfDataframes)==0:
        return None
    elif len(listOfDataframes)==1:
        return listOfDataframes[0]

    df = pd.concat(listOfDataframes, sort=False, axis=0)

    return df


def getLobmScarglePeriodogramOfDataframe(df_data, NumberOfCPUs=4, parallel=True):

    """Calculate Lobm-Scargle periodogram of DataFrame.
    
    Args:
        df: pandas DataFrame
        parallel: calculate in parallel mode (>1 process)
        NumberOfCPUs: number of processes to create if parallel

    Returns:
        New pandas Dataframe

    Usage:
        df_periodograms = getLobmScarglePeriodogramOfDataframe(df_data)
    """

    if parallel:

        results = runCPUs(NumberOfCPUs, pLombScargle, [(series.index[~np.isnan(series)].values, series[~np.isnan(series)].values, df_data.columns.values) for index, series in df_data.iterrows()])

        df_periodograms = pd.DataFrame(data=results[1::2], index=df_data.index, columns=results[0])

    else:
        frequencies = None
        intensities = []

        for index, series in df_data.iterrows():
            values = series[~np.isnan(series)].values
            times = series.index[~np.isnan(series)].values

            tempFrequencies, tempIntensities = LombScargle(times, values, series.index.values, OversamplingRate=1)

            if frequencies is None:
                frequencies = tempFrequencies

            intensities.append(tempIntensities)

        df_periodograms = pd.DataFrame(data=np.vstack(intensities), index=df_data.index, columns=frequencies)

    return df_periodograms


def hdf5_usage_information():

    """Store/export any lagge datasets in hdf5 format via 'pandas' or 'h5py'

    # mode='w' creates/recreates file from scratch
    # mode='a' creates (if no file exists) or appends to the existing file, and reads it
    # mode='r' is read only

    # Save data to file using 'pandas': 
    df_example = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]}, index=['a', 'b', 'c'])
    df_example.to_hdf('data.h5', key='my_df1', mode='a')
    or
    series_example = pd.Series([1, 2, 3, 4])
    series_example.to_hdf('data.h5', key='my_series', mode='a')

    # Create groups and datasets using 'h5py' and 'numpy' arrays:
    tempFile = h5py.File('data.h5', 'a')
    tempArray = np.array([[1,2,3,4,5],[6,7,8,9,10]]).astype(float)
    if not 'arrays/my_array' in tempFile:
        dataset_example = tempFile.create_dataset('arrays/my_array', data=tempArray, maxshape=(None,2), dtype=tempArray.dtype, 
                                                    chunks=True) #auto-chunked, else use e.g. chunks=(100, 2)
                                                    #compression='gzip', compression_opts=6
    else:
        dataset_example = tempFile['arrays/my_array']

    group_example = tempFile.create_group('more_data/additional')

    # Modify values by slicing the dataset or replacing etire one using [...]
    dataset_example[:] = np.array([[10,2,3,4,1],[60,7,8,9,1]])

    # New shapes cannot be broadcasted, the dataset needs to be resized explicitly
    dataset_example.resize(dataset_example.shape[0]+10, axis=0) #add more rows (initiated with zeros)


    # Read data from h5 file:
    df_example = pd.read_hdf('data.h5', 'my_df1')

    tempFile = h5py.File('data.h5', 'r')
    array_example = tempFile['arrays/my_array'].value
    """

    print(hdf5_usage_information.__doc__)

    return None

###################################################################################################



### Data processing functions #####################################################################
def timeSeriesClassification(df_data, dataName, saveDir, hdf5fileName=None, p_cutoff=0.05,
                             NumberOfRandomSamples=10**5, NumberOfCPUs=4, frequencyBasedClassification=False, 
                             calculateAutocorrelations=False, calculatePeriodograms=False):
        
    """Time series classification.
    
    Args:
        df_data: pandas DataFrame
        dataName: data name, e.g. "myData_1"
        saveDir: path of directories poining to data storage
        hdf5fileName: preferred hdf5 file name and location
        p_cutoff: significance cutoff signals selection
        NumberOfRandomSamples: size of the bootstrap distribution to generate
        NumberOfCPUs: number of processes allowed to use in calculations
        frequencyBasedClassification: whether Autocorrelation of Frequency based
        calculateAutocorrelations: whether to recalculate Autocorrelations
        calculatePeriodograms: whether to recalculate Periodograms

    Returns:
        None

    Usage:
        timeSeriesClassification(df_data, dataName, saveDir, NumberOfRandomSamples = 10**5, NumberOfCPUs = 4, p_cutoff = 0.05, frequencyBasedClassification=False)
    """

    print('\n', '-'*70, '\n\tProcessing %s (%s)'%(dataName, 'Periodograms' if frequencyBasedClassification else 'Autocorrelations'), '\n', '-'*70)

    if not os.path.exists(saveDir):
        os.makedirs(saveDir)

    if hdf5fileName is None:
        hdf5fileName = saveDir + dataName + '.h5'

    df_data = filterOutAllZeroSignalsDataframe(df_data)
    df_data = filterOutFirstPointZeroSignalsDataframe(df_data)
    df_data = filterOutFractionZeroSignalsDataframe(df_data, 0.75)
    df_data = tagMissingValuesDataframe(df_data)
    df_data = tagLowValuesDataframe(df_data, 1., 1.)
    df_data = removeConstantSignalsDataframe(df_data, 0.)

    write(df_data, saveDir + dataName + '_df_data_transformed', hdf5fileName=hdf5fileName)

    if frequencyBasedClassification:
        calculateAutocorrelations = False
        if not calculatePeriodograms:
            df_dataPeriodograms = read(saveDir + dataName + '_dataPeriodograms', hdf5fileName=hdf5fileName)
            df_randomPeriodograms = read(saveDir + dataName + '_randomPeriodograms', hdf5fileName=hdf5fileName)
        
            if (df_dataPeriodograms is None) or (df_randomPeriodograms is None):
                print('Periodograms of data and the corresponding null distribution not found. Calculating...')
                calculatePeriodograms = True
    else:
        calculatePeriodograms = False
        if not calculateAutocorrelations:
            df_dataAutocorrelations = read(saveDir + dataName + '_dataAutocorrelations', hdf5fileName=hdf5fileName)
            df_randomAutocorrelations = read(saveDir + dataName + '_randomAutocorrelations', hdf5fileName=hdf5fileName)
        
            if (df_dataAutocorrelations is None) or (df_randomAutocorrelations is None):
                print('Autocorrelation of data and the corresponding null distribution not found. Calculating...')
                calculateAutocorrelations = True

    if calculatePeriodograms:
        df_data = read(saveDir + dataName + '_df_data_transformed', hdf5fileName=hdf5fileName)

        print('Calculating null distribution (periodogram) of %s samples...' %(NumberOfRandomSamples))
        df_randomPeriodograms = getRandomPeriodograms(df_data, NumberOfRandomSamples=NumberOfRandomSamples, NumberOfCPUs=NumberOfCPUs)

        write(df_randomPeriodograms, saveDir + dataName + '_randomPeriodograms', hdf5fileName=hdf5fileName)

        df_data = read(saveDir + dataName + '_df_data_transformed', hdf5fileName=hdf5fileName)
        df_data = normalizeSignalsToUnityDataframe(df_data)

        print('Calculating each Time Series Periodogram...')
        df_dataPeriodograms = getLobmScarglePeriodogramOfDataframe(df_data)

        write(df_dataPeriodograms, saveDir + dataName + '_dataPeriodograms', hdf5fileName=hdf5fileName)

    if calculateAutocorrelations:
        df_data = read(saveDir + dataName + '_df_data_transformed', hdf5fileName=hdf5fileName)

        print('Calculating null distribution (autocorrelation) of %s samples...' %(NumberOfRandomSamples))
        df_randomAutocorrelations = getRandomAutocorrelations(df_data, NumberOfRandomSamples=NumberOfRandomSamples, NumberOfCPUs=NumberOfCPUs)

        write(df_randomAutocorrelations, saveDir + dataName + '_randomAutocorrelations', hdf5fileName=hdf5fileName)

        df_data = read(saveDir + dataName + '_df_data_transformed', hdf5fileName=hdf5fileName)
        df_data = normalizeSignalsToUnityDataframe(df_data)

        print('Calculating each Time Series Autocorrelations...')
        df_dataAutocorrelations = runCPUs(NumberOfCPUs, getAutocorrelationsOfData, [(df_data.iloc[i], df_data.columns.values) for i in range(len(df_data.index))])

        df_dataAutocorrelations = pd.DataFrame(data=df_dataAutocorrelations[1::2], index=df_data.index, columns=df_dataAutocorrelations[0])
        df_dataAutocorrelations.columns = ['Lag ' + str(column) for column in df_dataAutocorrelations.columns]
        write(df_dataAutocorrelations, saveDir + dataName + '_dataAutocorrelations', hdf5fileName=hdf5fileName)

    df_data = read(saveDir + dataName + '_df_data_transformed', hdf5fileName=hdf5fileName)

    if frequencyBasedClassification:
        df_classifier = df_dataPeriodograms
        df_randomClassifier = df_randomPeriodograms
        info = 'Periodograms'
    else:
        df_classifier = df_dataAutocorrelations
        df_randomClassifier = df_randomAutocorrelations
        info = 'Autocorrelations'

    df_classifier.sort_index(inplace=True)
    df_data.sort_index(inplace=True)

    if not (df_data.index.values == df_classifier.index.values).all():
        raise ValueError('Index mismatch')
            
    QP = [1.0]
    QP.extend([np.quantile(df_randomClassifier.values.T[i], 1. - p_cutoff,interpolation='lower') for i in range(1,df_classifier.shape[1])])
    print('Quantiles:', list(np.round(QP, 16)), '\n')

    significant_index = np.vstack([df_classifier.values.T[lag] > QP[lag] for lag in range(df_classifier.shape[1])]).T

    print('Calculating spikes cutoffs...')
    spike_cutoffs = getSpikesCutoffs(df_data, p_cutoff, NumberOfRandomSamples=NumberOfRandomSamples)
    print(spike_cutoffs)

    df_data = normalizeSignalsToUnityDataframe(df_data)

    if not (df_data.index.values == df_classifier.index.values).all():
        raise ValueError('Index mismatch')

    print('Recording SpikeMax data...')
    max_spikes = df_data.index.values[getSpikes(df_data.values, np.max, spike_cutoffs)]
    print(len(max_spikes))
    significant_index_spike_max = [(gene in list(max_spikes)) for gene in df_data.index.values]
    lagSignigicantIndexSpikeMax = (np.sum(significant_index.T[1:],axis=0) == 0) * significant_index_spike_max
    write(df_classifier[lagSignigicantIndexSpikeMax], saveDir + dataName +'_selected%s_SpikeMax'%(info), hdf5fileName=hdf5fileName)
    write(df_data[lagSignigicantIndexSpikeMax], saveDir + dataName +'_selectedTimeSeries%s_SpikeMax'%(info), hdf5fileName=hdf5fileName)
            
    print('Recording SpikeMin data...')
    min_spikes = df_data.index.values[getSpikes(df_data.values, np.min, spike_cutoffs)]
    print(len(min_spikes))
    significant_index_spike_min = [(gene in list(min_spikes)) for gene in df_data.index.values]
    lagSignigicantIndexSpikeMin = (np.sum(significant_index.T[1:],axis=0) == 0) * (np.array(significant_index_spike_max) == 0) * significant_index_spike_min
    write(df_classifier[lagSignigicantIndexSpikeMin], saveDir + dataName +'_selected%s_SpikeMin'%(info), hdf5fileName=hdf5fileName)
    write(df_data[lagSignigicantIndexSpikeMin], saveDir + dataName +'_selectedTimeSeries%s_SpikeMin'%(info), hdf5fileName=hdf5fileName)

    print('Recording Lag%s-Lag%s data...'%(1,df_classifier.shape[1]))
    for lag in range(1,df_classifier.shape[1]):
        lagSignigicantIndex = (np.sum(significant_index.T[1:lag],axis=0) == 0) * (significant_index.T[lag])
        write(df_classifier[lagSignigicantIndex], saveDir + dataName +'_selected%s_LAG%s'%(info,lag), hdf5fileName=hdf5fileName)
        write(df_data[lagSignigicantIndex], saveDir + dataName +'_selectedTimeSeries%s_LAG%s'%(info,lag), hdf5fileName=hdf5fileName)
                
    return None


def visualizeTimeSeriesClassification(dataName, saveDir, numberOfLagsToDraw=3, hdf5fileName=None, exportClusteringObjects=False, writeClusteringObjectToBinaries=False, AutocorrNotPeriodogr=True, vectorImage=True):

    """Visualize time series classification.
    
    Args:
        dataName: data name
        saveDir: path of directories poining to data storage
        numberOfLagsToDraw: first top-N lags (or frequencies) to draw
        hdf5fileName: HDF5 storage path and name
        exportClusteringObjects: export clustering objects to xlsx files
        writeClusteringObjectToBinaries: export clustering objects to binary (pickle) files
        AutocorrNotPeriodogr: label to print on the plots

    Returns:
        None

    Usage:
        visualizeTimeSeriesClassification('myData_1', '/dir1/dir2/', AutocorrNotPeriodogr=True, writeClusteringObjectToBinaries=True)
    """

    info = 'Autocorrelations' if AutocorrNotPeriodogr else 'Periodograms'

    if hdf5fileName is None:
        hdf5fileName = saveDir + dataName + '.h5'

    def internalDraw(className):
        print('\n\n%s of Time Series:'%(className)) 
        df_data_selected = read(saveDir + dataName + '_selectedTimeSeries%s_%s'%(info,className), hdf5fileName=hdf5fileName)
        df_classifier_selected = read(saveDir + dataName + '_selected%s_%s'%(info,className), hdf5fileName=hdf5fileName)

        if (df_data_selected is None) or (df_classifier_selected is None):

            print('Selected %s time series not found in %s.'%(className, saveDir + dataName + '.h5'))
            print('Do time series classification first.')

            return 

        print('Creating clustering object.')
        clusteringObject = makeClusteringObject(df_data_selected, df_classifier_selected, significance='Elbow') #Silhouette

        print('Exporting clustering object.')
        if writeClusteringObjectToBinaries:
            write(clusteringObject, saveDir + 'consolidatedGroupsSubgroups/' + dataName + '_%s_%s'%(className,info) + '_GroupsSubgroups')
        
        if exportClusteringObjects:
            exportClusteringObject(clusteringObject, saveDir + 'consolidatedGroupsSubgroups/', dataName + '_%s_%s'%(className,info))

        print('Plotting Dendrogram with Heatmaps.')
        makeDendrogramHeatmap(clusteringObject, saveDir, dataName + '_%s_%sBased'%(className,info), AutocorrNotPeriodogr=AutocorrNotPeriodogr, vectorImage=vectorImage)

        return

    for lag in range(1,numberOfLagsToDraw + 1):
        internalDraw('LAG%s'%(lag))
            
    internalDraw('SpikeMax')
    internalDraw('SpikeMin')

    return None

###################################################################################################
