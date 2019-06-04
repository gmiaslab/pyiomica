import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib import cm

import numpy as np
import pandas as pd
import networkx as nx

import scipy
import scipy.signal
import scipy.stats
import scipy.cluster.hierarchy as hierarchy
from scipy.interpolate import UnivariateSpline

import sklearn
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE

import os
import pickle
import gzip
import copy
import multiprocessing

from pyiomica import ARI
from pyiomica import VisibilityGraph as vg

import urllib.request
import requests
import shutil

import pymysql
import datetime

print("MathIOmica >> PyIOmica (https://mathiomica.org), by G. Mias Lab")


### Utility functions #############################################################################
'''
Creates a path of directories, unless the path already exists.
'''
def createDirectories(path):

    if not os.path.exists(path):
        os.makedirs(path)

    return


'''
A handy way to parallelize a function call
'''
def runCPUs(NumberOfAvailableCPUs, func, list_of_tuples_of_func_params):

    instPool = multiprocessing.Pool(processes = NumberOfAvailableCPUs)
    return_values = instPool.map(func, list_of_tuples_of_func_params)
    instPool.close()
    instPool.join()

    return np.vstack(return_values)
  

'''
Pickle object into a file
'''
def write(data, fileName, withPKLZextension = True):

    with gzip.open(fileName + ('.pklz' if withPKLZextension else ''),'wb') as temp_file:
        pickle.dump(data, temp_file, protocol=4)

    return


'''
Unpickle object from a file
'''
def read(fileName, withPKLZextension = True):

    with gzip.open(fileName + ('.pklz' if withPKLZextension else ''),'rb') as temp_file:
        data = pickle.load(temp_file)
        return data

    return


'''
Efficient way to create a reverse dictionary from a dictionary.
Utilizes Pandas.Dataframe.groupby and fast Numpy arrays indexing
Note: any entries with missing values will be removed
'''
def createReverseDictionary(inputDictionary):

    keys, values = np.array(list(inputDictionary.keys())), np.array(list(inputDictionary.values()))
    df = pd.DataFrame(np.array([[keys[i], value] for i in range(len(keys)) for value in values[i]]))
    dfGrouped = df.groupby(df.columns[1])
    keys, values = list(dfGrouped.indices.keys()), list(dfGrouped.indices.values())
    GOs = df.values.T[0]

    return dict(zip(keys, [GOs[value] for value in values]))

###################################################################################################



### Global constants ##############################################################################
'''
ConstantGeneDictionary is a global gene/protein dictionary variable typically created by GetGeneDictionary.
'''
ConstantGeneDictionary = None

'''
ConstantMathIOmicaDataDirectory is a global variable pointing to the MathIOmica data directory.
'''
ConstantMathIOmicaDataDirectory = "\\".join([os.getcwd(), "Applications",  "MathIOmica", "MathIOmicaData"])

'''
ConstantMathIOmicaExamplesDirectory is a global variable pointing to the MathIOmica example data directory.
'''
ConstantMathIOmicaExamplesDirectory = "\\".join([os.getcwd(), "Applications",  "MathIOmica", "MathIOmicaData", "ExampleData"])

'''
ConstantMathIOmicaExampleVideosDirectory is a global variable pointing to the MathIOmica example videos directory.
'''
ConstantMathIOmicaExampleVideosDirectory = "\\".join([os.getcwd(), "Applications",  "MathIOmica", "MathIOmicaData", "ExampleVideos"])

for path in [ConstantMathIOmicaDataDirectory, ConstantMathIOmicaExamplesDirectory, ConstantMathIOmicaExampleVideosDirectory]:
    createDirectories(path)

###################################################################################################



### Annotations and Enumerations ##################################################################

def ReactomeAnalysis(inputData):

    uploadURL = "https://reactome.org/AnalysisService/identifiers/projection?interactors=false&pageSize=20&page=1&sortBy=ENTITIES_PVALUE&order=ASC&resource=TOTAL"

    def PARAMS(item):

        return {"Method": "POST", "Headers": {"accept": "application/json", "content-type": "text/plain"}, "Body": str(item)}

    # POST with JSON 
    import json
    r = requests.post(uploadURL, data=json.dumps({"Method": "POST", "Headers": {"accept": "application/json", "content-type": "text/plain"}, "Body": str('#GBM Uniprot\nP01023\nQ99758\nO15439\nO43184')}))

    # Response, status etc
    r.text
    r.status_code

    #data = {'api_dev_key':API_KEY, 
    #    'api_option':'paste', 
    #    'api_paste_code':source_code, 
    #    'api_paste_format':'python'}
    
    #queryReactome = requests.post(url = uploadURL, params = PARAMS(inputData[0])) 

    #temp = queryReactome.text
    
    #queryReactome = Query[All, All, All /* (URLExecute[
    #        HTTPRequest["https://reactome.org/AnalysisService/identifiers/projection?interactors=false&pageSize=20&page=1&sortBy=ENTITIES_PVALUE&order=ASC&resource=TOTAL", 
    #        PARAMS(item)], "RawJSON"] &)]@inputData


    ##downloadURL = "https://reactome.org/AnalysisService/identifiers/projection?interactors=false&pageSize=20&page=1&sortBy=ENTITIES_PVALUE&order=ASC&resource=TOTAL"
    enrichmentReturn = None
    #enrichmentReturn = Query[All, All, "summary", "token" /* (If[MissingQ[#], <|"Missing"|>, URLExecute[
    #        HTTPRequest["https://reactome.org/AnalysisService/download/" <> # <> "/pathways/TOTAL/result.csv" , 
    #        <|"Method" -> "GET", "Headers" -> {"accept" -> "application/json", "content-type" ->   "text/plain"}|>], "CSV"]] &)]@ queryReactome

    return enrichmentReturn










'''
#Analysis for Multi-Omics or Single-Omics input list
'''
def internalAnalysisFunction(data, multiCorr, MultipleList,  OutputID, InputID, Species, totalMembers,
                            pValueCutoff, ReportFilterFunction, ReportFilter, TestFunction, HypothesisFunction, AdditionalFilter,
                            AssignmentForwardDictionary, AssignmentReverseDictionary, prefix, infoDict):
    
    listData = data[list(data.keys())[0]]

    #If input data was a list of genes, convert it to pairs with label "Generic"
    if not type(listData[0]) is list:

        listData = [[item, 'Generic'] for item in listData]

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
    whatIsFilteredSignif = np.array([item[0][2] for item in list(ResultsHCct.values())]).astype(bool)
    whatIsFiltered = whatIsFilteredLength * whatIsFilteredSignif

    returning = dict(zip(list(np.array(list(ResultsHCct.keys()))[whatIsFiltered]),list(np.array(list(ResultsHCct.values()))[whatIsFiltered])))

    if AdditionalFilter!=None: 
        print("AdditionalFilter option is not yet available...")

    return {list(data.keys())[0]: returning}


'''
OBOGODictionary() is an Open Biomedical Ontologies (OBO) Gene Ontology (GO) vocabulary dictionary generator.
returns: Dictionary
'''
def OBOGODictionary(FileURL="http://purl.obolibrary.org/obo/go/go-basic.obo", ImportDirectly=False, MathIOmicaDataDirectory=None, OBOFile="goBasicObo.txt"):

    global ConstantMathIOmicaDataDirectory

    MathIOmicaDataDirectory = ConstantMathIOmicaDataDirectory if MathIOmicaDataDirectory==None else MathIOmicaDataDirectory
    fileGOOBO = "\\".join([MathIOmicaDataDirectory, OBOFile])

    #import the GO OBO file: we check if the OBO file Exist, if not, attempt to download and create it
    if not os.path.isfile(fileGOOBO):
        print("Did Not Find Annotation Files, Attempting to Download...")
        ImportDirectly = True

    if ImportDirectly:
        if os.path.isfile(fileGOOBO):
            os.remove(fileGOOBO)

        urllib.request.urlretrieve(FileURL.strip('"'), fileGOOBO)

        if os.path.isfile(fileGOOBO):
            print("Created Annotation Files at ", fileGOOBO)
        else:
            print("Did Not Find Annotation Files, Aborting Process")
            return

    with open(fileGOOBO, 'r') as tempFile:
        inputFile = tempFile.readlines()

    #Find keys "accessions (id):" and "name:" and "namespace" but extract their corresponding values in a list and map them to their corresponding [Term] positions, 
    #Once the "accessions (id):" and its corresponding "name:" in a list, make an association between them,
    #so you can search this association using the key "accessions (id):" to get the value "name:" and "namespace"
    outDictionary = {}

    for position in np.where([item=='[Term]\n'for item in inputFile])[0]:
        
        def getValue(index):

            return inputFile[position + index].strip(['id:', 'name:', 'namespace:'][index - 1]).strip('\n').strip()

        outDictionary[getValue(1)] = [getValue(2), getValue(3)]
    
    return outDictionary


'''
GetGeneDictionary() creates an ID/accession dictionary from a UCSC search - typically of gene annotations.
'''
def GetGeneDictionary(geneUCSCTable = None, UCSCSQLString = None, UCSCSQLSelectLabels = None,
                    ImportDirectly = False, JavaGBs = '8', Species = "human", KEGGUCSCSplit = [True,"KEGG Gene ID"]):
       
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

    global ConstantMathIOmicaDataDirectory

    MathIOmicaDataDirectory = ConstantMathIOmicaDataDirectory

    geneUCSCTable = "\\".join([MathIOmicaDataDirectory, Species + "GeneUCSCTable"])
       
    #If the user asked us to import directly, import directly with SQL, otherwise, get it from a directory they specify
    if not os.path.isfile(geneUCSCTable):
        print("Did Not Find Gene Translation Files, Attempting to Download from UCSC...")
        ImportDirectly = True
    else:
        termTable = read(geneUCSCTable, False)[1]

    if ImportDirectly:
        #Connect to the database from UCSC
        ucscDatabase = pymysql.connect("genome-mysql.cse.ucsc.edu","genomep","password")

        if ucscDatabase==None:
           print("Could not establish connection to UCSC. Please try again or add MathIOmica's dictionary manually at ", geneUCSCTable)
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
        write((datetime.datetime.now().isoformat(), termTable), geneUCSCTable, False)

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


'''
Function will download and create gene associations and restrict to required background set
'''
def GOAnalysisAssigner(MathIOmicaDataDirectory = None, ImportDirectly = False, BackgroundSet = [], Species = "human",
                        LengthFilter = None, LengthFilterFunction = np.greater_equal, GOFileName = None, GOFileColumns = [2, 5], 
                        GOURL = "http://current.geneontology.org/annotations/"):

    global ConstantMathIOmicaDataDirectory

    MathIOmicaDataDirectory = ConstantMathIOmicaDataDirectory if MathIOmicaDataDirectory==None else MathIOmicaDataDirectory

    #If the user asked us to import MathIOmicaDataDirectoryectly, import MathIOmicaDataDirectoryectly from GO website, otherwise, get it from a MathIOmicaDataDirectoryectory they specify
    file = "goa_" + Species + ".gaf.gz" if GOFileName==None else GOFileName
    localFile =  "\\".join([MathIOmicaDataDirectory, "goa_" + Species + ".gaf"])
    localZipFile =  "\\".join([MathIOmicaDataDirectory, "goa_" + Species + ".gaf.gz"])
    fileGOAssociations = ["\\".join([MathIOmicaDataDirectory, Species + item]) for item in ["GeneOntAssoc", "IdentifierAssoc"]]

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
        geneOntAssoc = dict(zip(keys, [np.unique(IDs[value]) for value in values]))

        identifierAssoc = createReverseDictionary(geneOntAssoc)

        #Save created annotations geneOntAssoc, identifierAssoc
        write((datetime.datetime.now().isoformat(), geneOntAssoc), fileGOAssociations[0], False)
        write((datetime.datetime.now().isoformat(), identifierAssoc), fileGOAssociations[1], False)
        
        os.remove(localFile)

        if np.array(list(map(os.path.isfile, fileGOAssociations))).all():
            print("Created Annotation Files at ", fileGOAssociations)
        else:
            print("Did Not Find Annotation Files, Aborting Process")
            return
    else:
        #Otherwise we get from the user specified MathIOmicaDataDirectoryectory
        geneOntAssoc = read(fileGOAssociations[0], False)[-1]
        identifierAssoc = read(fileGOAssociations[1], False)[-1]

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


'''
Obtain gene dictionary - if it exists can either augment with new information or Species or create new, if not exist then create variable
'''
def obtainConstantGeneDictionary(GeneDictionary, GetGeneDictionaryOptions, AugmentDictionary):

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

    return


'''
GOAnalysis calculates input data over-representation analysis for Gene Ontology (GO) categories.
MultipleListCorrection==None, #Correct for multiple lists, e.g protein+RNA
MultipleList==False, #whether input is multiple omics or single - for non-omics-object inputs
AdditionalFilter==None, #Select[MatchQ[#[[3,1,2]],"biological_process"]&]
'''
def GOAnalysis(data, GetGeneDictionaryOptions={}, AugmentDictionary=True, InputID=["UniProt ID","Gene Symbol"], OutputID="UniProt ID",
                 GOAnalysisAssignerOptions={}, BackgroundSet="All", Species="human", OntologyLengthFilter=2, ReportFilter=1, ReportFilterFunction=np.greater_equal,
                 pValueCutoff=0.05, TestFunction=lambda n, N, M, x: 1. - scipy.stats.hypergeom.cdf(x-1, M, n, N), 
                 HypothesisFunction=lambda data, SignificanceLevel: BenjaminiHochbergFDR(data, SignificanceLevel=SignificanceLevel)["Results"], 
                 FilterSignificant=True, OBODictionaryVariable=None,
                 OBOGODictionaryOptions={}, MultipleListCorrection=None, MultipleList=False, AdditionalFilter=None, GeneDictionary=None):

    global ConstantGeneDictionary

    #Obtain OBO dictionary with OBOGODictionaryOptions if any. If externally defined use user definition for OBODict Variable
    OBODict = OBOGODictionary(**OBOGODictionaryOptions) if OBODictionaryVariable==None else OBODictionaryVariable

    #Obtain gene dictionary - if it exists can either augment with new information or Species or create new, if not exist then create variable
    obtainConstantGeneDictionary(GeneDictionary, GetGeneDictionaryOptions, AugmentDictionary)
    
    #Get the right GO terms for the BackgroundSet requested and correct Species
    Assignment = GOAnalysisAssigner(BackgroundSet=[], Species=Species , LengthFilter=OntologyLengthFilter) if GOAnalysisAssignerOptions=={} else GOAnalysisAssigner(**GOAnalysisAssignerOptions)
    
    #If the input is simply a list
    listToggle = True if type(data) is list else False
    data = {'dummy': data} if listToggle else data

    #Check if a clustering object
    if "linkage" in data.keys():
        if MultipleListCorrection==None:
            multiCorr = 1
        elif MultipleListCorrection=='Automatic':
            multiCorr = 1
            for keyGroup in sorted([item for item in list(data.keys()) if not item=='linkage']):
                dataClass = data[keyGroup]
                for keySubGroup in sorted([item for item in list(data[keyGroup].keys()) if not item=='linkage']):
                    multiCorr = max(max(np.unique(data[keyGroup][keySubGroup]['data'].index.get_level_values('gene'), return_counts=True)[1]), multiCorr)
        else:
            multiCorr = MultipleListCorrection

        #Loop through the clustering object, calculate GO for each SubGroup
        returning = {}
        for keyGroup in sorted([item for item in list(data.keys()) if not item=='linkage']):
            returning[keyGroup] = {}
            for keySubGroup in sorted([item for item in list(data[keyGroup].keys()) if not item=='linkage']):
                SubGroupMultiIndex = data[keyGroup][keySubGroup]['data'].index
                SubGroupGenes = list(SubGroupMultiIndex.get_level_values('gene'))
                SubGroupMeta = list(SubGroupMultiIndex.get_level_values('source'))
                SubGroupData = [[SubGroupGenes[i], SubGroupMeta[i]] for i in range(len(SubGroupMultiIndex))]

                returning[keyGroup][keySubGroup] = internalAnalysisFunction({keySubGroup:SubGroupData},
                                                                     multiCorr, MultipleList,  OutputID, InputID, Species, len(Assignment[Species]["IDToGO"].keys()),
                                                                     pValueCutoff, ReportFilterFunction, ReportFilter, TestFunction, HypothesisFunction, AdditionalFilter,
                                                                     AssignmentForwardDictionary=Assignment[Species]['IDToGO'],
                                                                     AssignmentReverseDictionary=Assignment[Species]['GOToID'],
                                                                     prefix='', infoDict=OBODict)[keySubGroup]
    #The data is a dictionary of type {'Name': [data]}
    else:
        if MultipleListCorrection==None:
            multiCorr = 1
        elif MultipleList and MultipleListCorrection=='Automatic':
            multiCorr = max(np.unique([item[0] for item in data[list(data.keys())[0]]], return_counts=True)[1])
        else:
            multiCorr = MultipleListCorrection

        returning = internalAnalysisFunction(data, multiCorr, MultipleList,  OutputID, InputID, Species, len(Assignment[Species]["IDToGO"].keys()),
                                            pValueCutoff, ReportFilterFunction, ReportFilter, TestFunction, HypothesisFunction, AdditionalFilter,
                                            AssignmentForwardDictionary=Assignment[Species]['IDToGO'],
                                            AssignmentReverseDictionary=Assignment[Species]['GOToID'],
                                            prefix='', infoDict=OBODict)

        #If a single list was provided, return the association for Gene Ontologies
        returning = returning[list(data.keys())[0]] if listToggle else returning

    return returning


'''
GeneTranslation(inputList,targetIDList,geneDictionary) uses geneDictionary to convert inputList IDs to different annotations as indicated by targetIDList.
NOTE: There is a problem with this function!!! -> returning shape/form needs revision
'''
def GeneTranslation(InputList, TargetIDList, GeneDictionary, InputID = None, Species = "human"):

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
                allEntries = GeneDictionary[Species][TargetID][np.where(GeneDictionary[Species][key]==item)[0]]
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


'''
KEGGAnalysisAssigner() creates KEGG: Kyoto Encyclopedia of Genes and Genomes pathway associations, 
restricted to required background set, downloading the data if necessary.
'''
def KEGGAnalysisAssigner(MathIOmicaDataDirectory = None, ImportDirectly = False, BackgroundSet = [], KEGGQuery1 = "pathway", KEGGQuery2 = "hsa",
                        LengthFilter = None, LengthFilterFunction = np.greater_equal, Labels = ["IDToPath", "PathToID"]):
    
    global ConstantMathIOmicaDataDirectory
    
    MathIOmicaDataDirectory = ConstantMathIOmicaDataDirectory if MathIOmicaDataDirectory==None else MathIOmicaDataDirectory

    ##if the user asked us to import directly, import directly from KEGG website, otherwise, get it from a directory they specify
    fileAssociations = ["\\".join([MathIOmicaDataDirectory, item]) for item in [KEGGQuery1 + "_" + KEGGQuery2 + "KEGGMemberToPathAssociation", KEGGQuery1 + "_" + KEGGQuery2 + "KEGGPathToMemberAssociation"]]

    if not np.array(list(map(os.path.isfile, fileAssociations))).all():
        print("Did Not Find Annotation Files, Attempting to Download...")
        ImportDirectly = True

    if ImportDirectly:
        localFile = "\\".join([MathIOmicaDataDirectory, KEGGQuery1 + "_" + KEGGQuery2 + ".tsv"])
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
        pathToID = dict(zip(keys, [np.unique(IDs[value]) for value in values]))
        idToPath = createReverseDictionary(pathToID)

        write((datetime.datetime.now().isoformat(), idToPath), fileAssociations[0], False)
        write((datetime.datetime.now().isoformat(), pathToID), fileAssociations[1], False)

        os.remove(localFile)

        if np.array(list(map(os.path.isfile, fileAssociations))).all():
            print("Created Annotation Files at ", fileAssociations)
        else:
            print("Did Not Find Annotation Files, Aborting Process")
            return
    else:
        #otherwise import the necessary associations from MathIOmicaDataDirectoryectory
        idToPath = read(fileAssociations[0], False)[1]
        pathToID = read(fileAssociations[1], False)[1]

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


'''
KEGGDictionary() creates a dictionary from KEGG: Kyoto Encyclopedia of Genes and Genomes terms - typically association of pathways and members therein.
'''
def KEGGDictionary(MathIOmicaDataDirectory = None, ImportDirectly = False, KEGGQuery1 = "pathway", KEGGQuery2 = "hsa"):
    
    global ConstantMathIOmicaDataDirectory
    
    MathIOmicaDataDirectory = ConstantMathIOmicaDataDirectory if MathIOmicaDataDirectory==None else MathIOmicaDataDirectory

    #if the user asked us to import directly, import directly from KEGG website, otherwise, get it from a directory they specify
    fileKEGGDict = "\\".join([MathIOmicaDataDirectory, KEGGQuery1 + "_" + KEGGQuery2 + "_KEGGDictionary_Py"])

    if os.path.isfile(fileKEGGDict):
        associationKEGG = read(fileKEGGDict, False)[1]
    else:
        print("Did Not Find Annotation Files, Attempting to Download...")
        ImportDirectly = True

    if ImportDirectly:
        queryFile = "\\".join([MathIOmicaDataDirectory, KEGGQuery1 + "_" + KEGGQuery2 + ".tsv"])

        if os.path.isfile(queryFile): 
           os.remove(queryFile)

        urllib.request.urlretrieve("http://rest.kegg.jp/list/" + KEGGQuery1 + ("" if KEGGQuery2=="" else "/" + KEGGQuery2), queryFile)

        with open(queryFile, 'r') as tempFile:
            tempLines = tempFile.readlines()
            
        os.remove(queryFile)
        
        associationKEGG = dict([line.strip('\n').split('\t') for line in tempLines])

        write((datetime.datetime.now().isoformat(), associationKEGG), fileKEGGDict, False)

        if os.path.isfile(fileKEGGDict):
            print("Created Annotation Files at ", fileKEGGDict)
        else:
            print("Did Not Find Annotation Files, Aborting Process")
            return

    return associationKEGG


'''
KEGGAnalysis(data) calculates input data over-representation analysis for KEGG: Kyoto Encyclopedia of Genes and Genomes pathways.
Input can be: clustering object
MultipleListCorrection -> None, (*Correct for multiple lists, e.g protein+RNA*)
MultipleList -> False, (*whether input is multiple omics or single - for non-omics-object inputs*)
AdditionalFilter -> None (*Select[MatchQ[#[[3,1,2]],"biological_process"]&]*),
AnalysisType -> "Genomic" (*options are "Genome", "Molecular","All"*),
Species -> "human", (*Used in GeneDictionary*)
'''
def KEGGAnalysis(data, AnalysisType = "Genomic", GetGeneDictionaryOptions = {}, AugmentDictionary = True, InputID = ["UniProt ID", "Gene Symbol"],
                OutputID = "KEGG Gene ID", MolecularInputID = ["cpd"], MolecularOutputID = "cpd", KEGGAnalysisAssignerOptions = {}, BackgroundSet = [], 
                KEGGOrganism = "hsa", KEGGMolecular = "cpd", KEGGDatabase = "pathway", PathwayLengthFilter = 2, ReportFilter = 1, 
                ReportFilterFunction = np.greater_equal, pValueCutoff = 0.05, TestFunction = lambda n, N, M, x: 1. - scipy.stats.hypergeom.cdf(x-1, M, n, N), 
                HypothesisFunction = lambda data, SignificanceLevel: BenjaminiHochbergFDR(data, SignificanceLevel=SignificanceLevel)["Results"],
                FilterSignificant = True, KEGGDictionaryVariable = None, KEGGDictionaryOptions = {}, MultipleListCorrection = None, MultipleList = False, AdditionalFilter = None, 
                GeneDictionary = None, Species = "human", MolecularSpecies = "compound", NonUCSC = False, MathIOmicaDataDirectory = None):

    argsLocal = locals().copy()

    global ConstantMathIOmicaDataDirectory

    obtainConstantGeneDictionary(None, {}, True)
    
    MathIOmicaDataDirectory = ConstantMathIOmicaDataDirectory if MathIOmicaDataDirectory==None else MathIOmicaDataDirectory

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
        fileMolDict = "\\".join([MathIOmicaDataDirectory, "MathIOmicaMolecularDictionary_Py"])

        if os.path.isfile(fileMolDict):
            GeneDictionary = read(fileMolDict, False)[1]
        else:
            fileCSV = "\\".join([MathIOmicaDataDirectory, "MathIOmicaMolecularDictionary.csv"])

            print('Attempting to read:', fileCSV)

            if os.path.isfile(fileCSV):
                with open(fileCSV, 'r') as tempFile:
                    tempLines = tempFile.readlines()
            
                tempData = np.array([line.strip('\n').replace('"', '').split(',') for line in tempLines]).T
                tempData = {'compound': {'pumchem': tempData[0], 'cpd': tempData[1]}}
                write((datetime.datetime.now().isoformat(), tempData), fileMolDict, False)
            else:
                print("Could not find annotation file at " + fileMolDict + " Please either obtain an annotation file from mathiomica.org or provide a GeneDictionary option variable.")
                return

            GeneDictionary = read(fileMolDict, False)[1]

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

    #If the input is simply a list
    listToggle = True if type(data) is list else False
    data = {'dummy': data} if listToggle else data

    #Check if a clustering object
    if "linkage" in data.keys():
        if MultipleListCorrection==None:
            multiCorr = 1
        elif MultipleListCorrection=='Automatic':
            multiCorr = 1
            for keyGroup in sorted([item for item in list(data.keys()) if not item=='linkage']):
                dataClass = data[keyGroup]
                for keySubGroup in sorted([item for item in list(data[keyGroup].keys()) if not item=='linkage']):
                    multiCorr = max(max(np.unique(data[keyGroup][keySubGroup]['data'].index.get_level_values('gene'), return_counts=True)[1]), multiCorr)
        else:
            multiCorr = MultipleListCorrection

        #Loop through the clustering object, calculate GO for each SubGroup
        returning = {}
        for keyGroup in sorted([item for item in list(data.keys()) if not item=='linkage']):
            returning[keyGroup] = {}
            for keySubGroup in sorted([item for item in list(data[keyGroup].keys()) if not item=='linkage']):
                SubGroupMultiIndex = data[keyGroup][keySubGroup]['data'].index
                SubGroupGenes = list(SubGroupMultiIndex.get_level_values('gene'))
                SubGroupMeta = list(SubGroupMultiIndex.get_level_values('source'))
                SubGroupData = [[SubGroupGenes[i], SubGroupMeta[i]] for i in range(len(SubGroupMultiIndex))]

                returning[keyGroup][keySubGroup] = internalAnalysisFunction({keySubGroup:SubGroupData},
                                                                             multiCorr, MultipleList,  OutputID, InputID, Species, len(Assignment[KEGGOrganism]["IDToPath"]),
                                                                             pValueCutoff, ReportFilterFunction, ReportFilter, TestFunction, HypothesisFunction, AdditionalFilter,
                                                                             AssignmentForwardDictionary=Assignment[KEGGOrganism]['IDToPath'],
                                                                             AssignmentReverseDictionary=Assignment[KEGGOrganism]['PathToID'],
                                                                             prefix='hsa:' if AnalysisType=='Genomic' else '', infoDict=keggDict)[keySubGroup]
    #The data is a dictionary of type {'Name': [data]}
    else:
        if MultipleListCorrection==None:
            multiCorr = 1
        elif MultipleList and MultipleListCorrection=='Automatic':
            multiCorr = max(np.unique([item[0] for item in data[list(data.keys())[0]]], return_counts=True)[1])
        else:
            multiCorr = MultipleListCorrection

        returning = internalAnalysisFunction(data, multiCorr, MultipleList,  OutputID, InputID, Species, len(Assignment[KEGGOrganism]["IDToPath"]),
                                            pValueCutoff, ReportFilterFunction, ReportFilter, TestFunction, HypothesisFunction, AdditionalFilter,
                                            AssignmentForwardDictionary=Assignment[KEGGOrganism]['IDToPath'],
                                            AssignmentReverseDictionary=Assignment[KEGGOrganism]['PathToID'],
                                            prefix='hsa:' if AnalysisType=='Genomic' else '', infoDict=keggDict)

        #If a single list was provided
        returning = returning[list(data.keys())[0]] if listToggle else returning

    return returning


'''
MassMatcher(data, accuracy) assigns putative mass identification to input data based on monoisotopic mass 
(using MathIOmica's mass dictionary), using the accuracy in parts per million.    
'''
def MassMatcher(data, accuracy, MassDictionaryVariable = None, MolecularSpecies = "cpd"):
    
    ppm = accuracy*(10**-6)

    MassDict = MassDictionary() if MassDictionaryVariable==None else MassDictionaryVariable
    keys, values = np.array(list(MassDict[MolecularSpecies].keys())), np.array(list(MassDict[MolecularSpecies].values()))

    return keys[np.where((values > data*(1 - ppm)) * (values < data*(1 + ppm)))[0]]


'''
MassDictionary() loads PyIOmica's current mass dictionary
'''
def MassDictionary(MathIOmicaDataDirectory=None):

    global ConstantMathIOmicaDataDirectory

    MathIOmicaDataDirectory = ConstantMathIOmicaDataDirectory if MathIOmicaDataDirectory==None else MathIOmicaDataDirectory

    fileMassDict = "\\".join([MathIOmicaDataDirectory, "MathIOmicaMassDictionary" +  ".csv"])

    if os.path.isfile(fileMassDict):
        with open(fileMassDict, 'r') as tempFile:
            fileMassDictData = tempFile.readlines()
            fileMassDictData = np.array([item.strip('\n').replace('"','').split(",") for item in fileMassDictData])

        MassDict = {fileMassDictData[0][0].split(':')[0]: dict(zip(fileMassDictData.T[0],fileMassDictData.T[1].astype(float)))}
    else:
        print("Could not find MathIOmica's mass dictionary at ", fileMassDict, 
                "Please either obtain a mass dictionary file from mathiomica.org or provide a custom file at the above location.")

    return MassDict


'''
OmicsObjectUniqueMassConverter(omicsObject, massAccuracy) assigns a unique putative mass 
identification to each of omicsObject's inner association keys, using the massAccuracy in parts per million.
'''
def OmicsObjectUniqueMassConverter(omicsObject, massAccuracy, MassMatcherOptions = {}): 

    #keyMapper = Association@(( #[[2]] -> If[MatchQ[#[[1]], {}], #[[2]], If[Length[#[[1]]] == 1, Flatten[{#[[1]], #[[2]]}], #[[2]]]]) & /@ (({MassMatcher[#[[1]], massAccuracy, Sequence @@ MassMatcherOptions], #} &) /@ Query[1, Keys]@omicsObject));
   
    #returning = Query[All, KeyMap[keyMapper]]@omicsObject;
   
    return returning


'''
EnrichmentReportExport(results) exports results from enrichment analyses to Excel spreadsheets.
'''
def ExportEnrichmentReport(data, AppendString="", OutputDirectory=None):

    saveDir = "\\".join([os.getcwd(), "Enrichment reports"]) + "\\"

    createDirectories(saveDir)

    if AppendString=="":
        AppendString=(datetime.datetime.now().isoformat().replace(' ', '_').replace(':', '_').split('.')[0])

    def ExportToFile(fileName, data):

        writer = pd.ExcelWriter(fileName)

        for key in list(data.keys()):
            keys, values = list(data[key].keys()), list(data[key].values())

            listNum = [[item for sublist in list(value)[:2] for item in sublist] for value in values]
            listNon = [list(value)[2:] for value in values]

            dataDF = [listNum[i] + listNon[i] for i in range(len(keys))]
            columns = ['p-Value', 'BH-corrected p-Value', 'Significant', 'Counts 1', 'Counts 2', 'Counts 3', 'Counts 4', 'Description', 'List']

            pd.DataFrame(data=dataDF, index=keys, columns=columns).to_excel(writer, str(key))

        writer.save()

        print('Saved:' + fileName)

        return None

    if (type(data) is dict):
        idata = data[list(data.keys())[0]]
        if not type(idata) is dict:
            data = {"List": {"Results":data}}
            keyClass = list(data.keys())[0]
            ExportToFile(saveDir + str(keyClass) + '_' + AppendString + '.xlsx', data[keyClass])
        elif type(idata) is dict:
            idata = idata[list(idata.keys())[0]]
            if not type(idata) is dict:
                keyClass = list(data.keys())[0]
                data = {keyClass: {"Results":data[keyClass]}}
                ExportToFile(saveDir + str(keyClass) + '_' + AppendString + '.xlsx', data[keyClass])
            elif type(idata) is dict:
                #Loop through the clustering object, export each class to separate file
                for keyClass in list(data.keys()):
                    ExportToFile(saveDir + str(keyClass) + '_' + AppendString + '.xlsx', data[keyClass])
    else:
        print('Results type is not supported...')

    return None

###################################################################################################



### Core functions ################################################################################
'''
Equivalent of Mathematica.Chop Function
expr: a number or a pyhton sequence of numbers
tolerance such as default in Mathematica
'''
def chop(expr, tolerance=1e-10):
        
    if isinstance(expr, (list, tuple, np.ndarray)):

        expr_copy = np.copy(expr)
        expr_copy[np.abs(expr) < tolerance] = 0

    else:
        expr_copy = 0 if expr < tolerance else expr

    return expr_copy


'''
Calculates modified z-score of a 1D array based on "Median absolute deviation"
Warning: use on 1-D arrays only!
'''
def modifiedZScore(subset):

    '''
    1D, 2D Median absolute deviation of a sequence of numbers or pd.Series
    Default axis=None: multidimentional arrays are flattened
    axis=0: use if data in columns
    axis=1: use if data in rows
    '''
    def medianAbsoluteDeviation(expr, axis=None):

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
        
        return

    is_not_nan = np.abs(np.isnan(subset.values) * 1.0 - 1.0) > 0.0

    values = subset[is_not_nan].values

    MedianAD = medianAbsoluteDeviation(values, axis=None)

    if MedianAD == 0.:
        MeanAD = np.sum(np.abs(values - np.mean(values))) / len(values)
        print('MeanAD:', MeanAD, '\tMedian:', np.median(values))
        coefficient = 0.7978846 / MeanAD
    else:
        print('MedianAD:', MedianAD, '\tMedian:', np.median(values))
        coefficient = 0.6744897 / MedianAD
        
    subset.iloc[is_not_nan] = coefficient * (values - np.median(values))

    return subset


'''
Power transform from scipy.stats
subset: 1D numpy array
'''
def boxCoxTransform(subset, lmbda=None, giveLmbda=False):

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


'''
Lomb-Scargle core translated from MathIOmica.m
Used to calculate the different frequency components of our spectrum: project the cosine/sine component and normalize it:
func: Sin or Cos
freq: frequencies (1D array of floats)
times: input times (starting point adjusted w.r.t.dataset times), Zero-padded (is it???)
data: input Data with the mean subtracted from it, before zero-padding (for sure???)
'''
def ampSquaredNormed(func, freq, times, data):

    omega_freq = 2. * (np.pi) * freq
    theta_freq = 0.5 * np.arctan2(np.sum(np.sin(4. * (np.pi) * freq * times)), np.sum(np.cos(4. * (np.pi) * freq * times) + 10 ** -20))
    
    ampSum = np.sum(data * func(omega_freq * times - theta_freq)) ** 2
    ampNorm = np.sum(func(omega_freq * times - theta_freq) ** 2)

    return chop(ampSum) / ampNorm


'''
Autocorrelation from MathIOmica.m
inputTimes: times corresponding to provided data points (1D array of floats)
inputData: data points (1D array of floats)
inputSetTimes: a complete set of all possible N times during which data could have been collected
'''
def autocorrelation(inputTimes, inputData, inputSetTimes, UpperFrequencyFactor=1):
    
    #InverseAutocovariance from MathIOmica.m
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


'''
Wrapper of Autocorrelation function for use with Multiprocessing
'''
def pAutocorrelation(args):

    inputTimes, inputData, inputSetTimes = args
    
    return autocorrelation(inputTimes, inputData, inputSetTimes)


'''
inputData: data points (2D array of floats), rows are normalized signals
func: np.max or np.min
'''
def getSpikes(inputData, func, cutoffs):

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


'''
df_data:
p_cutoff:
NumberOfRandomSamples:
'''
def getSpikesCutoffs(df_data, p_cutoff, NumberOfRandomSamples=10**3):

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


'''
Lomb-Scargle Periodogram from MathIOmica.m
inputTimes: times corresponding to provided data points (1D array of floats)
inputData: data points (1D array of floats)
inputSetTimes: a complete set of all possible N times during which data could have been collected

TO DO: debug all optional parameters, such as FrequenciesOnly, NormalizeIntensities, etc.
'''
def LombScargle(inputTimes, inputData, inputSetTimes, FrequenciesOnly=False,NormalizeIntensities=False,OversamplingRate=1,PairReturn=False,UpperFrequencyFactor=1):

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
    #f0 0- effectively a lowpass filter) and the upper cutoff Nyquist by the
    #upper factor specified
    freq = np.linspace(f0, n / 2 * UpperFrequencyFactor * f0, f0 * (n / 2 * UpperFrequencyFactor) / freqStep)

    if FrequenciesOnly:
        #return Association@ MapIndexed[("f" <> ToString[Sequence @@ #2] ->
        ##1)&]@freq]
        return freq

    #get the periodogram
    periodogram = 1.0 / (2.0 * varianceInputPoints) * np.array(tuple(map(lambda f: chop(ampSquaredNormed(np.cos, f, inputTimesNormed, inputDataCentered)) + chop(ampSquaredNormed(np.sin, f, inputTimesNormed, inputDataCentered)), list(freq))))
    
    #the function finally returns:1) the list of frequencies, 2) the
    #corresponding list of Lomb-Scargle spectra
    if NormalizeIntensities:
        periodogram = periodogram / np.sqrt(np.dot(periodogram,periodogram))

    returning = np.vstack((freq, periodogram))

    if PairReturn:
        returning = np.ranspose(returning)

    return returning


'''
Calculate autocorrelation using Lomb-Scargle Autocorrelation
NOTE: there should be already no missing or non-numeric points in the input Series or Dataframe
df_data: pandas Series or Dataframe
setAllInputTimes: a complete set of all possible N times during which data could have been collected
'''
def getAutocorrelationsOfData(params):

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


'''
Generate autocorrelation null-distribution from permutated data
Calculate autocorrelation using Lomb-Scargle Autocorrelation
NOTE: there should be already no missing or non-numeric points in the input Series or Dataframe
df_data: pandas Series or Dataframe
NumberOfRandomSamples: size of the distribution to generate
'''
def getRandomAutocorrelations(df_data, NumberOfRandomSamples=10**5, NumberOfCPUs=4):

    data = np.vstack([np.random.choice(df_data.values[:,i], size=NumberOfRandomSamples, replace=True) for i in range(len(df_data.columns.values))]).T

    df_data_random = pd.DataFrame(data=data, index=range(NumberOfRandomSamples), columns=df_data.columns)
    df_data_random = filterOutFractionZeroSignalsDataframe(df_data_random, 0.75)
    df_data_random = normalizeSignalsToUnityDataframe(df_data_random)
    df_data_random = removeConstantSignalsDataframe(df_data_random, 0.)

    print('\nCalculating autocorrelations of %s random samples (sampled with replacement)...'%(df_data_random.shape[0]))

    results = runCPUs(NumberOfCPUs, pAutocorrelation, [(df_data_random.iloc[i].index.values, df_data_random.iloc[i].values, df_data.columns.values) for i in range(df_data_random.shape[0])])
    
    return pd.DataFrame(data=results[1::2], columns=results[0])


'''
HypothesisTesting BenjaminiHochbergFDR correction from MathIOmica.m
pValues: p-values (1D array of floats)
SignificanceLevel: default is 0.05
'''
def BenjaminiHochbergFDR(pValues, SignificanceLevel=0.05):

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


'''
Metric to calculate 'euclidean' distance between vectors u and v 
using only common non-missing points (not NaNs)
'''
def metricCommonEuclidean(u,v):

    where_common = (~np.isnan(u)) * (~np.isnan(v))

    return np.sqrt(((u[where_common] - v[where_common]) ** 2).sum())

###################################################################################################



### Clustering functions ##########################################################################
''' 
Get estimated number of clusters using ARI with KMeans
return: max peak, other possible peaks
'''
def getEstimatedNumberOfClusters(data, cluster_num_min, cluster_num_max, trials_to_do, numberOfAvailableCPUs=4, plotID=None, printScores=False):

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
                
    scores = runCPUs(numberOfAvailableCPUs, ARI.runForClusterNum, [(cluster_num, copy.deepcopy(data), trials_to_do) for cluster_num in range(cluster_num_min, cluster_num_max + 1)])

    if printScores: 
        print(scores)
                
    return getPeakPosition(scores, makePlot=True, plotID=plotID)[0]


''' 
Get optimal number clusters from linkage
'''
def get_optimal_number_clusters_from_linkage(Y):

    return np.diff(np.array([[nc, Y[-nc + 1][2]] for nc in range(2,min(50,len(Y)))]).T[1], 2).argmax() + 1 if len(Y) >= 5 else 1


''' 
Cluster data into N groups, if N is provided, else determine N
return: linkage matrix, cluster labels, possible cluster labels
'''
def getGroupingIndex(data, n_groups=None, method='weighted', metric='correlation', significance='Elbow'):

    Y = hierarchy.linkage(data, method=method, metric=metric, optimal_ordering=False)

    if n_groups == None:
        if significance=='Elbow':
            n_groups = get_optimal_number_clusters_from_linkage(Y)
        elif significance=='Silhouette':
            n_groups = 1
            print('Significance %s not implemented here!'%(significance))

    print('n_groups:', n_groups)

    labelsClusterIndex = scipy.cluster.hierarchy.fcluster(Y, t=n_groups, criterion='maxclust')

    groups = np.sort(np.unique(labelsClusterIndex))

    print([np.sum(labelsClusterIndex == group) for group in groups])

    return Y, labelsClusterIndex, groups


'''
Make a clustering Groups-Subgroups dictionary object
'''
def makeClusteringObject(df_data, df_data_autocorr):

    def getSubgroups(df_data):

        Y = hierarchy.linkage(df_data.values, method='weighted', metric=metricCommonEuclidean, optimal_ordering=True)
        leaves = hierarchy.dendrogram(Y, no_plot=True)['leaves']

        n_clusters = get_optimal_number_clusters_from_linkage(Y)

        clusters = scipy.cluster.hierarchy.fcluster(Y, t=n_clusters, criterion='maxclust')[leaves]

        return {cluster:df_data.index[leaves].values[clusters==cluster] for cluster in np.unique(clusters)}, Y

    ClusteringObject = {}

    ClusteringObject['linkage'], labelsClusterIndex, groups = getGroupingIndex(df_data_autocorr.values, method='weighted', metric='correlation')

    for group in groups:
        signals = df_data.index[labelsClusterIndex==group].values

        ClusteringObject[group], ClusteringObject[group]['linkage'] = ({1: signals}, None) if len(signals)==1 else getSubgroups(df_data.loc[signals])

        for subgroup in sorted([item for item in list(ClusteringObject[group].keys()) if not item=='linkage']):
            ClusteringObject[group][subgroup] = {'order':[np.where([temp==signal for temp in df_data.index.values])[0][0] for signal in list(ClusteringObject[group][subgroup])],
                                                 'data':df_data.loc[ClusteringObject[group][subgroup]], 
                                                 'dataAutocorr':df_data_autocorr.loc[ClusteringObject[group][subgroup]]}

    return ClusteringObject


'''
Export a clustering Groups-Subgroups dictionary object to a SpreadSheet
NOTE: linkage data is not exported
'''
def exportClusteringObject(ClusteringObject, saveDir, dataName, includeData=True, includeAutocorr=True):

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



### Visualization functions #######################################################################
'''
Make a histogram for each pandas Series (time point) in a pandas Dataframe
'''
def makeDataHistograms(df, saveDir, dataName):

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


'''
Make a combined plot of the signal and its Lomb-Scargle periodogram
for each pandas Series (time point) in a pandas Dataframe
'''
def makeLombScarglePeriodograms(df, saveDir, dataName):
        
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


'''
Make Dendrogram-Heatmap plot along with VIsibility graphs
'''
def makeDendrogramHeatmap(ClusteringObject, saveDir, dataName):

    def addAutocorrelationDendrogramAndHeatmap(ClusteringObject, groupColors, fig):

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
        
        axisDendro.plot([posA, axisDendro.get_xlim()[1]], [5., 5.], '--', color='black', linewidth = 1.0)
        axisDendro.plot([posA, axisDendro.get_xlim()[1]], [-5. + axisDendro.get_ylim()[1], -5. + axisDendro.get_ylim()[1]], '--', color='black', linewidth = 1.0)


        axisMatrixAC = fig.add_axes([0.78 + 0.07,0.1,0.18 - 0.075,0.8])

        cmap = plt.cm.bwr
        imAC = axisMatrixAC.imshow(tempData, aspect='auto', vmin=-1, vmax=1, origin='lower', cmap=cmap)
        for i in range(n_clusters - 1):
            axisMatrixAC.plot([-0.5, tempData.shape[1] - 0.5], [cluster_line_positions[i + 1] - 0.5, cluster_line_positions[i + 1] - 0.5], '--', color='black', linewidth = 1.0)

        axisMatrixAC.set_xticks([i for i in range(tempData.shape[1] - 1)])
        axisMatrixAC.set_xticklabels([i + 1 for i in range(tempData.shape[1] - 1)], fontsize=6)
        axisMatrixAC.set_yticks([])
        axisMatrixAC.set_xlabel('Lag')
        axisMatrixAC.set_title('Autocorrelation')

        axisColorAC = fig.add_axes([0.9 + 0.065,0.55,0.01,0.35])

        axisColorAC.tick_params(labelsize=6)
        plt.colorbar(imAC, cax=axisColorAC, ticks=[-1.0,1.0])

        return

    def addGroupDendrogramAndShowSubgroups(ClusteringObject, groupSize, bottom, top, group, groupColors, fig):

        Y = ClusteringObject[group]['linkage']

        n_clusters = get_optimal_number_clusters_from_linkage(Y)
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

        posA = ((axisDendro.get_xlim()[0] if n_clusters == 1 else Y[-n_clusters + 1][2]) + Y[-n_clusters][2]) / 2
        axisDendro.plot([posA, posA], [axisDendro.get_ylim()[0], axisDendro.get_ylim()[1]], 'k--', linewidth = 1)

        clusters = []
        for subgroup in sorted([item for item in list(ClusteringObject[group].keys()) if not item=='linkage']):
            clusters.extend([subgroup for _ in range(ClusteringObject[group][subgroup]['data'].values.shape[0])])

        clusters = np.array(clusters)

        cluster_line_positions = np.where(clusters - np.roll(clusters,1) != 0)[0]

        for i in range(n_clusters - 1):
            posB = cluster_line_positions[i + 1] * (axisDendro.get_ylim()[1] / len(Z['leaves'])) - 5. * 0
            axisDendro.plot([posA, axisDendro.get_xlim()[1]], [posB, posB], '--', color='black', linewidth = 1.0)
        
        axisDendro.plot([posA, axisDendro.get_xlim()[1]], [5., 5.], '--', color='black', linewidth = 1.0)
        axisDendro.plot([posA, axisDendro.get_xlim()[1]], [-5. + axisDendro.get_ylim()[1], -5. + axisDendro.get_ylim()[1]], '--', color='black', linewidth = 1.0)

        axisDendro.text(axisDendro.get_xlim()[0], 0.5 * axisDendro.get_ylim()[1], 
                        'G%s:' % group + str(groupSize), fontsize=14).set_path_effects([path_effects.Stroke(linewidth=1, foreground=groupColors[group - 1]),path_effects.Normal()])

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
            return axisMatrix.text(-1., pos, labelText, ha='right', va='center').set_path_effects([path_effects.Stroke(linewidth=0.4, foreground=groupColors[group - 1]),path_effects.Normal()])

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
            axisMatrix.set_xticklabels([np.int(i) for i in np.round(times,1)], rotation=0, fontsize=6)
            axisMatrix.set_xlabel('Time (hours)')

        if group == sorted([item for item in list(ClusteringObject.keys()) if not item=='linkage'])[-1]:
            axisMatrix.set_title('Transformed gene expression')

        axisColor = fig.add_axes([0.635 - 0.075 - 0.1 + 0.075,current_bottom + 0.01,0.01, max(0.01,(current_top - current_bottom) - 0.02)])
        plt.colorbar(im, cax=axisColor, ticks=[np.max(im._A),np.min(im._A)])
        axisColor.tick_params(labelsize=6)
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

    groupColors = ['b','g','r','c','m','y','k','b','g','r','c','m','y','k','b','g','r','c','m','y','k']
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


    def addVisibilityGraph(data, times, dataName, coords, numberOfVGs, groups_ac_colors, fig):

        group = int(dataName[:dataName.find('S')].strip('G'))

        fontsize = 4. * (8. + 5.) / (numberOfVGs + 5.)
        nodesize = 30. * (8. + 5.) / (numberOfVGs + 5.)

        (x1,x2,y1,y2) = coords

        def imputeWithMedian(data):

            data[np.isnan(data)] = np.median(data[np.isnan(data) == False])

            return data

        data = pd.DataFrame(data=data).apply(imputeWithMedian, axis=1).apply(lambda data: np.sum(data[data > 0.0]) / len(data), axis=0).values

        axisVG = fig.add_axes([x1,y1,x2 - x1,y2 - y1])
        graph_nx = nx.from_numpy_matrix(vg.getAdjecencyMatrixOfVisibilityGraph(data, times))
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list('GR', [(0, 1, 0), (1, 0, 0)], N=1000)

        pos = nx.circular_layout(graph_nx)
        keys = np.array(list(pos.keys())[::-1])
        values = np.array(list(pos.values()))
        keys = np.roll(keys, np.argmax(values.T[1]) - np.argmin(keys))
        pos = dict(zip(keys, values))

        shortest_path = nx.shortest_path(graph_nx, source=min(keys), target=max(keys))
        shortest_path_edges = [(shortest_path[i],shortest_path[i + 1]) for i in range(len(shortest_path) - 1)]

        nx.draw_networkx(graph_nx, pos=pos, ax=axisVG, node_color='y', edge_color='y', node_size=nodesize * 1.7, width=3.0, nodelist=shortest_path, edgelist=shortest_path_edges, with_labels=False)
        nx.draw_networkx(graph_nx, pos=pos, ax=axisVG, node_color=data, cmap=cmap, alpha=1.0, font_size=fontsize,  width=1.0, font_color='b', node_size=nodesize)


        def find_and_remove_node(graph_nx):
            bc = nx.betweenness_centrality(graph_nx)
            node_to_remove = list(bc.keys())[np.argmax(list(bc.values()))]
            graph_nx.remove_node(node_to_remove)
            return graph_nx, node_to_remove

        list_of_nodes = []
        graph_nx_inv = nx.from_numpy_matrix(vg.getAdjecencyMatrixOfVisibilityGraph(-data, times))
        for i in range(6):
            graph_nx_inv, node = find_and_remove_node(graph_nx_inv)
            list_of_nodes.append(node)
        
        if not 0 in list_of_nodes:
            list_of_nodes.append(0)

        list_of_nodes.append(list(graph_nx.nodes)[-1] + 1)
        list_of_nodes.sort()

        communities = [list(range(list_of_nodes[i],list_of_nodes[i + 1])) for i in range(len(list_of_nodes) - 1)]
        print(list_of_nodes)
        print()
        [print(community) for community in communities]
        print()
        xmin, xmax = axisVG.get_xlim()
        ymin, ymax = axisVG.get_ylim()
        X, Y = np.meshgrid(np.arange(xmin, xmax, (xmax - xmin) / 100.), np.arange(ymin, ymax, (ymax - ymin) / 100.))

        for icommunity, community in enumerate(communities):
            nX, nY = tuple(np.array([pos[node] for node in community]).T)
            Z = np.exp(X ** 2 - Y ** 2) * 0.
            for i in range(len(community)):
                Z += np.exp(-35. * (X - nX[i]) ** 2 - 35. * (Y - nY[i]) ** 2)
            level = 0.55
            CS = axisVG.contour(X, Y, Z, [level], linewidths=0.5, alpha=0.8, colors=groups_ac_colors[group - 1])
            #axisVG.clabel(CS, inline=True,fontsize=4,colors=group_colors[group-1], fmt ={level:'C%s'%icommunity})


        axisVG.spines['left'].set_visible(False)
        axisVG.spines['right'].set_visible(False)
        axisVG.spines['top'].set_visible(False)
        axisVG.spines['bottom'].set_visible(False)
        axisVG.set_xticklabels([])
        axisVG.set_yticklabels([])
        axisVG.set_xticks([])
        axisVG.set_yticks([])

        axisVG.text(axisVG.get_xlim()[1], (axisVG.get_ylim()[1] + axisVG.get_ylim()[0]) * 0.5, dataName, ha='left', va='center', fontsize=8).set_path_effects([path_effects.Stroke(linewidth=0.4, foreground=groups_ac_colors[group - 1]),path_effects.Normal()])

        titleText = dataName + ' (size: ' + str(data.shape[0]) + ')' + ' min=%s max=%s' % (np.round(min(data),2), np.round(max(data),2))
        #axisVG.set_title(titleText, fontsize=10)

        return

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
    
    fig.savefig(saveDir + dataName + '_DendrogramHeatmap.svg', dpi=600) #*.svg

    return

###################################################################################################



### Dataframe functions ###########################################################################
def prepareDataframe(dataDir, dataFileName, AlltimesFileName):

    df = pd.read_csv(dataDir + dataFileName, delimiter=',', header=None)

    df = df.set_index(df[df.columns[0]]).drop(columns=[df.columns[0]])

    df.columns = list(pd.read_csv(dataDir + AlltimesFileName, delimiter=',', header=None).values.T[0])

    return df

def filterOutAllZeroSignalsDataframe(df):

    print('Filtering out all-zero signals...')

    init = df.shape[0]

    df = df.loc[df.index[np.count_nonzero(df, axis=1) > 0]]

    print('Removed ', init - df.shape[0], 'signals out of %s.' % init) 
    print('Remaining ', df.shape[0], 'signals!')

    return df

def filterOutFractionZeroSignalsDataframe(df, max_fraction_of_allowed_zeros):

    print('Filtering out low-quality signals (with more than %s%% missing points)...' %(100.*(1.-max_fraction_of_allowed_zeros)))

    init = df.shape[0]

    min_number_of_non_zero_points = np.int(np.round(max_fraction_of_allowed_zeros * df.shape[1],0))
    df = df.loc[df.index[np.count_nonzero(df, axis=1) >= min_number_of_non_zero_points]]

    if (init - df.shape[0]) > 0:
        print('Removed ', init - df.shape[0], 'signals out of %s.' % init) 
        print('Remaining ', df.shape[0], 'signals!')

    return df

def filterOutFirstPointZeroSignalsDataframe(df):

    print('Filtering out first time point zeros signals...')

    init = df.shape[0]

    df = df.loc[df.iloc[:,0] > 0.0]

    if (init - df.shape[0]) > 0:
        print('Removed ', init - df.shape[0], 'signals out of %s.' % init) 
        print('Remaining ', df.shape[0], 'signals!')

    return df

def tagMissingValuesDataframe(df):

    print('Tagging missing (i.e. zero) values with NaN...')

    df[df == 0.] = np.NaN

    return df

def tagLowValuesDataframe(df, cutoff, replacement):

    print('Tagging low values (<=%s) with %s...'%(cutoff, replacement))

    df[df <= cutoff] = replacement

    return df

def removeConstantSignalsDataframe(df, theta_cutoff):

    print('\nRemoving constant genes. Cutoff value is %s' % (theta_cutoff))

    init = df.shape[0]

    df = df.iloc[np.where(np.std(df,axis=1) / np.mean(np.std(df,axis=1)) > theta_cutoff)[0]]

    print('Removed ', init - df.shape[0], 'signals out of %s.' % init)
    print('Remaining ', df.shape[0], 'signals!')

    return df

def boxCoxTransformDataframe(df):
    
    print('Box-cox transforming raw data...', end='\t', flush=True)
            
    df = df.apply(boxCoxTransform, axis=0)

    print('Done')

    return df

def modifiedZScoreDataframe(df):
            
    print('Z-score (Median-based) transforming box-cox transformed data...', end='\t', flush=True)

    df = df.apply(modifiedZScore, axis=0)

    print('Done')

    return df

def normalizeSignalsToUnityDataframe(df):

    print('Normalizing signals to unity...')

    #Subtract 0-time-point value from all time-points

    df.iloc[:] = (df.values.T - df.values.T[0]).T
    
    where_nan = np.isnan(df.values.astype(float))
    df[where_nan] = 0.0
    df = df.apply(lambda data: data / np.sqrt(np.dot(data,data)),axis=1)
    df[where_nan] = np.nan

    return df

###################################################################################################
