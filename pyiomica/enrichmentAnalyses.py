'''Annotations and Enumerations'''


import pymysql
import datetime
import urllib.request
import requests

from .globalVariables import *

from . import utilityFunctions
from . import dataStorage


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

    returning = dict(zip(list(np.array(list(ResultsHCct.keys()), dtype=object)[whatIsFiltered]),list(np.array(list(ResultsHCct.values()), dtype=object)[whatIsFiltered])))

    return {list(data.keys())[0]: returning}

def OBOGODictionary(FileURL="http://purl.obolibrary.org/obo/go/go-basic.obo", ImportDirectly=False, PyIOmicaDataDirectory=None, OBOFile="goBasicObo.txt"):

    """Generate Open Biomedical Ontologies (OBO) Gene Ontology (GO) vocabulary dictionary.
    
    Parameters: 
        FileURL: str, Default "http://purl.obolibrary.org/obo/go/go-basic.obo"
            Provides the location of the Open Biomedical Ontologies (OBO) Gene Ontology (GO) 
            file in case this will be downloaded from the web

        ImportDirectly: boolean, Default False
            Import from URL regardles is the file already exists

        PyIOmicaDataDirectory: str, Default None
            Path of directories to data storage

        OBOFile: str, Default "goBasicObo.txt"
            Name of file to store data in (file will be zipped)

    Returns:
        dictionary
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
    
    Parameters: 
        geneUCSCTable: str, Default None
            Path to a geneUCSCTable file

        UCSCSQLString: str, Default None
            An association to be used to obtain data from the UCSC Browser tables. The key of the association must 
            match the Species option value used (default: human). The value for the species corresponds to the actual MySQL command used

        UCSCSQLSelectLabels: str, Default None
            An association to be used to assign key labels for the data imported from the UCSC Browser tables. 
            The key of the association must match the Species option value used (default: human). The value is a multi component string 
            list corresponding to the matrices in the data file, or the tables used in the MySQL query provided by UCSCSQLString

        ImportDirectly: boolean, Default False
            Import from URL regardles is the file already exists

        Species: str, Default "human"
            Species considered in the calculation, by default corresponding to human

        KEGGUCSCSplit: list, Default [True,"KEGG Gene ID"]
            Two component list, {True/False, label}. If the first component is set to True the initially imported KEGG IDs, 
            identified by the second component label,  are split on + string to fix nomenclature issues, retaining the string following +

    Returns:
        dictionary
            Gene dictionary

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
        termTable = dataStorage.read(geneUCSCTable, jsonFormat=True)[1]
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

        except Exception as exception:
            print(exception)
            print ("Error: unable to fetch data")

        termTable = np.array(termTable).T
        termTable[np.where(termTable=="")] = None

        #Get all the terms we are going to need, import with SQL the combined tables,and export with a time stamp
        dataStorage.write((datetime.datetime.now().isoformat(), termTable.tolist()), geneUCSCTable, jsonFormat=True)

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

    Parameters: 
        PyIOmicaDataDirectory: str, Default None
            The directory where the default package data is stored

        ImportDirectly: boolean, Default False
            Import from URL regardles is the file already exists

        BackgroundSet: list, Default []
            Background list to create annotation projection to limited background space, involves
            considering pathways/groups/sets and that provides a list of IDs (e.g. gene accessions) that should 
            be considered as the background for the calculation

        Species: str, Default "human"
            Species considered in the calculation, by default corresponding to human

        LengthFilterFunction: function, Default np.greater_equal
            Performs computations of membership in pathways/ontologies/groups/sets, 
            that specifies which function to use to filter the number of members a reported category has 
            compared to the number typically provided by LengthFilter 

        LengthFilter: int, Default None
            Argument for LengthFilterFunction

        GOFileName: str, Default None
            The name for the specific GO file to download from the GOURL if option ImportDirectly is set to True

        GOFileColumns: list, Default [2, 5]
            Columns to use for IDs and GO:accessions respectively from the downloaded GO annotation file, 
            used when ImportDirectly is set to True to obtain a new GO association file

        GOURL: str, Default "http://current.geneontology.org/annotations/"
            The location (base URL) where the GO association annotation files are downloaded from

    Returns:
        dictionary
            Dictionary of IDToGO and GOToID dictionaries

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

        identifierAssoc = utilityFunctions.createReverseDictionary(geneOntAssoc)

        #Save created annotations geneOntAssoc, identifierAssoc
        dataStorage.write((datetime.datetime.now().isoformat(), geneOntAssoc), fileGOAssociations[0], jsonFormat=True)
        dataStorage.write((datetime.datetime.now().isoformat(), identifierAssoc), fileGOAssociations[1], jsonFormat=True)
        
        os.remove(localFile)

        if np.array(list(map(os.path.isfile, fileGOAssociations))).all():
            print("Created Annotation Files at ", fileGOAssociations)
        else:
            print("Did Not Find Annotation Files, Aborting Process")
            return
    else:
        #Otherwise we get from the user specified PyIOmicaDataDirectoryectory
        geneOntAssoc = dataStorage.read(fileGOAssociations[0], jsonFormat=True)[-1]
        identifierAssoc = dataStorage.read(fileGOAssociations[1], jsonFormat=True)[-1]

    if BackgroundSet!=[]:
        #Using provided background list to create annotation projection to limited background space, also remove entries with only one and missing value
        keys, values = np.array(list(identifierAssoc.keys())), np.array(list(identifierAssoc.values()))
        index = np.where([(((len(values[i])==True)*(values[i][0]!=values[i][0]))==False)*(keys[i] in BackgroundSet) for i in range(len(keys))])[0]
        identifierAssoc = dict(zip(keys[index],values[index]))

        #Create corresponding geneOntAssoc
        geneOntAssoc = utilityFunctions.createReverseDictionary(identifierAssoc)

    if LengthFilter!=None:
        keys, values = np.array(list(geneOntAssoc.keys()), dtype=object), np.array(list(geneOntAssoc.values()), dtype=object)
        index = np.where(LengthFilterFunction(np.array([len(value) for value in values]), LengthFilter))[0]
        geneOntAssoc = dict(zip(keys[index],values[index]))

        #Create corresponding identifierAssoc
        identifierAssoc = utilityFunctions.createReverseDictionary(geneOntAssoc)

    return {Species : {"IDToGO": identifierAssoc, "GOToID": geneOntAssoc}}

def obtainConstantGeneDictionary(GeneDictionary, GetGeneDictionaryOptions, AugmentDictionary):
    
    """Obtain gene dictionary - if it exists can either augment with new information or Species or create new, 
    if not exist then create variable.

    Parameters:
        GeneDictionary: dictionary or None
            An existing variable to use as a gene dictionary in annotations. 
            If set to None the default ConstantGeneDictionary will be used

        GetGeneDictionaryOptions: dictionary
            A list of options that will be passed to this internal GetGeneDictionary function

        AugmentDictionary: boolean
            A choice whether or not to augment the current ConstantGeneDictionary global variable or create a new one

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

    Parameters:
        data: pd.DataFrame or list
            Data to analyze

        GetGeneDictionaryOptions: dictionary, Default {}
            A list of options that will be passed to this internal GetGeneDictionary function

        AugmentDictionary: boolean, Default True
            A choice whether or not to augment the current ConstantGeneDictionary global variable or create a new one

        InputID: list, Default ["UniProt ID","Gene Symbol"]
            Kind of identifiers/accessions used as input

        OutputID: str, Default "UniProt ID"
            Kind of IDs/accessions to convert the input IDs/accession numbers in the function's analysis

        GOAnalysisAssignerOptions: dictionary, Default {}
            A list of options that will be passed to the internal GOAnalysisAssigner function

        BackgroundSet: list, Default []
            Background list to create annotation projection to limited background space, involves
            considering pathways/groups/sets and that provides a list of IDs (e.g. gene accessions) that should be 
            considered as the background for the calculation

        Species: str, Default "human"
            The species considered in the calculation, by default corresponding to human

        OntologyLengthFilter: int, Default 2
            Function that can be used to set the value for which terms to consider in the computation, 
            by excluding GO terms that have fewer items compared to the OntologyLengthFilter value. It is used by the internal
            GOAnalysisAssigner function

        ReportFilter: int, Default 1
            Functions that use pathways/ontologies/groups, and provides a cutoff for membership in ontologies/pathways/groups
            in selecting which terms/categories to return. It is typically used in conjunction with ReportFilterFunction

        ReportFilterFunction: function , Default np.greater_equal
            Specifies what operator form will be used to compare against ReportFilter option value in 
            selecting which terms/categories to return

        pValueCutoff: float, Default 0.05
            Significance cutoff

        TestFunction: function, Default lambda n, N, M, x: 1. - scipy.stats.hypergeom.cdf(x-1, M, n, N)
            Test function

        HypothesisFunction: function, Default lambda data, SignificanceLevel: BenjaminiHochbergFDR(data, SignificanceLevel=SignificanceLevel)["Results"]
            Allows the choice of function for implementing multiple hypothesis testing considerations

        FilterSignificant: boolean, Default True
            Can be set to True to filter data based on whether the analysis result is statistically significant, 
            or if set to False to return all membership computations

        OBODictionaryVariable: str, Default None
            A GO annotation variable. If set to None, OBOGODictionary will be used internally to 
            automatically generate the default GO annotation

        OBOGODictionaryOptions: dictionary, Default {}
            A list of options to be passed to the internal OBOGODictionary function that provides the GO annotations

        MultipleListCorrection: boolean, Default None
            Specifies whether or not to correct for multi-omics analysis. The choices are None, Automatic, 
            or a custom number, e.g protein+RNA

        MultipleList: boolean, Default False
            Specifies whether the input accessions list constituted a multi-omics list input that is annotated so

        GeneDictionary: str, Default None
            Points to an existing variable to use as a gene dictionary in annotations. If set to None 
            the default ConstantGeneDictionary will be used

    Returns:
        dictionary
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
    
    listToggle = False

    #If the input is simply a list
    if type(data) is list:
        data = {'dummy': data}
        listToggle = True

    #The data may be a subgroup from a clustering object, i.e. a pd.DataFrame
    if type(data) is pd.DataFrame:
        id = list(data.index.get_level_values('id'))
        source = list(data.index.get_level_values('source'))
        data = [[id[i], source[i]] for i in range(len(data))]
        data = {'dummy': data}
        listToggle = True

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
    
    Parameters:
        InputList: list
            List of names

        TargetIDList: list
            Target ID list

        GeneDictionary: dictionary
            An existing variable to use as a gene dictionary in annotations.
            If set to None the default ConstantGeneDictionary will be used

        InputID: str, Default None
            The kind of identifiers/accessions used as input

        Species: str, Default "human"
            The species considered in the calculation, by default corresponding to human
    
    Returns:
        dictionary
            Gene dictionary

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

    Parameters: 
        PyIOmicaDataDirectory: str, Default None
            Directory where the default package data is stored

        ImportDirectly: boolean, Default False
            Import from URL regardles is the file already exists

        BackgroundSet: list, Default []
            A list of IDs (e.g. gene accessions) that should be considered as the background for the calculation

        KEGGQuery1: str, Default "pathway"
            Make KEGG API calls, and sets string query1 in http://rest.kegg.jp/link/<> query1 <> / <> query2. 
            Typically this will be used as the target database to find related entries by using database cross-references

        KEGGQuery2: str, Default "hsa"
            KEGG API calls, and sets string query2 in http://rest.kegg.jp/link/<> query1 <> / <> query2. 
            Typically this will be used as the source database to find related entries by using database cross-references

        LengthFilterFunction: function, Default np.greater_equal
            Option for functions that perform computations of membership in 
            pathways/ontologies/groups/sets, that specifies which function to use to filter the number of members a reported 
            category has compared to the number typically provided by LengthFilter

        LengthFilter: int, Default None
            Allows the selection of how many members each category can have, as typically 
            restricted by the LengthFilterFunction

        Labels: list, Default ["IDToPath", "PathToID"]
            A string list for how keys in a created association will be named

    Returns:
        dictionary
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
        idToPath = utilityFunctions.createReverseDictionary(pathToID)

        dataStorage.write((datetime.datetime.now().isoformat(), idToPath), fileAssociations[0], jsonFormat=True)
        dataStorage.write((datetime.datetime.now().isoformat(), pathToID), fileAssociations[1], jsonFormat=True)

        os.remove(localFile)

        if np.array(list(map(os.path.isfile, fileAssociations))).all():
            print("Created Annotation Files at ", fileAssociations)
        else:
            print("Did Not Find Annotation Files, Aborting Process")
            return
    else:
        #otherwise import the necessary associations from PyIOmicaDataDirectoryectory
        idToPath = dataStorage.read(fileAssociations[0], jsonFormat=True)[1]
        pathToID = dataStorage.read(fileAssociations[1], jsonFormat=True)[1]

    if BackgroundSet!=[]:
        #Using provided background list to create annotation projection to limited background space
        keys, values = np.array(list(idToPath.keys())), np.array(list(idToPath.values()))
        index = np.where([(((len(values[i])==True)*(values[i][0]!=values[i][0]))==False)*(keys[i] in BackgroundSet) for i in range(len(keys))])[0]
        idToPath = dict(zip(keys[index],values[index]))

        #Create corresponding reverse dictionary
        pathToID = utilityFunctions.createReverseDictionary(idToPath)

    if LengthFilter!=None:
        keys, values = np.array(list(pathToID.keys())), np.array(list(pathToID.values()))
        index = np.where(LengthFilterFunction(np.array([len(value) for value in values]), LengthFilter))[0]
        pathToID = dict(zip(keys[index],values[index]))

        #Create corresponding reverse dictionary
        idToPath = utilityFunctions.createReverseDictionary(pathToID)

    return {KEGGQuery2 : {Labels[0]: idToPath, Labels[1]: pathToID}}

def KEGGDictionary(PyIOmicaDataDirectory = None, ImportDirectly = False, KEGGQuery1 = "pathway", KEGGQuery2 = "hsa"):

    """Create a dictionary from KEGG: Kyoto Encyclopedia of Genes and Genomes terms - 
    typically association of pathways and members therein.
    
    Parameters: 
        PyIOmicaDataDirectory: str, Default None
            directory where the default package data is stored

        ImportDirectly: boolean, Default False
            import from URL regardles is the file already exists

        KEGGQuery1: str, Default "pathway"
            make KEGG API calls, and sets string query1 in http://rest.kegg.jp/link/<> query1 <> / <> query2. 
            Typically this will be used as the target database to find related entries by using database cross-references

        KEGGQuery2: str, Default "hsa"
            KEGG API calls, and sets string query2 in http://rest.kegg.jp/link/<> query1 <> / <> query2. 
            Typically this will be used as the source database to find related entries by using database cross-references

    Returns:
        dictionary
            Dictionary of definitions

    Usage:
        KEGGDict = KEGGDictionary()
    """
    
    global ConstantPyIOmicaDataDirectory
    
    PyIOmicaDataDirectory = ConstantPyIOmicaDataDirectory if PyIOmicaDataDirectory==None else PyIOmicaDataDirectory

    #if the user asked us to import directly, import directly from KEGG website, otherwise, get it from a directory they specify
    fileKEGGDict = os.path.join(PyIOmicaDataDirectory, KEGGQuery1 + "_" + KEGGQuery2 + "_KEGGDictionary.json.gz")

    if os.path.isfile(fileKEGGDict):
        associationKEGG = dataStorage.read(fileKEGGDict, jsonFormat=True)[1]
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

        dataStorage.write((datetime.datetime.now().isoformat(), associationKEGG), fileKEGGDict, jsonFormat=True)

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

    Parameters:
        data: pandas.DetaFrame or list
            Data to analyze

        AnalysisType: str, Default "Genomic"
            Analysis methods that may be used, "Genomic", "Molecular" or "All"

        GetGeneDictionaryOptions: dictionary, Default {}
            A list of options that will be passed to this internal GetGeneDictionary function

        AugmentDictionary: boolean, Default True
            A choice whether or not to augment the current ConstantGeneDictionary global variable or create a new one

        InputID: list, Default ["UniProt ID", "Gene Symbol"]
            The kind of identifiers/accessions used as input

        OutputID: str, Default "KEGG Gene ID"
            A string value that specifies what kind of IDs/accessions to convert the input IDs/accession 
            numbers in the function's analysis

        MolecularInputID: list, Default ["cpd"]
            A string list to indicate the kind of ID to use for the input molecule entries

        MolecularOutputID: str, Default "cpd"
            A string list to indicate the kind of ID to use for the input molecule entries

        KEGGAnalysisAssignerOptions: dictionary, Default {}
            A list of options that will be passed to this internal KEGGAnalysisAssigner function

        BackgroundSet: list, Default []
            A list of IDs (e.g. gene accessions) that should be considered as the background for the calculation

        KEGGOrganism: str, Default "hsa"
            Indicates which organism (org) to use for \"Genomic\" type of analysis (default is human analysis: org=\"hsa\")

        KEGGMolecular: str, Default "cpd"
            Which database to use for molecular analysis (default is the compound database: cpd)

        KEGGDatabase: str, Default "pathway"
            KEGG database to use as the target database

        PathwayLengthFilter: int, Default 2
            Pathways to consider in the computation, by excluding pathways that have fewer items 
            compared to the PathwayLengthFilter value

        ReportFilter: int, Default 1
            Provides a cutoff for membership in ontologies/pathways/groups in selecting which terms/categories 
            to return. It is typically used in conjunction with ReportFilterFunction

        ReportFilterFunction: function, Default np.greater_equal
            Operator form will be used to compare against ReportFilter option value in selecting 
            which terms/categories to return

        pValueCutoff: float, Default 0.05
            A cutoff p-value for (adjusted) p-values to assess statistical significance

        TestFunction: function, Default lambda n, N, M, x: 1. - scipy.stats.hypergeom.cdf(x-1, M, n, N)
            A function used to calculate p-values

        HypothesisFunction: function, Default lambda data, SignificanceLevel: BenjaminiHochbergFDR(data, SignificanceLevel=SignificanceLevel)["Results"]
            Allows the choice of function for implementing multiple hypothesis testing considerations

        FilterSignificant: boolean, Default True
            Can be set to True to filter data based on whether the analysis result is statistically significant, 
            or if set to False to return all membership computations

        KEGGDictionaryVariable: str, Default None
            KEGG dictionary, and provides a KEGG annotation variable. If set to None, KEGGDictionary 
            will be used internally to automatically generate the default KEGG annotation

        KEGGDictionaryOptions: dictionary, Default {}
            A list of options to be passed to the internal KEGGDictionary function that provides the KEGG annotations

        MultipleListCorrection: boolean, Default None
            Specifies whether or not to correct for multi-omics analysis. 
            The choices are None, Automatic, or a custom number

        MultipleList: boolean, Default False 
            Whether the input accessions list constituted a multi-omics list input that is annotated so

        GeneDictionary: str, Default None
            Existing variable to use as a gene dictionary in annotations. If set to None the default ConstantGeneDictionary will be used

        Species: str, Default "human"
            The species considered in the calculation, by default corresponding to human

        MolecularSpecies: str, Default "compound"
            The kind of molecular input

        NonUCSC: , Default 
            If UCSC browser was used in determining an internal GeneDictionary used in ID translations,
            where the KEGG identifiers for genes are number strings (e.g. 4790).The NonUCSC option can be set to True 
            if standard KEGG accessions are used in a user provided GeneDictionary variable, 
            in the form OptionValue[KEGGOrganism] <>:<>numberString, e.g. hsa:4790

        PyIOmicaDataDirectory: str, Default None
            Directory where the default package data is stored

    Returns:
        dictionary
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
            GeneDictionary = dataStorage.read(fileMolDict, jsonFormat=True)[1]
        else:
            fileCSV = os.path.join(PackageDirectory, "data", "MathIOmicaMolecularDictionary.csv")

            print('Attempting to read:', fileCSV)

            if os.path.isfile(fileCSV):
                with open(fileCSV, 'r') as tempFile:
                    tempLines = tempFile.readlines()
            
                tempData = np.array([line.strip('\n').replace('"', '').split(',') for line in tempLines]).T
                tempData = {'compound': {'pumchem': tempData[0].tolist(), 'cpd': tempData[1].tolist()}}
                dataStorage.write((datetime.datetime.now().isoformat(), tempData), fileMolDict, jsonFormat=True)
            else:
                print("Could not find annotation file at " + fileMolDict + " Please either obtain an annotation file from mathiomica.org or provide a GeneDictionary option variable.")
                return

            GeneDictionary = dataStorage.read(fileMolDict, jsonFormat=True)[1]

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

    listToggle = False

    #If the input is simply a list
    if type(data) is list:
        data = {'dummy': data}
        listToggle = True

    #The data may be a subgroup from a clustering object, i.e. a pd.DataFrame
    if type(data) is pd.DataFrame:
        id = list(data.index.get_level_values('id'))
        source = list(data.index.get_level_values('source'))
        data = [[id[i], source[i]] for i in range(len(data))]
        data = {'dummy': data}
        listToggle = True

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
    
    Parameters: 
        data: np.array
            Input data

        accuracy: float
            Accuracy

        MassDictionaryVariable: boolean, Default None
            Mass dictionary variable. If set to None, inbuilt 
            mass dictionary (MassDictionary) will be loaded and used

        MolecularSpecies: str, Default "cpd"
            The kind of molecular input

    Returns:
        list
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
    
    Parameters:
        PyIOmicaDataDirectory: str, Default None
            Directory where the default package data is stored

    Returns:
        dictionary
            Mass dictionary

    Usage:
        MassDict = MassDictionary()
    """

    global ConstantPyIOmicaDataDirectory

    PyIOmicaDataDirectory = ConstantPyIOmicaDataDirectory if PyIOmicaDataDirectory==None else PyIOmicaDataDirectory

    fileMassDict = os.path.join(PyIOmicaDataDirectory, "PyIOmicaMassDictionary.json.gz")

    if os.path.isfile(fileMassDict):
        MassDict = dataStorage.read(fileMassDict, jsonFormat=True)[1]
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
            dataStorage.write((datetime.datetime.now().isoformat(), MassDict), fileMassDict, jsonFormat=True)

            print("Created mass dictionary at ", fileMassDict)
        else:
            print("Could not find mass dictionary at ", fileMassDict, 
                    "Please either obtain a mass dictionary file from mathiomica.org or provide a custom file at the above location.")

            return None

    return MassDict

def ExportEnrichmentReport(data, AppendString="", OutputDirectory=None):

    """Export results from enrichment analysis to Excel spreadsheets.
    
    Parameters:
        data: dictionary
            Enrichment results

        AppendString: str, Default ""
            Custom report name, if empty then time stamp will be used

        OutputDirectory: boolean, Default None
            Path of directories where the report will be saved

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
             
            format = writer.book.add_format({'text_wrap': False,
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

    utilityFunctions.createDirectories(saveDir)

    if AppendString=="":
        AppendString=(datetime.datetime.now().isoformat().replace(' ', '_').replace(':', '_').split('.')[0])

    ExportToFile(saveDir + AppendString + '.xlsx', FlattenDataForExport(data))

    return None

def BenjaminiHochbergFDR(pValues, SignificanceLevel=0.05):

    """HypothesisTesting BenjaminiHochbergFDR correction

    Parameters:
        pValues: 1d numpy.array
            Array of p-values

        SignificanceLevel: float, Default 0.05
            Significance level

    Returns:
        dictionary
            Corrected p-Values, p- and q-Value cuttoffs

    Usage:
        result = BenjaminiHochbergFDR(pValues)
    """

    #pValues = np.round(pValues,6)
      
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

def ReactomeAnalysis(data, 
                     uploadURL = 'https://reactome.org/AnalysisService/identifiers/projection',
                     preDownloadURL = 'https://reactome.org/AnalysisService/download/',
                     postDownloadURL = '/pathways/TOTAL/result.csv',
                     headersPOST = {'accept': 'application/json', 'content-type': 'text/plain'},
                     headersGET =  {'accept': 'text/CSV'},
                     URLparameters = (('interactors', 'false'), ('pageSize', '20'), ('page', '1'), ('sortBy', 'ENTITIES_PVALUE'), ('order', 'ASC'), ('resource', 'TOTAL'))):

    """Reactome POST-GET-style analysis.
    
    Parameters: 
        data: pd.DataFrame or list
            Data to analyze

        uploadURL: str, Default 'https://reactome.org/AnalysisService/identifiers/projection'
            URL for POST request

        preDownloadURL: str, Default 'https://reactome.org/AnalysisService/download/'
            Part 1 of URL for GET request

        postDownloadURL: str, Default '/pathways/TOTAL/result.csv'
            Part 2 of URL for GET request

        headersPOST: dict, Default {'accept': 'application/json', 'content-type': 'text/plain'}
            URL headers for POST request

        headersGET: dict, Default {'accept': 'text/CSV'}
            URL headers for GET request

        URLparameters: tuple, Default (('interactors', 'false'), ('pageSize', '20'), ('page', '1'), ('sortBy', 'ENTITIES_PVALUE'), ('order', 'ASC'), ('resource', 'TOTAL'))
            Parameters for POST request

    Returns:
        returning
            Enrichment object

    Usage:
        goExample1 = ReactomeAnalysis(["TAB1", "TNFSF13B", "MALT1", "TIRAP", "CHUK", 
                                "TNFRSF13C", "PARP1", "CSNK2A1", "CSNK2A2", "CSNK2B", "LTBR", 
                                "LYN", "MYD88", "GADD45B", "ATM", "NFKB1", "NFKB2", "NFKBIA", 
                                "IRAK4", "PIAS4", "PLAU"])
    """
    
    def internalQueryReactome(data):

        '''Function used internally

        Parameters:
            data: str, list or list-like
                Input identifiers in a form of one string, list or a list-like object

        Returns:
            pandas.DataFrame
                Reactome enrichemnt result

        Usage:
            internalQueryReactome('P01023, Q99758, O15439, O43184')
        '''

        # Check or convert data into a string of delimeter-separated values
        if type(data) is str:
            dataString = data.replace("'", "").replace("\"", "")
        else:
            if type(data) is list:
                dataString = str(data).replace("'", "").replace("\"", "").strip(']').strip('[')
            else:
                dataString = str(list(data)).replace("'", "").strip(']').strip('[')

        # POST request to uploadURL with specific 'data'
        response = requests.post(uploadURL, headers=headersPOST, params=URLparameters, data=dataString)

        # Read identifier of the POST request, to use it with GET request
        responseToken = response.json()['summary']['token']

        # GET the csv result from downloadURL using identifier from POST request
        response = requests.get(preDownloadURL + responseToken + postDownloadURL, headers=headersGET)

        # Set up a stream from GET response
        stream = io.StringIO(response.content.decode('utf-8'))
        
        # Read GET response stream into pandas DataFrame
        enrichmentDataFrame = pd.read_csv(stream, index_col=0)

        return enrichmentDataFrame

    listToggle = False

    #If the input is simply a list
    if type(data) is list:
        data = {'dummy': data}
        listToggle = True

    #The data may be a subgroup from a clustering object, i.e. a pd.DataFrame
    if type(data) is pd.DataFrame:
        id = list(data.index.get_level_values('id'))
        data = {'dummy': np.unique(id)}
        listToggle = True
    
    returning = {}

    #Check if a clustering object
    if "linkage" in data.keys():
        #Loop through the clustering object, calculate GO for each SubGroup
        for keyGroup in sorted([item for item in list(data.keys()) if not item=='linkage']):
            returning[keyGroup] = {}
            for keySubGroup in sorted([item for item in list(data[keyGroup].keys()) if not item=='linkage']):
                SubGroupMultiIndex = data[keyGroup][keySubGroup]['data'].index
                SubGroupGenes = list(SubGroupMultiIndex.get_level_values('id'))
                SubGroupList = np.unique(SubGroupGenes)

                returning[keyGroup][keySubGroup] = internalQueryReactome(SubGroupList)
                
    #The data is a dictionary of type {'Name1': [data1], 'Name2': [data2], ...}
    else:
        for key in list(data.keys()):
            returning.update({key: internalQueryReactome(data[key])})

        #If a single list was provided, return the association for Gene Ontologies
        returning = returning['dummy'] if listToggle else returning

    return returning

def ExportReactomeEnrichmentReport(data, AppendString="", OutputDirectory=None):

    """Export results from enrichment analysis to Excel spreadsheets.
    
    Parameters:
        data: dictionary or pandas.DataFrame
            Reactome pathway enrichment results

        AppendString: str, Default ""
            Custom report name, if empty then time stamp will be used

        OutputDirectory: boolean, Default None
            Path of directories where the report will be saved

    Returns:
        None

    Usage:
        ExportReactomeEnrichmentReport(example1, AppendString='example1', OutputDirectory=None)
    """

    def FlattenDataForExport(data):

        returning = {}

        if (type(data) is pd.DataFrame):
            returning['List'] = data
        elif (type(data) is dict):
            idata = data[list(data.keys())[0]]
            if type(idata) is pd.DataFrame:
                returning = data
            elif type(idata) is dict:
                idata = idata[list(idata.keys())[0]]
                if type(idata) is pd.DataFrame:
                    for keyClass in list(data.keys()):
                        for keySubClass in list(data[keyClass].keys()):
                            returning[str(keyClass)+' '+str(keySubClass)] = data[keyClass][keySubClass]
        else:
            print('Results type is not supported...')

        return returning

    def ExportToFile(fileName, data):

        writer = pd.ExcelWriter(fileName)

        for key in list(data.keys()):

            df = data[key]

            df.to_excel(writer, str(key))

            writer.sheets[str(key)].set_column('A:A', df.index.astype(str).map(len).max()+2)
             
            format = writer.book.add_format({'text_wrap': False,
                                             'valign': 'top'})

            for idx, column in enumerate(df.columns):
                max_len = max((df[column].astype(str).map(len).max(),  # len of largest item
                            len(str(df[column].name)))) + 1            # len of column name/header adding a little extra space

                width = 50 if ((column=='Pathway name') or (column=='Found reaction identifiers')) else min(180, max_len)

                writer.sheets[str(key)].set_column(idx+1, idx+1, width, format)  # set column width

        writer.save()

        print('Saved:', fileName)

        return None
    
    saveDir = os.path.join(os.getcwd(), "Enrichment reports") if OutputDirectory==None else OutputDirectory

    utilityFunctions.createDirectories(saveDir)

    if AppendString=="":
        AppendString=(datetime.datetime.now().isoformat().replace(' ', '_').replace(':', '_').split('.')[0])

    ExportToFile(os.path.join(saveDir, AppendString + '.xlsx'), FlattenDataForExport(data))

    return None


