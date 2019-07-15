from pyiomica import pyiomica

## Examples of GO Analysis ################################################################################################################################################
def testGOAnalysis(EnrichmentOutputDirectory):

    #Let's do a GO analysis for a group of genes, annotated with their "Gene Symbol":
    goExample1 = pyiomica.GOAnalysis(["TAB1", "TNFSF13B", "MALT1", "TIRAP", "CHUK", 
                                    "TNFRSF13C", "PARP1", "CSNK2A1", "CSNK2A2", "CSNK2B", "LTBR", 
                                    "LYN", "MYD88", "GADD45B", "ATM", "NFKB1", "NFKB2", "NFKBIA", 
                                    "IRAK4", "PIAS4", "PLAU"])

    pyiomica.ExportEnrichmentReport(goExample1, AppendString='goExample1', OutputDirectory=EnrichmentOutputDirectory + 'GOAnalysis/')

    #The information can be computed for multiple groups, if these are provided as an association:
    analysisGOAssociation = pyiomica.GOAnalysis({"Group1": ["C6orf57", "CD46", "DHX58", "HMGB3", "MAP3K5", "NFKB2", "NOS2", "PYCARD", "PYDC1", "SSC5D"], 
                                                    "Group2": ["TAB1", "TNFSF13B", "MALT1", "TIRAP", "CHUK", "TNFRSF13C", "PARP1", "CSNK2A1", "CSNK2A2", "CSNK2B", "LTBR", 
                                                                "LYN", "MYD88", "GADD45B", "ATM", "NFKB1", "NFKB2", "NFKBIA", "IRAK4", "PIAS4", "PLAU"]})

    pyiomica.ExportEnrichmentReport(analysisGOAssociation, AppendString='analysisGOAssociation', OutputDirectory=EnrichmentOutputDirectory + 'GOAnalysis/')

    #The data can be computed with or without a label. If labeled, the gene ID must be the first element for each ID provided. The data is in the form {ID,label}:
    analysisGOLabel = pyiomica.GOAnalysis([["C6orf57", "Protein"], ["CD46", "Protein"], ["DHX58", "Protein"], ["HMGB3", "Protein"], ["MAP3K5", "Protein"], 
                                            ["NFKB2", "Protein"], ["NOS2", "Protein"], ["PYCARD", "Protein"], ["PYDC1","Protein"], ["SSC5D", "Protein"]])

    pyiomica.ExportEnrichmentReport(analysisGOLabel, AppendString='analysisGOLabel', OutputDirectory=EnrichmentOutputDirectory + 'GOAnalysis/')

    #The data can be mixed, e.g. proteins and RNA with different labels:
    analysisGOMixed = pyiomica.GOAnalysis([["C6orf57", "Protein"], ["CD46", "Protein"], ["DHX58", "Protein"], ["HMGB3", "RNA"], ["HMGB3", "Protein"], ["MAP3K5", "Protein"], 
                                            ["NFKB2", "RNA"], ["NFKB2", "Protein"], ["NOS2", "RNA"], ["PYCARD", "RNA"], ["PYDC1", "Protein"], ["SSC5D", "Protein"]])

    pyiomica.ExportEnrichmentReport(analysisGOMixed, AppendString='analysisGOMixed', OutputDirectory=EnrichmentOutputDirectory + 'GOAnalysis/')

    #We can instead treat the data as different by setting the MultipleList and MultipleListCorrection options:
    analysisGOMixedMulti = pyiomica.GOAnalysis([["C6orf57", "Protein"], ["CD46", "Protein"], ["DHX58", "Protein"], ["HMGB3", "RNA"], ["HMGB3", "Protein"], ["MAP3K5", "Protein"], 
                                                ["NFKB2", "RNA"], ["NFKB2", "Protein"], ["NOS2", "RNA"], ["PYCARD", "RNA"], ["PYDC1", "Protein"], ["SSC5D", "Protein"]],
                                                MultipleList=True, MultipleListCorrection='Automatic')

    pyiomica.ExportEnrichmentReport(analysisGOMixedMulti, AppendString='analysisGOMixedMulti', OutputDirectory=EnrichmentOutputDirectory + 'GOAnalysis/')

    #Let's consider an example from real protein data. We will use already clustered data, from the examples. Let's import the data:
    ExampleClusteringObject = pyiomica.read(os.path.join(pyiomica.ConstantPyIOmicaExamplesDirectory, 'exampleClusteringObject_SLV_Delta_LAG1_Autocorr'))

    if not ExampleClusteringObject is None:
        #We calculate the GOAnalysis for each group in each class:
        ExampleClusteringObjectGO = pyiomica.GOAnalysis(ExampleClusteringObject, MultipleListCorrection='Automatic')

        pyiomica.ExportEnrichmentReport(ExampleClusteringObjectGO, AppendString='ExampleClusteringObjectGO', OutputDirectory=EnrichmentOutputDirectory + 'GOAnalysis/')

    return

## Examples of KEGG Analysis ##############################################################################################################################################
def testKEGGAnalysis(EnrichmentOutputDirectory):

    #Let's do a KEGG pathway analysis for a group of genes (most in the NFKB pathway), annotated with their "Gene Symbol":
    keggExample1 = pyiomica.KEGGAnalysis(["TAB1", "TNFSF13B", "MALT1", "TIRAP", "CHUK", "TNFRSF13C", "PARP1", "CSNK2A1", "CSNK2A2", "CSNK2B", "LTBR", "LYN", "MYD88", 
                                            "GADD45B", "ATM", "NFKB1", "NFKB2", "NFKBIA", "IRAK4", "PIAS4", "PLAU", "POLR3B", "NME1", "CTPS1", "POLR3A"])

    pyiomica.ExportEnrichmentReport(keggExample1, AppendString='keggExample1', OutputDirectory=EnrichmentOutputDirectory + 'KEGGAnalysis/')

    #The information can be computed for multiple groups, if these are provided as an association:
    analysisKEGGAssociation = pyiomica.KEGGAnalysis({"Group1": ["C6orf57", "CD46", "DHX58", "HMGB3", "MAP3K5", "NFKB2", "NOS2", "PYCARD", "PYDC1", "SSC5D"], 
                                                        "Group2": ["TAB1", "TNFSF13B", "MALT1", "TIRAP", "CHUK", "TNFRSF13C", "PARP1", "CSNK2A1", "CSNK2A2", "CSNK2B", "LTBR", 
                                                    "LYN", "MYD88", "GADD45B", "ATM", "NFKB1", "NFKB2", "NFKBIA", "IRAK4", "PIAS4", "PLAU", "POLR3B", "NME1", "CTPS1", "POLR3A"]})
        
    pyiomica.ExportEnrichmentReport(analysisKEGGAssociation, AppendString='analysisKEGGAssociation', OutputDirectory=EnrichmentOutputDirectory + 'KEGGAnalysis/')

    #The data can be computed with or without a label. If labeled, the gene ID must be the first element for each ID provided. The data is in the form {ID,label}:
    analysisKEGGLabel = pyiomica.KEGGAnalysis([["C6orf57", "Protein"], ["CD46", "Protein"], ["DHX58", "Protein"], ["HMGB3", "Protein"], ["MAP3K5", "Protein"], 
                                                ["NFKB2", "Protein"], ["NOS2", "Protein"], ["PYCARD", "Protein"], ["PYDC1","Protein"], ["SSC5D", "Protein"]])

    pyiomica.ExportEnrichmentReport(analysisKEGGLabel, AppendString='analysisKEGGLabel', OutputDirectory=EnrichmentOutputDirectory + 'KEGGAnalysis/')

    #The same result is obtained if IDs are enclosed in list brackets:
    analysisKEGGNoLabel = pyiomica.KEGGAnalysis([["C6orf57"], ["CD46"], ["DHX58"], ["HMGB3"], ["MAP3K5"], ["NFKB2"], ["NOS2"], ["PYCARD"], ["PYDC1"], ["SSC5D"]])

    pyiomica.ExportEnrichmentReport(analysisKEGGNoLabel, AppendString='analysisKEGGNoLabel', OutputDirectory=EnrichmentOutputDirectory + 'KEGGAnalysis/')

    #The same result is obtained if IDs are input as strings:
    analysisKEGGstrings = pyiomica.KEGGAnalysis(["C6orf57", "CD46", "DHX58", "HMGB3", "MAP3K5", "NFKB2", "NOS2", "PYCARD", "PYDC1", "SSC5D"])

    pyiomica.ExportEnrichmentReport(analysisKEGGstrings, AppendString='analysisKEGGstrings', OutputDirectory=EnrichmentOutputDirectory + 'KEGGAnalysis/')

    #The data can be mixed, e.g. proteins and RNA with different labels:
    analysisKEGGMixed = pyiomica.KEGGAnalysis([["C6orf57", "Protein"], ["CD46", "Protein"], ["DHX58", "Protein"], ["HMGB3", "RNA"], ["HMGB3", "Protein"], ["MAP3K5", "Protein"], 
                                                ["NFKB2", "RNA"], ["NFKB2", "Protein"], ["NOS2", "RNA"], ["PYCARD", "RNA"], ["PYDC1", "Protein"], ["SSC5D", "Protein"]])

    pyiomica.ExportEnrichmentReport(analysisKEGGMixed, AppendString='analysisKEGGMixed', OutputDirectory=EnrichmentOutputDirectory + 'KEGGAnalysis/')

    #The data in this case treated as originating from a single population. Protein and RNA labeled data for the same identifier are treated as equivalent.
    #We can instead treat the data as different by setting the MultipleList and MultipleListCorrection options:
    analysisKEGGMixedMulti = pyiomica.KEGGAnalysis([["C6orf57", "Protein"], ["CD46", "Protein"], ["DHX58", "Protein"], ["HMGB3", "RNA"], ["HMGB3", "Protein"], ["MAP3K5", "Protein"], 
                                                    ["NFKB2", "RNA"], ["NFKB2", "Protein"], ["NOS2", "RNA"], ["PYCARD", "RNA"], ["PYDC1", "Protein"], ["SSC5D", "Protein"]], 
                                                    MultipleList=True, MultipleListCorrection='Automatic')

    pyiomica.ExportEnrichmentReport(analysisKEGGMixedMulti, AppendString='analysisKEGGMixedMulti', OutputDirectory=EnrichmentOutputDirectory + 'KEGGAnalysis/')

    #We can carry out a "Molecular" analysis for compound data. We consider the following metabolomics data, which has labels "Meta" 
    #and additional mass and retention time information in the form {identifier,mass, retention time, label}:
    compoundsExample = [["cpd:C19691", 325.2075, 10.677681, "Meta"], ["cpd:C17905", 594.2002, 8.727458, "Meta"],["cpd:C09921", 204.0784, 12.3909445, "Meta"], 
                        ["cpd:C18218", 272.2356, 13.473582, "Meta"], ["cpd:C14169", 235.1573, 12.267084, "Meta"],["cpd:C14245", 262.2296, 13.545572, "Meta"], 
                        ["cpd:C09137", 352.2615, 14.0554285, "Meta"], ["cpd:C09674", 296.1624, 12.147417, "Meta"], ["cpd:C00449", 276.1334, 11.004139, "Meta"], 
                        ["cpd:C02999", 364.1497, 12.147243, "Meta"], ["cpd:C07915", 309.194, 7.3625283, "Meta"],["cpd:C08760", 496.2309, 8.7241125, "Meta"], 
                        ["cpd:C14549", 276.0972, 11.078914, "Meta"], ["cpd:C20533", 601.3378, 12.75722, "Meta"], ["cpd:C20790", 212.1051, 7.127666, "Meta"], 
                        ["cpd:C09137", 352.2613, 12.869867, "Meta"], ["cpd:C17648", 400.2085, 10.843841, "Meta"], ["cpd:C07807", 240.1471, 0.48564285, "Meta"], 
                        ["cpd:C08564", 324.0948, 10.281, "Meta"], ["cpd:C19426", 338.2818, 13.758765, "Meta"], ["cpd:C02943", 468.3218, 14.263261, "Meta"], 
                        ["cpd:C04882", 1193.342, 14.707576, "Meta"]]

    compoundsExampleKEGG = pyiomica.KEGGAnalysis(compoundsExample, FilterSignificant=True, AnalysisType='Molecular')

    pyiomica.ExportEnrichmentReport(compoundsExampleKEGG, AppendString='compoundsExampleKEGG', OutputDirectory=EnrichmentOutputDirectory + 'KEGGAnalysis/')

    #We can carry out multiomics data analysis. We consider the following simple example:
    multiOmicsData = [["C6orf57", "Protein"], ["CD46", "Protein"], ["DHX58", "Protein"], ["HMGB3", "RNA"], ["HMGB3", "Protein"], 
                        ["MAP3K5", "Protein"], ["NFKB2", "RNA"], ["NFKB2", "Protein"], ["NOS2", "RNA"], ["PYCARD", "RNA"], ["PYDC1", "Protein"], 
                        ["SSC5D", "Protein"], ["cpd:C19691", 325.2075, 10.677681, "Meta"], ["cpd:C17905", 594.2002, 8.727458, "Meta"], 
                        ["cpd:C09921", 204.0784, 12.3909445, "Meta"], ["cpd:C18218", 272.2356, 13.473582, "Meta"], 
                        ["cpd:C14169", 235.1573, 12.267084, "Meta"], ["cpd:C14245", 262.2296, 13.545572, "Meta"], 
                        ["cpd:C09137", 352.2615, 14.0554285, "Meta"], ["cpd:C09674", 296.1624, 12.147417, "Meta"],
                        ["cpd:C00449", 276.1334, 11.004139, "Meta"],["cpd:C02999", 364.1497, 12.147243, "Meta"],
                        ["cpd:C07915", 309.194, 7.3625283, "Meta"],["cpd:C08760", 496.2309, 8.7241125, "Meta"],
                        ["cpd:C14549", 276.0972, 11.078914, "Meta"],["cpd:C20533", 601.3378, 12.75722, "Meta"], 
                        ["cpd:C20790", 212.1051, 7.127666, "Meta"], ["cpd:C09137", 352.2613, 12.869867, "Meta"],
                        ["cpd:C17648", 400.2085, 10.843841, "Meta"], ["cpd:C07807", 240.1471, 0.48564285, "Meta"], 
                        ["cpd:C08564", 324.0948, 10.281, "Meta"], ["cpd:C19426", 338.2818, 13.758765, "Meta"], 
                        ["cpd:C02943", 468.3218, 14.263261, "Meta"], ["cpd:C04882", 1193.342, 14.707576, "Meta"]]

    #We can carry out "Genomic" and "Molecular" analysis concurrently by setting AnalysisType = "All":
    multiOmicsDataKEGG = pyiomica.KEGGAnalysis(multiOmicsData, AnalysisType='All', MultipleList=True, MultipleListCorrection='Automatic') 

    pyiomica.ExportEnrichmentReport(multiOmicsDataKEGG, AppendString='multiOmicsDataKEGG', OutputDirectory=EnrichmentOutputDirectory + 'KEGGAnalysis/')

    #Let's consider an example from real protein data. We will use already clustered data, from the examples. Let's import the data:
    pyiomica.read(os.path.join(pyiomica.ConstantPyIOmicaExamplesDirectory, 'exampleClusteringObject_SLV_Delta_LAG1_Autocorr'))

    if not ExampleClusteringObject is None:
        #We calculate the KEGGAnalysis for each group in each class:
        ExampleClusteringObject = pyiomica.KEGGAnalysis(ExampleClusteringObject)

        pyiomica.ExportEnrichmentReport(ExampleClusteringObject, AppendString='ExampleClusteringObject', OutputDirectory=EnrichmentOutputDirectory + 'KEGGAnalysis/')

    return