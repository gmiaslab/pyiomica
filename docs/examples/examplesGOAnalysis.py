#import sys
#sys.path.append("../..")

import pyiomica as pio

from pyiomica.enrichmentAnalyses import GOAnalysis, ExportEnrichmentReport
from pyiomica import dataStorage as ds

EnrichmentOutputDirectory = 'results/EnrichmentOutputDirectory/'

#Let's do a GO analysis for a group of genes, annotated with their "Gene Symbol":
goExample1 = GOAnalysis(["TAB1", "TNFSF13B", "MALT1", "TIRAP", "CHUK", 
                                "TNFRSF13C", "PARP1", "CSNK2A1", "CSNK2A2", "CSNK2B", "LTBR", 
                                "LYN", "MYD88", "GADD45B", "ATM", "NFKB1", "NFKB2", "NFKBIA", 
                                "IRAK4", "PIAS4", "PLAU"])

ExportEnrichmentReport(goExample1, AppendString='goExample1', OutputDirectory=EnrichmentOutputDirectory + 'GOAnalysis/')

#The information can be computed for multiple groups, if these are provided as an association:
analysisGOAssociation = GOAnalysis({"Group1": ["C6orf57", "CD46", "DHX58", "HMGB3", "MAP3K5", "NFKB2", "NOS2", "PYCARD", "PYDC1", "SSC5D"], 
                                                "Group2": ["TAB1", "TNFSF13B", "MALT1", "TIRAP", "CHUK", "TNFRSF13C", "PARP1", "CSNK2A1", "CSNK2A2", "CSNK2B", "LTBR", 
                                                            "LYN", "MYD88", "GADD45B", "ATM", "NFKB1", "NFKB2", "NFKBIA", "IRAK4", "PIAS4", "PLAU"]})

ExportEnrichmentReport(analysisGOAssociation, AppendString='analysisGOAssociation', OutputDirectory=EnrichmentOutputDirectory + 'GOAnalysis/')

#The data can be computed with or without a label. If labeled, the gene ID must be the first element for each ID provided. The data is in the form {ID,label}:
analysisGOLabel = GOAnalysis([["C6orf57", "Protein"], ["CD46", "Protein"], ["DHX58", "Protein"], ["HMGB3", "Protein"], ["MAP3K5", "Protein"], 
                                        ["NFKB2", "Protein"], ["NOS2", "Protein"], ["PYCARD", "Protein"], ["PYDC1","Protein"], ["SSC5D", "Protein"]])

ExportEnrichmentReport(analysisGOLabel, AppendString='analysisGOLabel', OutputDirectory=EnrichmentOutputDirectory + 'GOAnalysis/')

#The data can be mixed, e.g. proteins and RNA with different labels:
analysisGOMixed = GOAnalysis([["C6orf57", "Protein"], ["CD46", "Protein"], ["DHX58", "Protein"], ["HMGB3", "RNA"], ["HMGB3", "Protein"], ["MAP3K5", "Protein"], 
                                        ["NFKB2", "RNA"], ["NFKB2", "Protein"], ["NOS2", "RNA"], ["PYCARD", "RNA"], ["PYDC1", "Protein"], ["SSC5D", "Protein"]])

ExportEnrichmentReport(analysisGOMixed, AppendString='analysisGOMixed', OutputDirectory=EnrichmentOutputDirectory + 'GOAnalysis/')

#We can instead treat the data as different by setting the MultipleList and MultipleListCorrection options:
analysisGOMixedMulti = GOAnalysis([["C6orf57", "Protein"], ["CD46", "Protein"], ["DHX58", "Protein"], ["HMGB3", "RNA"], ["HMGB3", "Protein"], ["MAP3K5", "Protein"], 
                                            ["NFKB2", "RNA"], ["NFKB2", "Protein"], ["NOS2", "RNA"], ["PYCARD", "RNA"], ["PYDC1", "Protein"], ["SSC5D", "Protein"]],
                                            MultipleList=True, MultipleListCorrection='Automatic')

ExportEnrichmentReport(analysisGOMixedMulti, AppendString='analysisGOMixedMulti', OutputDirectory=EnrichmentOutputDirectory + 'GOAnalysis/')

#Let's consider an example from real protein data. We will use already clustered data, from the examples. Let's import the data:
ExampleClusteringObject = ds.read(pio.os.path.join(pio.ConstantPyIOmicaExamplesDirectory, 'exampleClusteringObject_SLV_Delta_LAG1_Autocorr'))

if not ExampleClusteringObject is None:
    #We calculate the GOAnalysis for each group in each class:
    ExampleClusteringObjectGO = GOAnalysis(ExampleClusteringObject, MultipleListCorrection='Automatic')

    ExportEnrichmentReport(ExampleClusteringObjectGO, AppendString='ExampleClusteringObjectGO', OutputDirectory=EnrichmentOutputDirectory + 'GOAnalysis/')