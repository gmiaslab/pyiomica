#import sys
#sys.path.append("../..")

import pyiomica as pio
from pyiomica.enrichmentAnalyses import ReactomeAnalysis, ExportReactomeEnrichmentReport
from pyiomica import dataStorage as ds

EnrichmentOutputDirectory = pio.os.path.join('results', 'EnrichmentOutputDirectory', '')

#Let's do a Reactome analysis for a group of genes, annotated with their "Gene Symbol":
ReactomeExample1 = ReactomeAnalysis(["TAB1", "TNFSF13B", "MALT1", "TIRAP", "CHUK", 
                        "TNFRSF13C", "PARP1", "CSNK2A1", "CSNK2A2", "CSNK2B", "LTBR", 
                        "LYN", "MYD88", "GADD45B", "ATM", "NFKB1", "NFKB2", "NFKBIA", 
                        "IRAK4", "PIAS4", "PLAU"])

ExportReactomeEnrichmentReport(ReactomeExample1, 
                               AppendString='ReactomeExample1', 
                               OutputDirectory=EnrichmentOutputDirectory + 'ReactomeAnalysis/')

#The information can be computed for multiple groups, if these are provided as an association:
analysisReactomeAssociation = ReactomeAnalysis({"Group1": ["C6orf57", "CD46", "DHX58", "HMGB3", "MAP3K5", "NFKB2", "NOS2", "PYCARD", "PYDC1", "SSC5D"], 
                                    "Group2": ["TAB1", "TNFSF13B", "MALT1", "TIRAP", "CHUK", "TNFRSF13C", "PARP1", "CSNK2A1", "CSNK2A2", "CSNK2B", "LTBR", "LYN", "MYD88", "GADD45B", "ATM", "NFKB1", "NFKB2", "NFKBIA", "IRAK4", "PIAS4", "PLAU"]})

ExportReactomeEnrichmentReport(analysisReactomeAssociation, 
                               AppendString='analysisReactomeAssociation', 
                               OutputDirectory=EnrichmentOutputDirectory + 'ReactomeAnalysis/')

#Let's consider an example from real experimental protein data. We will use already clustered data, from the examples. Let's import the data:
#ExampleClusteringObject = ds.read(pio.os.path.join(pio.ConstantPyIOmicaExamplesDirectory, 'exampleClusteringObject_SLV_Delta_LAG1_Autocorr'))
ExampleClusteringObject = ds.read('dev\\results\\SLV_Hourly1TimeSeries\\consolidatedGroupsSubgroups\\SLV_Hourly1TimeSeries_LAG3_Autocorrelations_GroupsSubgroups')

if not ExampleClusteringObject is None:
    # Input data is a clustering object
    ReactomeExampleClusteringObject = ReactomeAnalysis(ExampleClusteringObject)
    ExportReactomeEnrichmentReport(ReactomeExampleClusteringObject, 
                                   AppendString='ReactomeExampleClusteringObject', 
                                   OutputDirectory=EnrichmentOutputDirectory + 'ReactomeAnalysis/')

    # Input data is a part of a clustering object, i.e. a DataFrame
    ReactomeDataFrameEaxample = ReactomeAnalysis(ExampleClusteringObject[1][1]['data'])

    ExportReactomeEnrichmentReport(ReactomeDataFrameEaxample, 
                                   AppendString='ReactomeDataFrameEaxample', 
                                   OutputDirectory=EnrichmentOutputDirectory + 'ReactomeAnalysis/')
