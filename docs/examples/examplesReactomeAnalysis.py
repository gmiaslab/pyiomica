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

#Enrichment analysis function ReactomeAnalysis can be used with a clustering object.
#First run examples of use of categorization functions to generate clustering objects.
#Then run "results = ReactomeAnalysis(ds.read(pathToClusteringObjectOfInterest))",
#to calculate enrichment for each group in each class, and then export enrichment results to a file if necesary using ExportReactomeEnrichmentReport function.
