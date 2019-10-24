#import sys
#sys.path.append("../..")

import pyiomica as pio

from pyiomica import categorizationFunctions as cf
from pyiomica import dataStorage as ds
from pyiomica import enrichmentAnalyses as ea

if __name__ == '__main__':

    # Unzip example data
    with pio.zipfile.ZipFile(pio.os.path.join(pio.ConstantPyIOmicaExamplesDirectory, 'SLV.zip'), "r") as zipFile:
        zipFile.extractall(path=pio.ConstantPyIOmicaExamplesDirectory)

    # Name of the fisrt data set 
    dataName = 'DailyTimeSeries_SLV_Protein'

    # Define a directory name where results are be saved
    saveDir = pio.os.path.join('results', dataName, '')

    # Directory name where example data is (*.csv files)
    dataDir = pio.os.path.join(pio.ConstantPyIOmicaExamplesDirectory, 'SLV')

    # Read the example data into a DataFrame
    df_data = pio.pd.read_csv(pio.os.path.join(dataDir, dataName + '.csv'), index_col=[0,1], header=0)

    # Calculate time series categorization
    cf.calculateTimeSeriesCategorization(df_data, dataName, saveDir, NumberOfRandomSamples = 10**5, referencePoint=2, preProcessData=False)

    # Cluster the time series categorization results
    cf.clusterTimeSeriesCategorization(dataName, saveDir)

    # Make plots of the clustered time series categorization
    cf.visualizeTimeSeriesCategorization(dataName, saveDir)

    # Do enrichment GO and KEGG analysis on LAG1 results
    LAG1 = ds.read('results/DailyTimeSeries_SLV_Protein/consolidatedGroupsSubgroups/DailyTimeSeries_SLV_Protein_LAG1_Autocorrelations_GroupsSubgroups')
    ea.ExportEnrichmentReport(ea.GOAnalysis(LAG1), AppendString='GO_LAG1', OutputDirectory='results/DailyTimeSeries_SLV_Protein/')
    ea.ExportEnrichmentReport(ea.KEGGAnalysis(LAG1), AppendString='KEGG_LAG1', OutputDirectory='results/DailyTimeSeries_SLV_Protein/')
