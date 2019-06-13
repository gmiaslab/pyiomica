from pyiomica import pyiomica
from pyiomica import examples

import webbrowser
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(0)

if __name__ == '__main__':

    ####### Demo on GO and KEGG Annotations #######################################################################################################################################

    if False:
        examples.testGOAnalysis('EnrichmentOutputDirectory/')
        examples.testKEGGAnalysis('EnrichmentOutputDirectory/')



    ####### Demo on Time Series Classification ####################################################################################################################################

    if True:
        def processDataTempFunction(data_dir, dataFileName, timesFileName, saveDir, Delta=False):

            if Delta:
                dataName = 'SLV_Delta'
                df_dataH1 = pyiomica.read('results/results SLV H1/SLV_Hourly1TimeSeries_df_data_transformed', hdf5fileName='results/results SLV H1/SLV_Hourly1TimeSeries.h5')
                df_dataH2 = pyiomica.read('results/results SLV H2/SLV_Hourly2TimeSeries_df_data_transformed', hdf5fileName='results/results SLV H2/SLV_Hourly2TimeSeries.h5')
                df_data = pyiomica.compareTwoTimeSeriesDataframe(df_dataH2, df_dataH1, function=np.subtract, compareAllLevelsInIndex=False, mergeFunction=np.median)
                df_data[np.isnan(df_data)] = 0.0
            else:
                dataName = 'SLV' + '_' + dataFileName.split('_')[0]
                df_data = pyiomica.prepareDataframe(data_dir, dataFileName, timesFileName)
                df_data.index = pd.MultiIndex.from_tuples([(item.split(':')[1], 
                                                                item.split(':')[0].split('_')[0],
                                                                (' '.join(item.split(':')[0].split('_')[1:]),)) for item in df_data.index.values], 
                                                              names=['source', 'id', 'metadata'])
        
            pyiomica.timeSeriesClassification(df_data, dataName, saveDir, NumberOfRandomSamples = 10**5, NumberOfCPUs = 4, p_cutoff = 0.05, frequencyBasedClassification=False)
            pyiomica.timeSeriesClassification(df_data, dataName, saveDir, NumberOfRandomSamples = 10**5, NumberOfCPUs = 4, p_cutoff = 0.05, frequencyBasedClassification=True)

            pyiomica.visualizeTimeSeriesClassification(dataName, saveDir, AutocorrNotPeriodogr=True)
            pyiomica.visualizeTimeSeriesClassification(dataName, saveDir, AutocorrNotPeriodogr=False)

            return

        processDataTempFunction('data/SLV/', 'Hourly2TimeSeries_SLV_KallistoNormedGeneGencodeGC.csv', 'TimesHourly.csv', 'results/results SLV H2/')
        processDataTempFunction('data/SLV/', 'Hourly1TimeSeries_SLV_KallistoNormedGeneGencodeGC.csv', 'TimesHourly.csv', 'results/results SLV H1/')
        processDataTempFunction('data/SLV/', 'DailyTimeSeries_SLV_KallistoNormedGeneGencodeGC.csv',   'TimesDaily.csv',  'results/results SLV D1/')
        processDataTempFunction('data/SLV/', 'Hourly2TimeSeries_SLV_KallistoNormedGeneGencodeGC.csv', 'TimesHourly.csv', 'results/results SLV Delta/', Delta=True)



    ####### Demo on Visibility Graph ##############################################################################################################################################

    if False:
        exampleData = np.random.rand(30)
        exampleTimes = list(range(len(exampleData)))

        fig = plt.figure(figsize=(8,8))

        pyiomica.addVisibilityGraph(exampleData, exampleTimes, fig=fig, fontsize=16, nodesize=700, level=0.85, commLineWidth=3.0, lineWidth=2.0, withLabel=False)
        
        saveName = 'results/randomVG.svg'
        fig.savefig(saveName)
        
        webbrowser.open("file:///" + os.getcwd() + '\\' + saveName, new=2)
    