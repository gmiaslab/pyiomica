from pyiomica import pyiomica
from pyiomica import examples

import os
import numpy as np
import pandas as pd

import json

np.random.seed(0)

if __name__ == '__main__':

    if True:
        examples.testGOAnalysis('EnrichmentOutputDirectory/')
        examples.testKEGGAnalysis('EnrichmentOutputDirectory/')

    def processDataTempFunction(data_dir, dataFileName, timesFileName, saveDir, Delta=False):

        if Delta:
            dataName = 'SLV_Delta'
            df_dataH1 = pyiomica.read('results SLV H1/SLV_Hourly1TimeSeries_df_data_transformed', hdf5fileName='results SLV H1/SLV_Hourly1TimeSeries.h5')
            df_dataH2 = pyiomica.read('results SLV H2/SLV_Hourly2TimeSeries_df_data_transformed', hdf5fileName='results SLV H2/SLV_Hourly2TimeSeries.h5')
            df_data = pyiomica.compareTwoTimeSeriesDataframe(df_dataH2, df_dataH1, function=np.subtract, compareAllLevelsInIndex=False, mergeFunction=np.median)
            df_data[np.isnan(df_data)] = 0.0
        else:
            dataName = 'SLV' + '_' + dataFileName.split('_')[0]
            df_data = pyiomica.prepareDataframe(data_dir, dataFileName, timesFileName)
            df_data.index = pd.MultiIndex.from_tuples([(item.split(':')[1], 
                                                            item.split(':')[0].split('_')[0],
                                                            (' '.join(item.split(':')[0].split('_')[1:]),)) for item in df_data.index.values], 
                                                          names=['source', 'id', 'metadata'])
        
        pyiomica.timeSeriesClassification(df_data, dataName, saveDir, NumberOfRandomSamples = 10**5, NumberOfCPUs = 4, p_cutoff = 0.05)

        pyiomica.visualizeTimeSeriesClassification(dataName, saveDir)

        return

    processDataTempFunction('data/SLV/', 'Hourly1TimeSeries_SLV_KallistoNormedGeneGencodeGC.csv', 'TimesHourly.csv', 'results SLV H1/')
    processDataTempFunction('data/SLV/', 'Hourly2TimeSeries_SLV_KallistoNormedGeneGencodeGC.csv', 'TimesHourly.csv', 'results SLV H2/')
    processDataTempFunction('data/SLV/', 'DailyTimeSeries_SLV_KallistoNormedGeneGencodeGC.csv',   'TimesDaily.csv',  'results SLV D1/')
    processDataTempFunction('data/SLV/', 'Hourly2TimeSeries_SLV_KallistoNormedGeneGencodeGC.csv', 'TimesHourly.csv', 'results SLV Delta/', Delta=True)
