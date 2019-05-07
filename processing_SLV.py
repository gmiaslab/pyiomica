import pyiomica

import os
import timeMeasure

import numpy as np
import pandas as pd



def mathematica_round(values, N=6):

    def auxilary_round(value, N=6):
        exponent = np.ceil(np.log10(value))
        return 10**exponent*np.round(value*10**(-exponent), N)

    return np.frompyfunc(auxilary_round, 1, 1)(values)

if __name__ == '__main__':

    #dictionary = pyiomica.OBOGODictionary()





    np.random.seed(0)

    def processData(dataFileName, timesFileName, saveDir):
        data_dir = 'data/SLV/'
        dataName = 'SLV' + '_' + dataFileName.split('_')[0]

        if not os.path.exists(saveDir):
            os.makedirs(saveDir)

        NumberOfCPUs = 24

        NumberOfRandomSamples = 10**5
        p_cutoff = 0.05

        TestEstimate = False

        processRawData = True
        drawHistograms = False
        drawSamplePeriodograms = False
        calculateNullDistributionsAutocorrelations = True
        calculateAutocorrelations = True
        calculateSignificant = True
        DrawDendrogramHeatmap = True

        if TestEstimate:
            randomAutocorrelations = pyiomica.read(saveDir + dataName + '_randomAutocorrelations')
            print(randomAutocorrelations.shape)

            max_length = randomAutocorrelations.shape[0]

            QPs = []

            for i in range(10**3, max_length, 10**3):
                sel_index = np.random.choice(list(range(max_length)), size=i)
                QPs.append([np.quantile(randomAutocorrelations.values[sel_index].T[i], 1.-p_cutoff,interpolation='lower') for i in range(1,12)])
        
            QPs = np.array(QPs)

            print(QPs)

            np.savetxt('QPs.csv', QPs, delimiter=',', fmt='%1.16f')

            exit()

        if processRawData:
            df_data = pyiomica.prepareDataframe(data_dir, dataFileName, timesFileName)
            df_data = pyiomica.filterOutAllZeroSignalsDataframe(df_data)
            df_data = pyiomica.filterOutFirstPointZeroSignalsDataframe(df_data)
            df_data = pyiomica.filterOutFractionZeroSignalsDataframe(df_data, 0.75)
            df_data = pyiomica.tagMissingValuesDataframe(df_data)
            df_data = pyiomica.tagLowValuesDataframe(df_data, 1., 1.)
            df_data = pyiomica.removeConstantSignalsDataframe(df_data, 0.)

            if drawHistograms:
                sT = timeMeasure.getStartTime()
                print('\nPlotting raw data histogram for each time point...', end='\t', flush=True)
                pyiomica.makeDataHistograms(df_data, saveDir=saveDir + 'RawHistograms/', dataName=dataName)
                print('Done')
                timeMeasure.getElapsedTime(sT)

            if drawSamplePeriodograms:
                sT = timeMeasure.getStartTime()
                print('Plotting Lomb-Scargle periodograms for first 20 genes...', end='\t', flush=True)
                pyiomica.makeLombScarglePeriodograms(df_data[:20], saveDir=saveDir + 'LombScarglePeriodograms/', dataName=dataName)
                print('Done')
                timeMeasure.getElapsedTime(sT)

            pyiomica.write(df_data, saveDir + dataName + '_df_data_transformed')

        if calculateNullDistributionsAutocorrelations:
            df_data = pyiomica.read(saveDir + dataName + '_df_data_transformed')

            sT = timeMeasure.getStartTime()
            print('Calculating null distribution of %s samples...' % (NumberOfRandomSamples), end='\t', flush=True)
            randomAutocorrelations = pyiomica.getRandomAutocorrelations(df_data, NumberOfRandomSamples=NumberOfRandomSamples, NumberOfCPUs=NumberOfCPUs)
            print('Done')
            timeMeasure.getElapsedTime(sT)

            pyiomica.write(randomAutocorrelations, saveDir + dataName + '_randomAutocorrelations')

        if calculateAutocorrelations:
            df_data = pyiomica.read(saveDir + dataName + '_df_data_transformed')
            df_data = pyiomica.normalizeSignalsToUnityDataframe(df_data)

            sT = timeMeasure.getStartTime()
            print('Calculating each Time Series Autocorrelations...', end='\t', flush=True)
            dataAutocorrelations = pyiomica.runCPUs(NumberOfCPUs, pyiomica.getAutocorrelationsOfData, [(df_data.iloc[i], df_data.columns.values) for i in range(len(df_data.index))])
            print('Done')
            timeMeasure.getElapsedTime(sT)

            dataAutocorrelations = pd.DataFrame(data=dataAutocorrelations[1::2], index=df_data.index, columns=dataAutocorrelations[0])
            pyiomica.write(dataAutocorrelations, saveDir + dataName + '_dataAutocorrelations')

        if calculateSignificant:
            dataAutocorrelations = pyiomica.read(saveDir + dataName + '_dataAutocorrelations')
            randomAutocorrelations = pyiomica.read(saveDir + dataName + '_randomAutocorrelations')

            df_data = pyiomica.read(saveDir + dataName + '_df_data_transformed')
            
            QM = [1.0,-0.006244847659959638, 0.5734236884110672, 0.20792126175804138, 0.4493948421905076, 0.28343882235267753,
                 0.37888036168064276, 0.2823468480136749, 0.3295401814590595, 0.2551744666336719, 0.2912304011408658, 0.22097851418737666]
            print('Quantiles MathIOmica:', list(np.round(QM, 16)), '\n')

            QP = [1.0]; QP.extend([np.quantile(randomAutocorrelations.values.T[i], 1.-p_cutoff,interpolation='lower') for i in range(1,12)])
            print('Quantiles PyIOmica:', list(np.round(QP, 16)), '\n')

            significant_index = np.vstack([dataAutocorrelations.values.T[lag]>QP[lag] for lag in range(dataAutocorrelations.shape[1])]).T





            spike_cutoffs = {21:(0.9974359066568781, -0.27324957752788553), 20:(0.997055978650893, -0.2819888033231614), 
                            22:(0.9977278664058562, -0.26381407888037633), 19:(0.9972552653042678, -0.2918064323591995), 23:(0.9979375307710489, -0.25487050078876056), 
                            18:(0.9975598037852295, -0.3033004215269215), 24:(0.9974537417128844, -0.23810906765712997)}

            spike_cutoffs = pyiomica.getSpikesCutoffs(df_data, p_cutoff, NumberOfRandomSamples=NumberOfRandomSamples)
            [print(i, spike_cutoffs[i]) for i in range(18,25)]

            df_data = pyiomica.normalizeSignalsToUnityDataframe(df_data)

            max_spikes = df_data.index.values[pyiomica.getSpikes(df_data.values, np.max, spike_cutoffs)]
            min_spikes = df_data.index.values[pyiomica.getSpikes(df_data.values, np.min, spike_cutoffs)]

            print(len(max_spikes))
            print(len(min_spikes))
            
            significant_index_spike_max = [(gene in max_spikes) for gene in df_data.index.values]
            significant_index_spike_min = [(gene in min_spikes) for gene in df_data.index.values]

            lagSignigicantIndexSpikeMax = (np.sum(significant_index.T[1:],axis=0)==0) * significant_index_spike_max
            lagSignigicantIndexSpikeMin = (np.sum(significant_index.T[1:],axis=0)==0) * (np.array(significant_index_spike_max)==0)* significant_index_spike_min




            def getThisData(sheet_name):
                df_processed = pd.read_excel('ProcessedData/ClustersSLVRNAH%s.xlsx'%(saveDir[-2]), sheet_name=sheet_name)
                listThisLag = [];
                for i in range(df_processed.values.shape[1]):
                    listThisLag.extend(df_processed[df_processed.columns[i]])
                thisLagData = np.unique(listThisLag)[np.where((np.unique(listThisLag))!='nan')]
                thisLag = pd.DataFrame(data=thisLagData).apply(lambda val: np.frompyfunc(lambda data: data.strip('}').strip('{').strip('"').replace('", "','_') if data==data else data, 1, 1)(val), axis=0).values
                return thisLag

            thisSpikeMax = getThisData('SpikeMax')
            thisSpikeMin = getThisData('SpikeMin')

            for lag in range(1,dataAutocorrelations.shape[1]):
                lagSignigicantIndex = (np.sum(significant_index.T[1:lag],axis=0)==0) * (significant_index.T[lag])

                #dataAutocorrelations[lagSignigicantIndex].to_excel(saveDir + dataName + '_selectedAutocorrelations_LAG%s_%s.xlsx'%(lag,p_cutoff))
                #df_data[lagSignigicantIndex].to_excel(saveDir + dataName + '_selectedTimeSeries_LAG%s_%s.xlsx'%(lag,p_cutoff))

                thisLag = getThisData('Lag%s'%lag)
                print('\nLag',lag, np.array(list(set(df_data.index.values[lagSignigicantIndex]) - set(thisLag.T[0]))),
                      np.array(list(set(thisLag.T[0]) - set(df_data.index.values[lagSignigicantIndex]))))
                print('SD:', len(df_data.index.values[lagSignigicantIndex]), '\tGM:', len(thisLag), '\tcommon:',len(np.intersect1d(thisLag, df_data.index.values[lagSignigicantIndex])))


            print('\nSpikeMax', np.array(list(set(thisSpikeMax.T[0]) - set(df_data.index.values[lagSignigicantIndexSpikeMax]))),
                 np.array(list(set(df_data.index.values[lagSignigicantIndexSpikeMax]) - set(thisSpikeMax.T[0]))))

            print('SD:', len(df_data.index.values[lagSignigicantIndexSpikeMax]), '\t\tGM:', len(thisSpikeMax), '\t\tcommon:',len(np.intersect1d(thisSpikeMax, df_data.index.values[lagSignigicantIndexSpikeMax])))

            print('\nSpikeMin', np.array(list(set(thisSpikeMin.T[0]) - set(df_data.index.values[lagSignigicantIndexSpikeMin]))),
                  np.array(list(set(df_data.index.values[lagSignigicantIndexSpikeMin])  - set(thisSpikeMin.T[0]))))
            print('SD:', len(df_data.index.values[lagSignigicantIndexSpikeMin]), '\tGM:', len(thisSpikeMin), '\tcommon:',len(np.intersect1d(thisSpikeMin, df_data.index.values[lagSignigicantIndexSpikeMin])))
                
            print('Done')

        if DrawDendrogramHeatmap:
            for lag in range(1,11+1):
                df_LAG_data = pd.read_excel(saveDir + dataName + '_selectedTimeSeries_LAG%s_%s.xlsx'%(lag,p_cutoff))
                df_LAG_data = pyiomica.normalizeSignalsToUnityDataframe(df_LAG_data)

                df_LAG_data_autocor = pd.read_excel(saveDir + dataName + '_selectedAutocorrelations_LAG%s_%s.xlsx'%(lag,p_cutoff))

                print('Lag %s # of Time Series:'%lag, len(df_LAG_data))

                sT = timeMeasure.getStartTime()
                print('Plotting Dendrogram with Heatmaps of Gene expression and its Autocorrelation...', end='\t', flush=True)
                pyiomica.makeDendrogramHeatmap(df_LAG_data.values, df_LAG_data.columns, df_LAG_data_autocor.values, saveDir=saveDir, dataName=dataName, lag=lag)
                print('Done')
                timeMeasure.getElapsedTime(sT)

        return

    #processData('Hourly1TimeSeries_SLV_KallistoNormedGeneGencodeGC.csv', 'TimesHourly.csv', 'results SLV H1/')
    processData('Hourly2TimeSeries_SLV_KallistoNormedGeneGencodeGC.csv', 'TimesHourly.csv', 'results SLV H2/')
    #processData('DailyTimeSeries_SLV_KallistoNormedGeneGencodeGC.csv', 'TimesDaily.csv', 'results SLV D1/')
