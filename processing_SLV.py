from pyiomica import pyiomica

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

    np.random.seed(0)

    testAnnotations = False

    if testAnnotations:
        #Example of a clustering object
        dataC = pyiomica.read('cObject')

        uniProtExampleIDs = ["Q6ZRT9", "Q6NZ36", "H7C361", "Q6ZRT9", "A8MQT6", "Q9BUW7", "Q6NZ67", "Q6P582", "P39019", "E9PM41", "A8MTZ0", 
                            "A8MTZ0", "E9PRI7", "A8MTZ0", "Q9H6L5", "Q5H9J7", "Q5H9J7", "Q5H9J7", "P06454", "Q53S24", "B8ZZW7", "A0PJW6"]

        ExampleDict = {'AD':["TAB1", "TNFSF13B", "MALT1", "TIRAP", "CHUK", "TNFRSF13C", "PARP1", "CSNK2A1", "CSNK2A2", "CSNK2B", 
                             "LTBR", "LYN", "MYD88", "GADD45B", "ATM", "NFKB1", "NFKB2", "NFKBIA", "IRAK4", "PIAS4", "PLAU"]}

        ExampleProtein = [["C6orf57","Protein"],["CD46","Protein"],["DHX58","Protein"],["HMGB3","RNA"],["HMGB3","Protein"],["MAP3K5","Protein"],
                                    ["NFKB2","RNA"],["NFKB2","Protein"],["NOS2","RNA"],["PYCARD","RNA"],["PYDC1","Protein"],["SSC5D","Protein"]]

        ExampleMixed = [["C6orf57","Protein"],["CD46","Protein"],["DHX58","Protein"],["HMGB3","RNA"],["HMGB3","Protein"],["MAP3K5","Protein"],
                                    ["NFKB2","RNA"],["NFKB2","Protein"],["NOS2","RNA"],["PYCARD","RNA"],["PYDC1","Protein"],["SSC5D","Protein"]]

        compoundsExample = [["cpd:C19691", 325.2075, 10.677681, "Meta"], ["cpd:C17905", 594.2002, 8.727458, "Meta"],
                            ["cpd:C09921", 204.0784, 12.3909445, "Meta"], ["cpd:C18218", 272.2356, 13.473582, "Meta"],
                            ["cpd:C14169", 235.1573, 12.267084, "Meta"], ["cpd:C14245", 262.2296, 13.545572, "Meta"],
                            ["cpd:C09137", 352.2615, 14.0554285, "Meta"], ["cpd:C09674", 296.1624, 12.147417, "Meta"],
                            ["cpd:C00449", 276.1334, 11.004139, "Meta"], ["cpd:C02999", 364.1497, 12.147243, "Meta"],
                            ["cpd:C07915", 309.194, 7.3625283, "Meta"], ["cpd:C08760", 496.2309, 8.7241125, "Meta"],
                            ["cpd:C14549", 276.0972, 11.078914, "Meta"], ["cpd:C20533", 601.3378, 12.75722, "Meta"],
                            ["cpd:C20790", 212.1051, 7.127666, "Meta"], ["cpd:C09137", 352.2613, 12.869867, "Meta"],
                            ["cpd:C17648", 400.2085, 10.843841, "Meta"], ["cpd:C07807", 240.1471, 0.48564285, "Meta"],
                            ["cpd:C08564", 324.0948, 10.281, "Meta"], ["cpd:C19426", 338.2818, 13.758765, "Meta"],
                            ["cpd:C02943", 468.3218, 14.263261, "Meta"], ["cpd:C04882", 1193.342, 14.707576, "Meta"]]

        cObjectGO = pyiomica.GOAnalysis(dataC, MultipleListCorrection='Automatic')
        cObjectKEGG = pyiomica.KEGGAnalysis(dataC)
        G1S1 = pyiomica.GOAnalysis(ExampleG1S1)
        ResultUniProtExampleIDs = pyiomica.ReactomeAnalysis(uniProtExampleIDs)
        analysisGOExample1 = pyiomica.GOAnalysis(ExampleDict)
        analysisGOMixed_S = pyiomica.GOAnalysis(ExampleProtein)
        analysisGOMixed_M = pyiomica.GOAnalysis(ExampleMixed,
        MultipleListCorrection='Automatic', MultipleList=True)
        compoundsExampleMolecular = pyiomica.KEGGAnalysis(compoundsExample,AnalysisType='Molecular')
        compoundsExampleGenomic = pyiomica.KEGGAnalysis(compoundsExample,AnalysisType='Genomic')
        compoundsExampleAll = pyiomica.KEGGAnalysis(compoundsExample,AnalysisType='All')

    def processData(dataFileName, timesFileName, saveDir):
        data_dir = 'data/SLV/'
        dataName = 'SLV' + '_' + dataFileName.split('_')[0]

        if not os.path.exists(saveDir):
            os.makedirs(saveDir)

        NumberOfCPUs = 4

        NumberOfRandomSamples = 10 ** 5
        p_cutoff = 0.05

        processRawData = False
        drawHistograms = False
        drawSamplePeriodograms = False
        calculateNullDistributionsAutocorrelations = False
        calculateAutocorrelations = False
        calculateSignificant = False
        DrawDendrogramHeatmap = True

        if processRawData:
            df_data = pyiomica.prepareDataframe(data_dir, dataFileName, timesFileName)

            df_data.index = pd.MultiIndex.from_tuples([(item.split(':')[1], 
                                                        item.split(':')[0].split('_')[0],
                                                        ' '.join(item.split(':')[0].split('_')[1:])) for item in df_data.index.values], 
                                                      names=['source', 'gene', 'info'])

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

            QP = [1.0]
            QP.extend([np.quantile(randomAutocorrelations.values.T[i], 1. - p_cutoff,interpolation='lower') for i in range(1,12)])
            print('Quantiles PyIOmica:', list(np.round(QP, 16)), '\n')

            significant_index = np.vstack([dataAutocorrelations.values.T[lag] > QP[lag] for lag in range(dataAutocorrelations.shape[1])]).T


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

            significant_index_spike_max = [(gene in list(max_spikes)) for gene in df_data.index.values]
            significant_index_spike_min = [(gene in list(min_spikes)) for gene in df_data.index.values]

            lagSignigicantIndexSpikeMax = (np.sum(significant_index.T[1:],axis=0) == 0) * significant_index_spike_max
            lagSignigicantIndexSpikeMin = (np.sum(significant_index.T[1:],axis=0) == 0) * (np.array(significant_index_spike_max) == 0) * significant_index_spike_min


            def getThisData(sheet_name):
                df_processed = pd.read_excel('ProcessedData/ClustersSLVRNAH%s.xlsx' % (saveDir[-2]), sheet_name=sheet_name)
                listThisLag = []
                for i in range(df_processed.values.shape[1]):
                    listThisLag.extend(df_processed[df_processed.columns[i]])
                thisLagData = np.unique(listThisLag)[np.where((np.unique(listThisLag)) != 'nan')]
                thisLag = pd.DataFrame(data=thisLagData).apply(lambda val: np.frompyfunc(lambda data: data.strip('}').strip('{').strip('"').replace('", "','_') if data == data else data, 1, 1)(val), axis=0).values
                return thisLag

            thisSpikeMax = getThisData('SpikeMax')
            thisSpikeMin = getThisData('SpikeMin')

            for lag in range(1,dataAutocorrelations.shape[1]):
                lagSignigicantIndex = (np.sum(significant_index.T[1:lag],axis=0) == 0) * (significant_index.T[lag])

                dataAutocorrelations[lagSignigicantIndex].to_excel(saveDir + dataName +'_selectedAutocorrelations_LAG%s_%s.xlsx'%(lag,p_cutoff))
                df_data[lagSignigicantIndex].to_excel(saveDir + dataName +'_selectedTimeSeries_LAG%s_%s.xlsx'%(lag,p_cutoff))

                #thisLag = getThisData('Lag%s' % lag)
                #print('\nLag',lag, np.array(list(set(df_data.index.values[lagSignigicantIndex]) - set(thisLag.T[0]))),
                #      np.array(list(set(thisLag.T[0]) - set(df_data.index.values[lagSignigicantIndex]))))
                #print('SD:', len(df_data.index.values[lagSignigicantIndex]), '\tGM:', len(thisLag), '\tcommon:',len(np.intersect1d(thisLag, df_data.index.values[lagSignigicantIndex])))


            #print('\nSpikeMax', np.array(list(set(thisSpikeMax.T[0]) - set(df_data.index.values[lagSignigicantIndexSpikeMax]))),
            #     np.array(list(set(df_data.index.values[lagSignigicantIndexSpikeMax]) - set(thisSpikeMax.T[0]))))

            #print('SD:', len(df_data.index.values[lagSignigicantIndexSpikeMax]), '\t\tGM:', len(thisSpikeMax), '\t\tcommon:',len(np.intersect1d(thisSpikeMax, df_data.index.values[lagSignigicantIndexSpikeMax])))

            #print('\nSpikeMin', np.array(list(set(thisSpikeMin.T[0]) - set(df_data.index.values[lagSignigicantIndexSpikeMin]))),
            #      np.array(list(set(df_data.index.values[lagSignigicantIndexSpikeMin]) - set(thisSpikeMin.T[0]))))
            #print('SD:', len(df_data.index.values[lagSignigicantIndexSpikeMin]), '\tGM:', len(thisSpikeMin), '\tcommon:',len(np.intersect1d(thisSpikeMin, df_data.index.values[lagSignigicantIndexSpikeMin])))
                
            print('Done')

        if DrawDendrogramHeatmap:
            for lag in range(1,11 + 1):
                sT = timeMeasure.getStartTime()
                print('Lag %s # of Time Series:' % lag, end=' ', flush=True) 

                df_LAG_data = pd.read_excel(saveDir + dataName + '_selectedTimeSeries_LAG%s_%s.xlsx' % (lag,p_cutoff), index_col=[0,1,2])
                print(len(df_LAG_data))

                df_LAG_data = pyiomica.normalizeSignalsToUnityDataframe(df_LAG_data)
                df_LAG_data_autocor = pd.read_excel(saveDir + dataName + '_selectedAutocorrelations_LAG%s_%s.xlsx' % (lag,p_cutoff), index_col=[0,1,2])
                df_LAG_data_autocor.columns = ['Lag ' + str(column) for column in df_LAG_data_autocor.columns]

                print('Creating clustering object...')
                cObject = pyiomica.makeClusteringObject(df_LAG_data, df_LAG_data_autocor)

                print('Exporting clustering object...')
                pyiomica.exportClusteringObject(cObject, saveDir + 'consolidatedGroupsSubgroups/', dataName + '_Lag_%s' % lag, includeData=True, includeAutocorr=True)

                print('Plotting Dendrogram with Heatmaps of Gene expression and its Autocorrelation...')
                pyiomica.makeDendrogramHeatmap(cObject, saveDir, dataName + '_Lag_%s' % lag)

                timeMeasure.getElapsedTime(sT)

        return

    processData('Hourly1TimeSeries_SLV_KallistoNormedGeneGencodeGC.csv', 'TimesHourly.csv', 'results SLV H1/')
    #processData('Hourly2TimeSeries_SLV_KallistoNormedGeneGencodeGC.csv', 'TimesHourly.csv', 'results SLV H2/')
