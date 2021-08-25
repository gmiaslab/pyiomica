'''Categorization functions'''


from .globalVariables import *

from . import (utilityFunctions,
              visualizationFunctions,
              extendedDataFrame,
              clusteringFunctions,
              coreFunctions,
              dataStorage)

from .extendedDataFrame import DataFrame


def calculateTimeSeriesCategorization(df_data, dataName, saveDir, hdf5fileName=None, p_cutoff=0.05, fraction=0.75, constantSignalsCutoff=0., lowValuesToTag=1., lowValuesToTagWith=1., NumberOfRandomSamples=10**5, NumberOfCPUs=4, referencePoint=0, autocorrelationBased=True, calculateAutocorrelations=False, calculatePeriodograms=False, preProcessData=True):
        
    """Time series classification.
    
    Parameters:
        df_data: pandas.DataFrame
            Data to process

        dataName: str
            Data name, e.g. "myData_1"

        saveDir: str
            Path of directories poining to data storage

        hdf5fileName: str, Default None
            Preferred hdf5 file name and location

        p_cutoff: float, Default 0.05
            Significance cutoff signals selection

        fraction: float, Default 0.75
            Fraction of non-zero point in a signal

        constantSignalsCutoff: float, Default 0.
            Parameter to consider a signal constant

        lowValuesToTag: float, Default 1.
            Values below this are considered low

        lowValuesToTagWith: float, Default 1.
            Low values to tag with

        NumberOfRandomSamples: int, Default 10**5
            Size of the bootstrap distribution to generate

        NumberOfCPUs: int, Default 4
            Number of processes allowed to use in calculations
            
        referencePoint: int, Default 0
            Reference point

        autocorrelationBased: boolean, Default True
            Whether Autocorrelation of Frequency based

        calculateAutocorrelations: boolean, Default False
            Whether to recalculate Autocorrelations

        calculatePeriodograms: boolean, Default False
            Whether to recalculate Periodograms

        preProcessData: boolean, Default True
            Whether to preprocess data, i.e. filter, normalize etc.

    Returns:
        None

    Usage:
        calculateTimeSeriesCategorization(df_data, dataName, saveDir)
    """

    print('\n', '-'*70, '\n\tProcessing %s (%s)'%(dataName, 'Periodograms' if not autocorrelationBased else 'Autocorrelations'), '\n', '-'*70)

    if not os.path.exists(saveDir):
        os.makedirs(saveDir)

    if hdf5fileName is None:
        hdf5fileName = saveDir + dataName + '.h5'

    df_data = extendedDataFrame.DataFrame(df_data)

    df_data.columns = df_data.columns.astype(float)

    if preProcessData:
        df_data.filterOutAllZeroSignals(inplace=True)
        df_data.filterOutReferencePointZeroSignals(referencePoint=referencePoint, inplace=True)
        df_data.filterOutFractionZeroSignals(fraction, inplace=True)
        df_data.tagValueAsMissing(inplace=True)
        df_data.tagLowValues(lowValuesToTag, lowValuesToTagWith, inplace=True)
        df_data.removeConstantSignals(constantSignalsCutoff, inplace=True)

    dataStorage.write(df_data, saveDir + dataName + '_df_data_transformed', hdf5fileName=hdf5fileName)

    if not autocorrelationBased:
        calculateAutocorrelations = False
        if not calculatePeriodograms:
            df_dataPeriodograms = dataStorage.read(saveDir + dataName + '_dataPeriodograms', hdf5fileName=hdf5fileName)
            df_randomPeriodograms = dataStorage.read(saveDir + dataName + '_randomPeriodograms', hdf5fileName=hdf5fileName)
        
            if (df_dataPeriodograms is None) or (df_randomPeriodograms is None):
                print('Periodograms of data and the corresponding null distribution not found. Calculating...')
                calculatePeriodograms = True
    else:
        calculatePeriodograms = False
        if not calculateAutocorrelations:
            df_dataAutocorrelations = dataStorage.read(saveDir + dataName + '_dataAutocorrelations', hdf5fileName=hdf5fileName)
            df_randomAutocorrelations = dataStorage.read(saveDir + dataName + '_randomAutocorrelations', hdf5fileName=hdf5fileName)
        
            if (df_dataAutocorrelations is None) or (df_randomAutocorrelations is None):
                print('Autocorrelation of data and the corresponding null distribution not found. Calculating...')
                calculateAutocorrelations = True

    if calculatePeriodograms:
        df_data = dataStorage.read(saveDir + dataName + '_df_data_transformed', hdf5fileName=hdf5fileName)

        print('Calculating null distribution (periodogram) of %s samples...' %(NumberOfRandomSamples))
        df_randomPeriodograms = extendedDataFrame.getRandomPeriodograms(df_data, NumberOfRandomSamples=NumberOfRandomSamples, NumberOfCPUs=NumberOfCPUs, fraction=fraction, referencePoint=referencePoint)

        dataStorage.write(df_randomPeriodograms, saveDir + dataName + '_randomPeriodograms', hdf5fileName=hdf5fileName)

        df_data = dataStorage.read(saveDir + dataName + '_df_data_transformed', hdf5fileName=hdf5fileName)
        df_data = df_data.normalizeSignalsToUnity(referencePoint=referencePoint)

        print('Calculating each Time Series Periodogram...')
        df_dataPeriodograms = extendedDataFrame.getLombScarglePeriodogramOfDataframe(df_data)

        dataStorage.write(df_dataPeriodograms, saveDir + dataName + '_dataPeriodograms', hdf5fileName=hdf5fileName)

    if calculateAutocorrelations:
        df_data = dataStorage.read(saveDir + dataName + '_df_data_transformed', hdf5fileName=hdf5fileName)

        print('Calculating null distribution (autocorrelation) of %s samples...' %(NumberOfRandomSamples))
        df_randomAutocorrelations = extendedDataFrame.getRandomAutocorrelations(df_data, NumberOfRandomSamples=NumberOfRandomSamples, NumberOfCPUs=NumberOfCPUs, fraction=fraction, referencePoint=referencePoint)

        dataStorage.write(df_randomAutocorrelations, saveDir + dataName + '_randomAutocorrelations', hdf5fileName=hdf5fileName)

        df_data = dataStorage.read(saveDir + dataName + '_df_data_transformed', hdf5fileName=hdf5fileName)

        df_data = df_data.normalizeSignalsToUnity(referencePoint=referencePoint)

        print('Calculating each Time Series Autocorrelations...')
        df_dataAutocorrelations = utilityFunctions.runCPUs(NumberOfCPUs, coreFunctions.getAutocorrelationsOfData, [(df_data.iloc[i], df_data.columns.values) for i in range(len(df_data.index))])

        df_dataAutocorrelations = pd.DataFrame(data=df_dataAutocorrelations[1::2], index=df_data.index, columns=df_dataAutocorrelations[0])
        df_dataAutocorrelations.columns = ['Lag ' + str(columnID) for columnID in range(len(df_dataAutocorrelations.columns))]
        dataStorage.write(df_dataAutocorrelations, saveDir + dataName + '_dataAutocorrelations', hdf5fileName=hdf5fileName)

    df_data = dataStorage.read(saveDir + dataName + '_df_data_transformed', hdf5fileName=hdf5fileName)

    if not autocorrelationBased:
        df_classifier = df_dataPeriodograms
        df_randomClassifier = df_randomPeriodograms
        info = 'Periodograms'
    else:
        df_classifier = df_dataAutocorrelations
        df_randomClassifier = df_randomAutocorrelations
        info = 'Autocorrelations'

    df_classifier.sort_index(inplace=True)
    df_data.sort_index(inplace=True)

    if not (df_data.index.values == df_classifier.index.values).all():
        raise ValueError('Index mismatch')
            
    QP = [1.0]
    QP.extend([np.quantile(df_randomClassifier.values.T[i], 1. - p_cutoff,interpolation='lower') for i in range(1,df_classifier.shape[1])])
    print('Quantiles:', list(np.round(QP, 16)), '\n')

    significant_index = np.vstack([df_classifier.values.T[lag] > QP[lag] for lag in range(df_classifier.shape[1])]).T

    print('Calculating spikes cutoffs...')
    spike_cutoffs = extendedDataFrame.getRandomSpikesCutoffs(df_data, p_cutoff, NumberOfRandomSamples=NumberOfRandomSamples)
    print(spike_cutoffs)

    df_data = df_data.normalizeSignalsToUnity(referencePoint=referencePoint)

    if not (df_data.index.values == df_classifier.index.values).all():
        raise ValueError('Index mismatch')

    print('Recording SpikeMax data...')
    max_spikes = df_data.index.values[coreFunctions.getSpikes(df_data.values, np.max, spike_cutoffs)]
    print(len(max_spikes))
    significant_index_spike_max = [(gene in list(max_spikes)) for gene in df_data.index.values]
    lagSignigicantIndexSpikeMax = (np.sum(significant_index.T[1:],axis=0) == 0) * significant_index_spike_max
    dataStorage.write(df_classifier[lagSignigicantIndexSpikeMax], saveDir + dataName +'_selected%s_SpikeMax'%(info), hdf5fileName=hdf5fileName)
    dataStorage.write(df_data[lagSignigicantIndexSpikeMax], saveDir + dataName +'_selectedTimeSeries%s_SpikeMax'%(info), hdf5fileName=hdf5fileName)
            
    print('Recording SpikeMin data...')
    min_spikes = df_data.index.values[coreFunctions.getSpikes(df_data.values, np.min, spike_cutoffs)]
    print(len(min_spikes))
    significant_index_spike_min = [(gene in list(min_spikes)) for gene in df_data.index.values]
    lagSignigicantIndexSpikeMin = (np.sum(significant_index.T[1:],axis=0) == 0) * (np.array(significant_index_spike_max) == 0) * significant_index_spike_min
    dataStorage.write(df_classifier[lagSignigicantIndexSpikeMin], saveDir + dataName +'_selected%s_SpikeMin'%(info), hdf5fileName=hdf5fileName)
    dataStorage.write(df_data[lagSignigicantIndexSpikeMin], saveDir + dataName +'_selectedTimeSeries%s_SpikeMin'%(info), hdf5fileName=hdf5fileName)

    print('Recording Lag%s-Lag%s data...'%(1,df_classifier.shape[1]))
    for lag in range(1,df_classifier.shape[1]):
        lagSignigicantIndex = (np.sum(significant_index.T[1:lag],axis=0) == 0) * (significant_index.T[lag])
        dataStorage.write(df_classifier[lagSignigicantIndex], saveDir + dataName +'_selected%s_LAG%s'%(info,lag), hdf5fileName=hdf5fileName)
        dataStorage.write(df_data[lagSignigicantIndex], saveDir + dataName +'_selectedTimeSeries%s_LAG%s'%(info,lag), hdf5fileName=hdf5fileName)
                
    return None

def clusterTimeSeriesCategorization(dataName, saveDir, numberOfLagsToDraw=3, hdf5fileName=None, 
                                    exportClusteringObjects=False, writeClusteringObjectToBinaries=True, autocorrelationBased=True,
                                    method='weighted', metric='correlation', significance='Elbow'):

    """Visualize time series classification.
    
    Parameters:
        dataName: str
            Data name, e.g. "myData_1"

        saveDir: str
            Path of directories pointing to data storage

        numberOfLagsToDraw: int, Default 3
            First top-N lags (or frequencies) to draw

        hdf5fileName: str, Default None
            HDF5 storage path and name

        exportClusteringObjects: boolean, Default False
            Whether to export clustering objects to xlsx files

        writeClusteringObjectToBinaries: boolean, Default True
            Whether to export clustering objects to binary (pickle) files

        autocorrelationBased: boolean, Default True
            Whether to label to print on the plots

        method: str, Default 'weighted'
            Linkage calculation method

        metric: str, Default 'correlation'
            Distance measure

        significance: str, Default 'Elbow'
            Method for determining optimal number of groups and subgroups

    Returns:
        None

    Usage:
        clusterTimeSeriesClassification('myData_1', '/dir1/dir2/')
    """

    info = 'Autocorrelations' if autocorrelationBased else 'Periodograms'

    if hdf5fileName is None:
        hdf5fileName = saveDir + dataName + '.h5'

    def internal(className):
        print('\n\n%s of Time Series:'%(className)) 
        df_data_selected = dataStorage.read(saveDir + dataName + '_selectedTimeSeries%s_%s'%(info, className), hdf5fileName=hdf5fileName)
        df_classifier_selected = dataStorage.read(saveDir + dataName + '_selected%s_%s'%(info, className), hdf5fileName=hdf5fileName)

        if (df_data_selected is None) or (df_classifier_selected is None):

            print('Selected %s time series not found in %s.'%(className, saveDir + dataName + '.h5'))
            print('Do time series classification first.')

            return 

        print('Creating clustering object.')
        clusteringObject = clusteringFunctions.makeClusteringObject(df_data_selected, df_classifier_selected, method=method, metric=metric, significance=significance)

        if clusteringObject is None:
            print('Error creating clustering object')
            return

        print('Exporting clustering object.')
        if writeClusteringObjectToBinaries:
            dataStorage.write(clusteringObject, saveDir + 'consolidatedGroupsSubgroups/' + dataName + '_%s_%s'%(className,info) + '_GroupsSubgroups')
        
        if exportClusteringObjects:
            clusteringFunctions.exportClusteringObject(clusteringObject, saveDir + 'consolidatedGroupsSubgroups/', dataName + '_%s_%s'%(className,info))

        return

    for lag in range(1,numberOfLagsToDraw + 1):
        internal('LAG%s'%(lag))
            
    internal('SpikeMax')
    internal('SpikeMin')

    return None

def visualizeTimeSeriesCategorization(dataName, saveDir, numberOfLagsToDraw=3, autocorrelationBased=True,xLabel='Time', plotLabel='Transformed Expression',horizontal=False, minNumberOfCommunities=2, communitiesMethod='WDPVG', direction='left', weight='distance'):

    """Visualize time series classification.
    
    Parameters:
        dataName: str
            Data name, e.g. "myData_1"

        saveDir: str
            Path of directories pointing to data storage

        numberOfLagsToDraw: boolean, Default 3
            First top-N lags (or frequencies) to draw

        autocorrelationBased: boolean, Default True
            Whether autocorrelation or frequency based

        xLabel: str, Default 'Time'
            X-axis label

        plotLabel: str, Default 'Transformed Expression'
            Label for the heatmap plot

        
        horizontal: boolean, Default False
            Whether to use horizontal or natural visibility graph. 

        minNumberOfCommunities: int, Default 2
            Number of communities to find depends on the number of splits.
            This parameter is ignored in methods that automatically
            estimate optimal number of communities.

        communitiesMethod: str, Default 'WDPVG'
            String defining the method to use for communitiy detection:
                'Girvan_Newman': edge betweenness centrality based approach

                'betweenness_centrality': reflected graph node betweenness centrality based approach

                'WDPVG': weighted dual perspective visibility graph method (note to also set weight variable)

        direction:str, default 'left'
            The direction that nodes aggregate to communities:
                None: no specific direction, e.g. both sides.

                'left': nodes can only aggregate to the left side hubs, e.g. early hubs

                'right': nodes can only aggregate to the right side hubs, e.g. later hubs

        weight: str, Default 'distance'
            Type of weight for communitiesMethod='WDPVG':
                None: no weighted

                'time': weight = abs(times[i] - times[j])

                'tan': weight = abs((data[i] - data[j])/(times[i] - times[j])) + 10**(-8)

                'distance': weight = A[i, j] = A[j, i] = ((data[i] - data[j])**2 + (times[i] - times[j])**2)**0.5
    Returns:
        None

    Usage:
        visualizeTimeSeriesClassification('myData_1', '/dir1/dir2/')
    """

    info = 'Autocorrelations' if autocorrelationBased else 'Periodograms'

    def internal(className):
        print('\n\n%s of Time Series:'%(className)) 

        clusteringObject = dataStorage.read(saveDir + 'consolidatedGroupsSubgroups/' + dataName + '_%s_%s'%(className,info) + '_GroupsSubgroups')

        if clusteringObject is None:
            print('Clustering object not found')
            return
        if len(clusteringObject['linkage']) < 2:
            print('Clustering linkage array has only 1 row')
            return 
        
        print('Plotting Dendrogram with Heatmaps.')
        visualizationFunctions.makeDendrogramHeatmapOfClusteringObject(clusteringObject, saveDir, dataName + '_%s_%sBased'%(className,info), AutocorrNotPeriodogr=autocorrelationBased,xLabel=xLabel, plotLabel=plotLabel,horizontal=horizontal, minNumberOfCommunities=minNumberOfCommunities, communitiesMethod=communitiesMethod, direction=direction, weight=weight)

        return

    for lag in range(1,numberOfLagsToDraw + 1):
        internal('LAG%s'%(lag))
            
    internal('SpikeMax')
    internal('SpikeMin')

    return None
