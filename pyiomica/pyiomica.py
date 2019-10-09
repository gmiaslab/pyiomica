"""
PyIOmica is a general omics package with multiple tools for analyzing omics data.

Usage:
    from pyiomica import pyiomica

Notes:
    For additional information visit: https://github.com/gmiaslab/pyiomica and https://mathiomica.org by G. Mias Lab
"""

print("Loading PyIOmica (https://github.com/gmiaslab/pyiomica by G. Mias Lab)")


from .globalVariables import *
from .utilityFunctions import *
from .enrichmentAnalyses import *
from .visualizationFunctions import *
from .visibilityGraphAuxilaryFunctions import *
from .extendedDataFrame import *
from .clusteringFunctions import *
from .coreFunctions import *


def timeSeriesClassification(df_data, dataName, saveDir, hdf5fileName=None, p_cutoff=0.05,
                             NumberOfRandomSamples=10**5, NumberOfCPUs=4, frequencyBasedClassification=False, 
                             calculateAutocorrelations=False, calculatePeriodograms=False):
        
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

        NumberOfRandomSamples: int, Default 10**5
            Size of the bootstrap distribution to generate

        NumberOfCPUs: int, Default 4
            Number of processes allowed to use in calculations

        frequencyBasedClassification: boolean, Default False
            Whether Autocorrelation of Frequency based

        calculateAutocorrelations: boolean, Default False
            Whether to recalculate Autocorrelations

        calculatePeriodograms: boolean, Default False
            Whether to recalculate Periodograms

    Returns:
        None

    Usage:
        timeSeriesClassification(df_data, dataName, saveDir, NumberOfRandomSamples = 10**5, NumberOfCPUs = 4, p_cutoff = 0.05, frequencyBasedClassification=False)
    """

    print('\n', '-'*70, '\n\tProcessing %s (%s)'%(dataName, 'Periodograms' if frequencyBasedClassification else 'Autocorrelations'), '\n', '-'*70)

    if not os.path.exists(saveDir):
        os.makedirs(saveDir)

    if hdf5fileName is None:
        hdf5fileName = saveDir + dataName + '.h5'

    df_data = filterOutAllZeroSignalsDataframe(df_data)
    df_data = filterOutFirstPointZeroSignalsDataframe(df_data)
    df_data = filterOutFractionZeroSignalsDataframe(df_data, 0.75)
    df_data = tagMissingValuesDataframe(df_data)
    df_data = tagLowValuesDataframe(df_data, 1., 1.)
    df_data = removeConstantSignalsDataframe(df_data, 0.)

    write(df_data, saveDir + dataName + '_df_data_transformed', hdf5fileName=hdf5fileName)

    if frequencyBasedClassification:
        calculateAutocorrelations = False
        if not calculatePeriodograms:
            df_dataPeriodograms = read(saveDir + dataName + '_dataPeriodograms', hdf5fileName=hdf5fileName)
            df_randomPeriodograms = read(saveDir + dataName + '_randomPeriodograms', hdf5fileName=hdf5fileName)
        
            if (df_dataPeriodograms is None) or (df_randomPeriodograms is None):
                print('Periodograms of data and the corresponding null distribution not found. Calculating...')
                calculatePeriodograms = True
    else:
        calculatePeriodograms = False
        if not calculateAutocorrelations:
            df_dataAutocorrelations = read(saveDir + dataName + '_dataAutocorrelations', hdf5fileName=hdf5fileName)
            df_randomAutocorrelations = read(saveDir + dataName + '_randomAutocorrelations', hdf5fileName=hdf5fileName)
        
            if (df_dataAutocorrelations is None) or (df_randomAutocorrelations is None):
                print('Autocorrelation of data and the corresponding null distribution not found. Calculating...')
                calculateAutocorrelations = True

    if calculatePeriodograms:
        df_data = read(saveDir + dataName + '_df_data_transformed', hdf5fileName=hdf5fileName)

        print('Calculating null distribution (periodogram) of %s samples...' %(NumberOfRandomSamples))
        df_randomPeriodograms = getRandomPeriodograms(df_data, NumberOfRandomSamples=NumberOfRandomSamples, NumberOfCPUs=NumberOfCPUs)

        write(df_randomPeriodograms, saveDir + dataName + '_randomPeriodograms', hdf5fileName=hdf5fileName)

        df_data = read(saveDir + dataName + '_df_data_transformed', hdf5fileName=hdf5fileName)
        df_data = normalizeSignalsToUnityDataframe(df_data)

        print('Calculating each Time Series Periodogram...')
        df_dataPeriodograms = getLobmScarglePeriodogramOfDataframe(df_data)

        write(df_dataPeriodograms, saveDir + dataName + '_dataPeriodograms', hdf5fileName=hdf5fileName)

    if calculateAutocorrelations:
        df_data = read(saveDir + dataName + '_df_data_transformed', hdf5fileName=hdf5fileName)

        print('Calculating null distribution (autocorrelation) of %s samples...' %(NumberOfRandomSamples))
        df_randomAutocorrelations = getRandomAutocorrelations(df_data, NumberOfRandomSamples=NumberOfRandomSamples, NumberOfCPUs=NumberOfCPUs)

        write(df_randomAutocorrelations, saveDir + dataName + '_randomAutocorrelations', hdf5fileName=hdf5fileName)

        df_data = read(saveDir + dataName + '_df_data_transformed', hdf5fileName=hdf5fileName)
        df_data = normalizeSignalsToUnityDataframe(df_data)

        print('Calculating each Time Series Autocorrelations...')
        df_dataAutocorrelations = runCPUs(NumberOfCPUs, getAutocorrelationsOfData, [(df_data.iloc[i], df_data.columns.values) for i in range(len(df_data.index))])

        df_dataAutocorrelations = pd.DataFrame(data=df_dataAutocorrelations[1::2], index=df_data.index, columns=df_dataAutocorrelations[0])
        df_dataAutocorrelations.columns = ['Lag ' + str(column) for column in df_dataAutocorrelations.columns]
        write(df_dataAutocorrelations, saveDir + dataName + '_dataAutocorrelations', hdf5fileName=hdf5fileName)

    df_data = read(saveDir + dataName + '_df_data_transformed', hdf5fileName=hdf5fileName)

    if frequencyBasedClassification:
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
    spike_cutoffs = getSpikesCutoffs(df_data, p_cutoff, NumberOfRandomSamples=NumberOfRandomSamples)
    print(spike_cutoffs)

    df_data = normalizeSignalsToUnityDataframe(df_data)

    if not (df_data.index.values == df_classifier.index.values).all():
        raise ValueError('Index mismatch')

    print('Recording SpikeMax data...')
    max_spikes = df_data.index.values[getSpikes(df_data.values, np.max, spike_cutoffs)]
    print(len(max_spikes))
    significant_index_spike_max = [(gene in list(max_spikes)) for gene in df_data.index.values]
    lagSignigicantIndexSpikeMax = (np.sum(significant_index.T[1:],axis=0) == 0) * significant_index_spike_max
    write(df_classifier[lagSignigicantIndexSpikeMax], saveDir + dataName +'_selected%s_SpikeMax'%(info), hdf5fileName=hdf5fileName)
    write(df_data[lagSignigicantIndexSpikeMax], saveDir + dataName +'_selectedTimeSeries%s_SpikeMax'%(info), hdf5fileName=hdf5fileName)
            
    print('Recording SpikeMin data...')
    min_spikes = df_data.index.values[getSpikes(df_data.values, np.min, spike_cutoffs)]
    print(len(min_spikes))
    significant_index_spike_min = [(gene in list(min_spikes)) for gene in df_data.index.values]
    lagSignigicantIndexSpikeMin = (np.sum(significant_index.T[1:],axis=0) == 0) * (np.array(significant_index_spike_max) == 0) * significant_index_spike_min
    write(df_classifier[lagSignigicantIndexSpikeMin], saveDir + dataName +'_selected%s_SpikeMin'%(info), hdf5fileName=hdf5fileName)
    write(df_data[lagSignigicantIndexSpikeMin], saveDir + dataName +'_selectedTimeSeries%s_SpikeMin'%(info), hdf5fileName=hdf5fileName)

    print('Recording Lag%s-Lag%s data...'%(1,df_classifier.shape[1]))
    for lag in range(1,df_classifier.shape[1]):
        lagSignigicantIndex = (np.sum(significant_index.T[1:lag],axis=0) == 0) * (significant_index.T[lag])
        write(df_classifier[lagSignigicantIndex], saveDir + dataName +'_selected%s_LAG%s'%(info,lag), hdf5fileName=hdf5fileName)
        write(df_data[lagSignigicantIndex], saveDir + dataName +'_selectedTimeSeries%s_LAG%s'%(info,lag), hdf5fileName=hdf5fileName)
                
    return None

def clusterTimeSeriesClassification(dataName, saveDir, numberOfLagsToDraw=3, hdf5fileName=None, exportClusteringObjects=False, writeClusteringObjectToBinaries=True, AutocorrNotPeriodogr=True, vectorImage=True):

    """Visualize time series classification.
    
    Parameters:
        dataName: str
            Data name, e.g. "myData_1"

        saveDir: str
            Path of directories poining to data storage

        numberOfLagsToDraw: int, Default 3
            First top-N lags (or frequencies) to draw

        hdf5fileName: str, Default None
            HDF5 storage path and name

        exportClusteringObjects: boolean, Default False
            Whether to export clustering objects to xlsx files

        writeClusteringObjectToBinaries: boolean, Default True
            Whether to export clustering objects to binary (pickle) files

        AutocorrNotPeriodogr: boolean, Default True
            Whether to label to print on the plots

        vectorImage: boolean, Default True
            Whether to make vector image instead of raster

    Returns:
        None

    Usage:
        clusterTimeSeriesClassification('myData_1', '/dir1/dir2/', AutocorrNotPeriodogr=True, writeClusteringObjectToBinaries=True)
    """

    info = 'Autocorrelations' if AutocorrNotPeriodogr else 'Periodograms'

    if hdf5fileName is None:
        hdf5fileName = saveDir + dataName + '.h5'

    def internal(className):
        print('\n\n%s of Time Series:'%(className)) 
        df_data_selected = read(saveDir + dataName + '_selectedTimeSeries%s_%s'%(info,className), hdf5fileName=hdf5fileName)
        df_classifier_selected = read(saveDir + dataName + '_selected%s_%s'%(info,className), hdf5fileName=hdf5fileName)

        if (df_data_selected is None) or (df_classifier_selected is None):

            print('Selected %s time series not found in %s.'%(className, saveDir + dataName + '.h5'))
            print('Do time series classification first.')

            return 

        print('Creating clustering object.')
        clusteringObject = makeClusteringObject(df_data_selected, df_classifier_selected, significance='Elbow') #Silhouette

        print('Exporting clustering object.')
        if writeClusteringObjectToBinaries:
            write(clusteringObject, saveDir + 'consolidatedGroupsSubgroups/' + dataName + '_%s_%s'%(className,info) + '_GroupsSubgroups')
        
        if exportClusteringObjects:
            exportClusteringObject(clusteringObject, saveDir + 'consolidatedGroupsSubgroups/', dataName + '_%s_%s'%(className,info))

        return

    for lag in range(1,numberOfLagsToDraw + 1):
        internal('LAG%s'%(lag))
            
    internal('SpikeMax')
    internal('SpikeMin')

    return None

def visualizeTimeSeriesClassification(dataName, saveDir, numberOfLagsToDraw=3, AutocorrNotPeriodogr=True, vectorImage=True):

    """Visualize time series classification.
    
    Parameters:
        dataName: str
            Data name, e.g. "myData_1"

        saveDir: str
            Path of directories poining to data storage

        numberOfLagsToDraw: boolean, Default 3
            First top-N lags (or frequencies) to draw

        AutocorrNotPeriodogr: boolean, Default True
            Whether to label to print on the plots

        vectorImage: boolean, Default True
            Whether to raster or vector image

    Returns:
        None

    Usage:
        visualizeTimeSeriesClassification('myData_1', '/dir1/dir2/', AutocorrNotPeriodogr=True)
    """

    info = 'Autocorrelations' if AutocorrNotPeriodogr else 'Periodograms'

    def internal(className):
        print('\n\n%s of Time Series:'%(className)) 

        clusteringObject = read(saveDir + 'consolidatedGroupsSubgroups/' + dataName + '_%s_%s'%(className,info) + '_GroupsSubgroups')

        if clusteringObject is None:
            print('Cluster time series classification first.')
            return 
        
        print('Plotting Dendrogram with Heatmaps.')
        makeDendrogramHeatmap(clusteringObject, saveDir, dataName + '_%s_%sBased'%(className,info), AutocorrNotPeriodogr=AutocorrNotPeriodogr, vectorImage=vectorImage)

        return

    for lag in range(1,numberOfLagsToDraw + 1):
        internal('LAG%s'%(lag))
            
    internal('SpikeMax')
    internal('SpikeMin')

    return None
