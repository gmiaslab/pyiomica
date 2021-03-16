'''Core function of PyIOmica'''


from .globalVariables import *


def modifiedZScore(subset, printValues=False):

    """Calculate modified z-score based on "Median absolute deviation".

    Parameters:
        subset: pandas.Series
            Data to transform

    Returns:
        pandas.Series
            Transformed subset

    Usage:
        data = modifiedZScore(data)
    """

    def medianAbsoluteDeviation(expr, axis=None):

        """1D, 2D Median absolute deviation of a sequence of numbers or pd.Series.

        Parameters:
            expr: pandas.Series, pandas.DataFrame or numpy.array
                Data for analysis

            axis: int, Default None
                Multidimensional arrays are flattened, 0: use if data in columns, 1: use if data in rows

        Returns:
            float
                Median absolute deviation (M.A.D.)

        Usage:
            MedianAD = medianAbsoluteDeviation(data, axis=None)
        """

        data = None

        if isinstance(expr, np.ndarray):

            data = expr

        elif isinstance(expr, (pd.Series, pd.DataFrame)):

            data = expr.values

        try:

            if len(data) > 1:

                if axis == None or axis == 0:

                    return np.median(np.abs(data - np.median(data,axis)),axis)

                elif axis == 1:

                    if len(data.shape) < 2:

                        print('Warning: axis = %s option is invalid for 1-D array...' % (axis))

                    else:

                        return np.median((np.abs(data.transpose() - np.median(data,axis)).transpose()),axis)

        except Exception as exception:
            print(exception)

            print('Unsupported data type: ', type(expr))
        
        return None

    values = subset[~np.isnan(subset.values)].values

    MedianAD = medianAbsoluteDeviation(values, axis=None)

    if MedianAD == 0.:
        MeanAD = np.sum(np.abs(values - np.mean(values))) / len(values)
        if printValues:
            print('MeanAD:', MeanAD, '\tMedian:', np.median(values))
        coefficient = 0.7978846 / MeanAD
    else:
        if printValues:
            print('MedianAD:', MedianAD, '\tMedian:', np.median(values))
        coefficient = 0.6744897 / MedianAD
        
    subset.iloc[~np.isnan(subset.values)] = coefficient * (values - np.median(values))

    return subset

def boxCoxTransform(subset, lmbda=None, giveLmbda=False, printLambda=True):

    """Power transform from scipy.stats

    Parameters:
        subset: pandas.Series
            Data to transform

        lmbda: float, Default None
            Lambda parameter, if not specified optimal value will be determined

        giveLmbda: boolean, Default False
            Also return Lambda value

        printLmbda: boolean, Default False
            Print Lambda value

    Returns:
        pandas.Series
            Transformed data and Lambda parameter

    Usage:
        myData = boxCoxTransform(myData)
    """

    where_negative = np.where(subset < 0)
    if len(where_negative[0]>0):
        errMsg = 'Warning: negative values are present in the data. Review the sequence of the data processing steps.'
        print(errMsg)

    where_positive = np.where(subset > 0)

    if lmbda == None:
        transformed_data = scipy.stats.boxcox(subset.values[where_positive])
    else:
        transformed_data = (scipy.stats.boxcox(subset.values[where_positive], lmbda=lmbda),lmbda)

    subset.iloc[where_positive] = transformed_data[0]

    lmbda = transformed_data[1]

    if giveLmbda:

        return subset, lmbda

    if printLambda:
        print('Fitted lambda:', lmbda)

    return subset

def ampSquaredNormed(func, freq, times, data):

    """Lomb-Scargle core function. For internal use only.
    Calculate the different frequency components of our spectrum: project the cosine/sine component and normalize it

    Parameters:
        func: function
            One of two functions, i.e. Sin or Cos

        freq: float
            Frequency

        times: 1d numpy.array
            Input times (starting point adjusted w.r.t.dataset times), Zero-padded

        data: 1d numpy.array
            Input Data with the mean subtracted from it, before zero-padding.

    Returns:
        float
            Squared amplitude normalized.

    Usage:
        coef = ampSquaredNormed(np.cos, freguency, inputTimesNormed, inputDataCentered)

        Intended for internal use only.
    """

    omega_freq = 2. * (np.pi) * freq
    theta_freq = 0.5 * np.arctan2(np.sum(np.sin(4. * (np.pi) * freq * times)), np.sum(np.cos(4. * (np.pi) * freq * times) + 10 ** -20))
    
    ampSum = np.sum(data * func(omega_freq * times - theta_freq)) ** 2
    ampNorm = np.sum(func(omega_freq * times - theta_freq) ** 2)

    return chop(ampSum) / ampNorm

def autocorrelation(inputTimes, inputData, inputSetTimes, UpperFrequencyFactor=1):
    
    """Autocorrelation function

    Parameters:
        inputTimes: 1d numpy.array
            Times corresponding to provided data points

        inputData: 1d numpy.array
            Data points

        inputSetTimes: 1d numpy.array
            A complete set of all possible N times during which data could have been collected.

        UpperFrequencyFactor: int, Default 1
            Upper frequency factor.

    Returns:
        2d numpy.array
            Array of time lags with corresponding autocorrelations

    Usage:
        result = autocorrelation(inputTimes, inputData, inputSetTimes)
    """
    # in case the values passed contain nan entries, theses are removed - this could affect window estimation and ensuing functions.
    inputTimes = inputTimes[~np.isnan(inputData)]
    inputData = inputData[~np.isnan(inputData)]
    def InverseAutocovariance(inputTimes, inputData, inputSetTimes, UpperFrequencyFactor=1):

        #adjust inputTimes starting point w.r.t.dataset times, AND ZERO-PAD THEM
        inputTimesNormed = np.concatenate((inputTimes, inputSetTimes + inputSetTimes[-1])) - inputSetTimes[0]

        #calculate the number of timepoints in the overall set-since we cut
        #freqStep to half f0, we should compensate n by multiplication by two
        n = 2 * len(inputSetTimes)

        #calculate the time window of observation
        window = np.max(inputSetTimes) - np.min(inputSetTimes)

        #invert this window to get the fundamental frequency,with the n/n-1
        #correction to account for the difference between the time window and
        #time period of the signal (WE REMOVE THIS FOR NOW)
        f0 = 1.0 / window

        #subtract the mean from the inputData,BEFORE YOU ZERO-PAD IT!
        inputDataCentered = np.concatenate((inputData - np.mean(inputData), np.zeros(len(inputSetTimes))))

        #calculate a variance for the centered data
        varianceInputPoints = np.var(inputDataCentered, ddof=1)

        #define the frequency step as HALF the fundamental frequency in order
        #to zero-pad and get an evenly spaced mesh
        freqStep = 0.5 * f0

        #get the list of frequencies
        freq = np.linspace(0.5 * f0, n * UpperFrequencyFactor * 0.5 * f0, int(n * UpperFrequencyFactor))

        #calculate the inverse autocorrelation
        inverseAuto = 1.0 / (2.0 * varianceInputPoints) * np.array(tuple(map(lambda f: ampSquaredNormed(np.cos, f, inputTimesNormed, inputDataCentered) + ampSquaredNormed(np.sin, f, inputTimesNormed, inputDataCentered), list(freq))))
    
        #return: 1) the list of frequencies, 2) the correspoinding list of inverse autocovariances
        return np.transpose(np.vstack((freq, inverseAuto)))

    inputInverseAuto = InverseAutocovariance(inputTimes[np.isnan(inputData) == False], inputData[np.isnan(inputData) == False], inputSetTimes, UpperFrequencyFactor = UpperFrequencyFactor)

    #create the amplitude spectrum from the input data:
    #add a zero at the first element to make the DFT work, sample only half the
    #points because we have oversampled by 2 in the inverseAutocovariance
    inverseAmplitudes = np.concatenate(([0], inputInverseAuto[:np.int(inputInverseAuto.shape[0] / 2), 1]))

    #do the DCT-III transform:
    autoCorrs = scipy.fftpack.dct(inverseAmplitudes, type=3, norm='ortho')

    #divide everything by a normalization factor so that the autocorrelation at lag 0 = 1
    #make sure we are only returning autocorrelations for the lags we can rely on, i.e.  for up to N/2 time points
    values = autoCorrs[:np.int(np.floor(0.5 * len(autoCorrs)))] / autoCorrs[0]

    return np.vstack((inputSetTimes[:len(values)], values))

def pAutocorrelation(args):

    """Wrapper of Autocorrelation function for use with Multiprocessing.

    Parameters:
        args: tuple
            A tuple of arguments in the form (inputTimes, inputData, inputSetTimes):
                inputTimes: 1d numpy.array
                    Times corresponding to provided data points

                inputData: 1d numpy.array
                    Data points

                inputSetTimes: 1d numpy.array
                    A complete set of all possible N times during which data could have been collected.

    Returns:
        2d numpy.array
            Array of time lags with corresponding autocorrelations

    Usage:
        result = pAutocorrelation((inputTimes, inputData, inputSetTimes))
    """

    inputTimes, inputData, inputSetTimes = args
    
    return autocorrelation(inputTimes, inputData, inputSetTimes)

def getSpikes(inputData, func, cutoffs):

    """Get sorted index of signals with statistically significant spikes,
    i.e. those that pass the provided cutoff.

    Parameters:
        inputData: 2d numpy.array
            Data points where rows are normalized signals

        func: function
            Function np.max or np.min

        cutoffs: dictionary
            A dictionary of cutoff values

    Returns:
        list
            Index of data with statistically significant spikes

    Usage:
        index = getSpikes(inputData, np.max, cutoffs)
    """

    data = inputData.copy()
    counts_non_missing = np.sum(~np.isnan(data), axis=1)
    data[np.isnan(data)] = 0.

    spikesIndex = []

    for i in list(range(data.shape[1]+1)):
        ipos = np.where(counts_non_missing==i)[0]
        if len(data[ipos])>0:
            points = func(data[ipos], axis=1)

            if i in cutoffs.keys():
                spikesIndex.extend(ipos[np.where((points>cutoffs[i][0]) | (points<cutoffs[i][1]))[0]])
            else:
                print('Cuttoff for %s non-missing points is missing'%(i))

    return sorted(spikesIndex)

def LombScargle(inputTimes, inputData, inputSetTimes, FrequenciesOnly=False,NormalizeIntensities=False,OversamplingRate=1,UpperFrequencyFactor=1):

    """Calculate Lomb-Scargle periodogram.

    Parameters:
        inputTimes: 1d numpy.array 
            Times corresponding to provided data points (1D array of floats)

        inputData: 1d numpy.array 
            Data points

        inputSetTimes: 1d numpy.array 
            A complete set of all possible N times during which data could have been collected

        FrequenciesOnly: boolean, Default False
            Return frequencies only

        NormalizeIntensities: boolean, Default False
            Normalize intensities to unity

        OversamplingRate: int, Default 1
            Oversampling rate

        UpperFrequencyFactor: float, Default 1
            Upper frequency factor

    Returns:
        2d numpy.array
            Periodogram with a list of frequencies.

    Usage:
        pgram = LombScargle(inputTimes, inputData, inputSetTimes)
    """

    #adjust inputTimes starting point w.r.t.dataset times
    inputTimesNormed = inputTimes - inputSetTimes[0]

    #calculate the number of timepoints in the overall set
    n = len(inputSetTimes)

    #calculate the time window of observation
    window = np.max(inputSetTimes) - np.min(inputSetTimes)

    #invert this window to get the fundamental frequency, with the n/n-1
    #correction to account for the difference between the time window and time
    #period of the signal (WE ARE,FOR NOW,NOT INCLUDING THIS!)
    f0 = n / ((n - 1) * window)

    #subtract the mean from the inputData
    inputDataCentered = inputData - np.mean(inputData)

    #calculate a variance for the centered data
    varianceInputPoints = np.var(inputDataCentered, ddof=1)

    #define a frequency step
    freqStep = 1 / (OversamplingRate * (np.floor(n / 2) - 1)) * (n / 2 * UpperFrequencyFactor - 1) * f0

    #get the list of frequencies, adjusting both the lower frequency (to equal
    #f0 0-effectively a lowpass filter) and the upper cutoff Nyquist by the upper factor specified
    freq = np.linspace(f0, n / 2 * UpperFrequencyFactor * f0, int(f0 * (n / 2 * UpperFrequencyFactor) / freqStep))

    if FrequenciesOnly:
        return freq

    #get the periodogram
    periodogram = 1.0 / (2.0 * varianceInputPoints) * np.array(tuple(map(lambda f: chop(ampSquaredNormed(np.cos, f, inputTimesNormed, inputDataCentered)) + chop(ampSquaredNormed(np.sin, f, inputTimesNormed, inputDataCentered)), list(freq))))
    
    #the function finally returns:1) the list of frequencies, 2) the
    #corresponding list of Lomb-Scargle spectra
    if NormalizeIntensities:
        periodogram = periodogram / np.sqrt(np.dot(periodogram,periodogram))

    returning = np.vstack((freq, periodogram))

    return returning

def pLombScargle(args):

    """Wrapper of LombScargle function for use with Multiprocessing.

    Parameters:
        args: 
            A tuple of arguments in the form (inputTimes, inputData, inputSetTimes):
                inputTimes: 1d numpy.array
                    Times corresponding to provided data points

                inputData: 1d numpy.array
                    Data points

                inputSetTimes: 1d numpy.array
                    A complete set of all possible N times during which data could have been collected.

    Returns:
        2d numpy.array
            Array of frequencies with corresponding intensities

    Usage:
        result = pLombScargle((inputTimes, inputData, inputSetTimes))
    """

    inputTimes, inputData, inputSetTimes = args
    
    return LombScargle(inputTimes, inputData, inputSetTimes)

def getAutocorrelationsOfData(params):

    """Calculate autocorrelation using Lomb-Scargle Autocorrelation.
    NOTE: there should be already no missing or non-numeric points in the input Series or Dataframe

    Parameters:
        params: tuple
            A tuple of parameters in the form (df_data, setAllInputTimes), where
            df_data is a pandas Series or Dataframe, 
            setAllInputTimes is a complete set of all possible N times during which data could have been collected.

    Returns:
        2d numpy.array
            Array of autocorrelations of data.

    Usage:
        result  = autocorrelation(df_data, setAllInputTimes)
    """

    df, setAllInputTimes = params

    if isinstance(df, pd.Series):

        return autocorrelation(df.index.values, df.values, setAllInputTimes)

    elif isinstance(df, pd.DataFrame):
        listOfAutocorrelations = []

        for timeSeriesIndex in df.index:
            listOfAutocorrelations.append(autocorrelation(df.loc[timeSeriesIndex].index.values, df.loc[timeSeriesIndex].values, setAllInputTimes))

        return np.vstack(listOfAutocorrelations)

    print('Warning: Input data type unrecognized: use <pandas.Series> or <pandas.DataFrame>')

    return None

def metricCommonEuclidean(u,v):

    """Metric to calculate 'euclidean' distance between vectors u and v 
    using only common non-missing points (not NaNs).

    Parameters:
        u: 1d numpy.array
            Numpy 1-D array

        v: 1d numpy.array
            Numpy 1-D array

    Returns:
        float
            Measure of the distance between u and v

    Usage:
        dist = metricCommonEuclidean(u,v)
    """

    where_common = (~np.isnan(u)) * (~np.isnan(v))

    return np.sqrt(((u[where_common] - v[where_common]) ** 2).sum())

def chop(expr, tolerance=1e-10):

    """Equivalent of Mathematica.Chop Function.

    Parameters:
        expr: float, tuple, list or numpy.array
            A number or a pyhton sequence of numbers

        tolerance: 
            Default is the same as in Mathematica

    Returns:
        Chopped data

    Usage
        data = chop(data)
    """
        
    if isinstance(expr, (list, tuple, np.ndarray)):

        expr_copy = np.copy(expr)
        expr_copy[np.abs(expr) < tolerance] = 0

    else:
        expr_copy = 0 if expr < tolerance else expr

    return expr_copy
