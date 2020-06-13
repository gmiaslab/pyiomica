'''PyIOmica Dataframe extending Pandas DataFrame with new functions'''

import sklearn.preprocessing

from .globalVariables import *

from . import utilityFunctions
from . import coreFunctions


class DataFrame(pd.DataFrame):

    '''Class based on pandas.DataFrame extending capabilities into the doamin of PyIOmica
    
    Initialization parameters are identical to those in pandas.DataFrame
    See https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html for detail.    
    '''

    def __init__(self, data=None, index=None, columns=None, dtype=None, copy=False):

        '''Initialization method'''

        super().__init__(data=data, index=index, columns=columns, dtype=dtype, copy=copy)

        return

    def filterOutAllZeroSignals(self, inplace=False):

        """Filter out all-zero signals from a DataFrame.
    
        Parameters:
            inplace: boolean, Default False
                Whether to modify data in place or return a new one

        Returns:
            Dataframe or None
                Processed data

        Usage:
            df_data = df_data.filterOutAllZeroSignals()

            or

            df_data.filterOutAllZeroSignalse(inplace=True)
       """

        print('Filtering out all-zero signals')

        init = self.shape[0]

        new_data = self.loc[self.index[np.count_nonzero(self, axis=1) > 0]]

        print('Removed ', init - new_data.shape[0], 'signals out of %s.' % init) 
        print('Remaining ', new_data.shape[0], 'signals!')

        if inplace:
            self._update_inplace(new_data)
        else:
            return self._constructor(new_data).__finalize__(self)

        return

    def filterOutFractionZeroSignals(self, min_fraction_of_non_zeros, inplace=False):
       
        """Filter out fraction-zero signals from a DataFrame.
    
        Parameters:
            min_fraction_of_non_zeros: float
                Maximum fraction of allowed zeros

            inplace: boolean, Default False
                Whether to modify data in place or return a new one

        Returns:
            Dataframe or None
                Processed data

        Usage:
            df_data = df_data.filterOutFractionZeroSignals(0.75)

            or

            df_data.filterOutFractionZeroSignals(0.75, inplace=True)
       """

        print('Filtering out low-quality signals (with more than %s%% zero points)' %(np.round(100.*(1.-min_fraction_of_non_zeros), 3)))

        min_number_of_non_zero_points = np.int(np.ceil(min_fraction_of_non_zeros * self.shape[1]))
        new_data = self.loc[self.index[np.count_nonzero(self, axis=1) >= min_number_of_non_zero_points]]

        if (self.shape[0] - new_data.shape[0]) > 0:
            print('Removed ', self.shape[0] - new_data.shape[0], 'signals out of %s.'%(self.shape[0])) 
            print('Remaining ', new_data.shape[0], 'signals!')

        if inplace:
            self._update_inplace(new_data)
        else:
            return self._constructor(new_data).__finalize__(self)

        return
    
    def filterOutFractionMissingSignals(self, min_fraction_of_non_missing, inplace=False):
       
        """Filter out fraction-zero signals from a DataFrame.
    
        Parameters:
            min_fraction_of_non_missing: float
                Maximum fraction of allowed zeros

            inplace: boolean, Default False
                Whether to modify data in place or return a new one

        Returns:
            Dataframe or None
                Processed data

        Usage:
            df_data = df_data.filterOutFractionMissingSignals(0.75)

            or

            df_data.filterOutFractionMissingSignals(0.75, inplace=True)
       """

        print('Filtering out low-quality signals (with more than %s%% missing points)' %(np.round(100.*(1.-min_fraction_of_non_missing), 3)))

        min_number_of_non_zero_points = np.int(np.ceil(min_fraction_of_non_missing * self.shape[1]))
        new_data = self.loc[self.index[(~np.isnan(self)).sum(axis=1) >= min_number_of_non_zero_points]]

        if (self.shape[0] - new_data.shape[0]) > 0:
            print('Removed ', self.shape[0] - new_data.shape[0], 'signals out of %s.'%(self.shape[0])) 
            print('Remaining ', new_data.shape[0], 'signals!')

        if inplace:
            self._update_inplace(new_data)
        else:
            return self._constructor(new_data).__finalize__(self)

        return

    def filterOutReferencePointZeroSignals(self, referencePoint=0, inplace=False):

        """Filter out out first time point zeros signals from a DataFrame.
    
        Parameters:
            referencePoint: int, Default 0
                Index of the reference point
            
            inplace: boolean, Default False
                Whether to modify data in place or return a new one

        Returns:
            Dataframe or None
                Processed data

        Usage:
            df_data = df_data.filterOutFirstPointZeroSignals()

            or

            df_data.filterOutFirstPointZeroSignals(inplace=True)
       """

        print('Filtering out first time point zeros signals')

        new_data = self.loc[~(self.iloc[:,0] == 0.0)].copy()

        if (self.shape[0] - new_data.shape[0]) > 0:
            print('Removed ', self.shape[0] - new_data.shape[0], 'signals out of %s.'%(self.shape[0])) 
            print('Remaining ', new_data.shape[0], 'signals!')
        
        if inplace:
            self._update_inplace(new_data)
        else:
            return self._constructor(new_data).__finalize__(self)

        return self

    def tagValueAsMissing(self, value=0.0, inplace=False):

        """Tag zero values with NaN.
    
        Parameters:
            inplace: boolean, Default False
                Whether to modify data in place or return a new one

        Returns:
            Dataframe or None
                Processed data

        Usage:
            df_data = df_data.tagValueAsMissing()

            or

            df_data.tagValueAsMissing(inplace=True)
        """

        print('Tagging %s values with %s'%(value, np.NaN))

        new_data = self.replace(to_replace=value, value=np.NaN, inplace=False)

        if inplace:
            self._update_inplace(new_data)
        else:
            return self._constructor(new_data).__finalize__(self)

        return

    def tagMissingAsValue(self, value=0.0, inplace=False):

        """Tag NaN with zero.
    
        Parameters:
            inplace: boolean, Default False
                Whether to modify data in place or return a new one

        Returns:
            Dataframe or None
                Processed data

        Usage:
            df_data = df_data.tagMissingAsValue()

            or

            df_data.tagMissingAsValue(inplace=True)
        """

        print('Tagging %s values with %s'%(np.NaN, value))

        new_data = self.replace(to_replace=np.NaN, value=value, inplace=False)

        if inplace:
            self._update_inplace(new_data)
        else:
            return self._constructor(new_data).__finalize__(self)

        return

    def tagLowValues(self, cutoff, replacement, inplace=False):

        """Tag low values with replacement value.
    
        Parameters:
            cutoff: float
                Values below the "cutoff" are replaced with "replacement" value

            replacement: float
                Values below the "cutoff" are replaced with "replacement" value
                
            inplace: boolean, Default False
                Whether to modify data in place or return a new one

        Returns:
            Dataframe or None
                Processed data

        Usage:
            df_data = df_data.tagLowValues(1., 1.)

            or

            df_data.tagLowValues(1., 1., inplace=True)
        """

        print('Tagging low values (<=%s) with %s'%(cutoff, replacement))

        new_data = self.mask(self <= cutoff, other=replacement, inplace=False)

        if inplace:
            self._update_inplace(new_data)
        else:
            return self._constructor(new_data).__finalize__(self)

        return

    def removeConstantSignals(self, theta_cutoff, inplace=False):

        """Remove constant signals.
      
        Parameters:
            theta_cutoff: float
                Parameter for filtering the signals
                
            inplace: boolean, Default False
                Whether to modify data in place or return a new one

        Returns:
            Dataframe or None
                Processed data
            
        Usage:
            df_data = df_data.removeConstantSignals(0.3)

            or

            df_data.removeConstantSignals(0.3, inplace=True)
        """

        print('\nRemoving constant signals. Cutoff value is %s'%(theta_cutoff))

        new_data = self.iloc[np.where(np.std(self,axis=1) / np.mean(np.std(self,axis=1)) > theta_cutoff)[0]]

        print('Removed ', self.shape[0] - new_data.shape[0], 'signals out of %s.' % self.shape[0])
        print('Remaining ', new_data.shape[0], 'signals!')

        if inplace:
            self._update_inplace(new_data)
        else:
            return self._constructor(new_data).__finalize__(self)

        return

    def boxCoxTransform(self, axis=1, inplace=False):

        """Box-cox transform data.
    
        Parameters:
            axis: int, Default 1
                Direction of processing, columns (1) or rows (0)
            
            inplace: boolean, Default False
                Whether to modify data in place or return a new one

        Returns:
            Dataframe or None
                Processed data

        Usage:
            df_data = df_data.boxCoxTransformDataframe()

            or

            df_data.boxCoxTransformDataframe(inplace=True)
        """
    
        print('Box-cox transforming raw data')
            
        new_data = self.apply(coreFunctions.boxCoxTransform, axis=axis)

        if inplace:
            self._update_inplace(new_data)
        else:
            return self._constructor(new_data).__finalize__(self)

        return

    def modifiedZScore(self, axis=0, inplace=False):

        """Z-score (Median-based) transform data.
    
        Parameters:
            axis: int, Default 1
                Direction of processing, rows (1) or columns (0)

            inplace: boolean, Default False
                Whether to modify data in place or return a new one

        Returns:
            Dataframe or None
                Processed data

        Usage:
            df_data = df_data.modifiedZScoreDataframe()

            or

            df_data.modifiedZScoreDataframe(inplace=True)
        """
            
        print('Z-score (Median-based) transforming box-cox transformed data')

        new_data = self.copy()
        new_data = new_data.apply(coreFunctions.modifiedZScore, axis=axis)

        if inplace:
            self._update_inplace(new_data)
        else:
            return self._constructor(new_data).__finalize__(self)

        return

    def normalizeSignalsToUnity(self, referencePoint=0, inplace=False):

        """Normalize signals to unity.
    
        Parameters:
            referencePoint: int, Default 0
                Index of the reference point

            inplace: boolean, Default False
                Whether to modify data in place or return a new one

        Returns:
            Dataframe or None
                Processed data

        Usage:
            df_data = df_data.normalizeSignalsToUnityDataframe()

            or

            df_data.normalizeSignalsToUnityDataframe(inplace=True)
        """

        print('Normalizing signals to unity')

        if not referencePoint is None:
            #Subtract reference time-point value from all time-points
            new_data = self.compareTimeSeriesToPoint(point=referencePoint, inplace=False).copy()
        else:
            new_data = self.copy()

        where_nan = np.isnan(new_data.values.astype(float))
        new_data[where_nan] = 0.0
        new_data = new_data.apply(lambda data: data / np.sqrt(np.dot(data,data)),axis=1)
        new_data[where_nan] = np.nan

        if inplace:
            self._update_inplace(new_data)
        else:
            return self._constructor(new_data).__finalize__(self)

        return

    def quantileNormalize(self, output_distribution='original', averaging=np.mean, ties=np.mean, inplace=False):

        """Quantile Normalize signals in a DataFrame. 
    
        Note that it is possible there may be equal values within the dataset. In such a scenario, by default, the quantile 
        normalization implementation considered here works by replacing the degenerate values with the mean over all the degenerate ranks.
        Note, that for the default option to work the data should not have any missing values.
        If output_distribution is set to 'uniform' or 'normal' then the scikit-learn's Quantile Transformation is used.

        Parameters:
            output_distribution: str, Default 'original'
                Output distribution. Other options are 'normal' and 'uniform'

            averaging: function, Default np.mean
                With what value to replace the same-rank elements across samples. 
                Default is to take the mean of same-rank elements

            ties: function or str, Default np.mean
                Function or name of the function. How ties should be handled. Default is to replace ties with their mean.
                Other possible options are: 'mean', 'median', 'prod', 'sum', etc.

            inplace: boolean, Default False
                Whether to modify data in place or return a new one

        Returns:
            Dataframe or None
                Processed data

        Usage:
            df_data = pd.DataFrame(index=['Gene 1','Gene 2','Gene 3','Gene 4'], columns=['Col 0','Col 1','Col 2'], data=np.array([[5, 4, 3], [2, 1, 4], [3, 4, 6], [4, 2, 8]]))

            df_data = df_data.quantileNormalize()

            or

            df_data.df_data.quantileNormalize(inplace=True)
        """

        print('Quantile normalizing signals...')

        if output_distribution=='original':

            def rankTransform(series, weights):

                se_temp = pd.Series(index=scipy.stats.rankdata(series.values, method='min'), 
                               data=weights[scipy.stats.rankdata(series.values, method='ordinal')-1])

                series[:] = pd.Series(se_temp.index).replace(to_replace=se_temp.groupby(level=0).agg(ties).to_dict()).values

                return series

            weights = averaging(np.sort(self.values, axis=0), axis=1)

            new_data = self.copy()
            new_data = new_data.apply(lambda col: rankTransform(col, weights), axis=0)

        elif output_distribution=='normal' or output_distribution=='uniform': 

            new_data = self.copy()
            new_data.iloc[:] = sklearn.preprocessing.quantile_transform(self.values, output_distribution=output_distribution, n_quantiles=min(self.shape[0],1000), copy=False)

        if inplace:
            self._update_inplace(new_data)
        else:
            return self._constructor(new_data).__finalize__(self)

        return

    def compareTimeSeriesToPoint(self, point='first', inplace=False):

        """Subtract a particular point of each time series (row) of a Dataframe.
    
        Parameters:     
            point: str, int or float
                Possible options are 'first', 'last', 0, 1, ... , 10, or a value.

            inplace: boolean, Default False
                Whether to modify data in place or return a new one

        Returns:
            Dataframe or None
                Processed data

        Usage:
            df_data = df_data.compareTimeSeriesToPoint()

            or

            df_data.compareTimeSeriesToPoint(df_data)
        """

        independent = True

        if point == 'first':
            idx = 0
        elif point == 'last':
            idx = len(self.columns) - 1
        elif type(point) is int:
            idx = point
        elif type(point) is float:
            independent = False
        else:
            print("Specify a valid comparison point: 'first', 'last', 0, 1, ..., 10, etc., or a value")
            return

        new_data = self.copy()

        if independent:
            new_data.iloc[:] = (self.values.T - self.values.T[idx]).T
        else:
            new_data.iloc[:] = (self.values.T - point).T

        if inplace:
            self._update_inplace(new_data)
        else:
            return self._constructor(new_data).__finalize__(self)

        return

    def compareTwoTimeSeries(self, df, function=np.subtract, compareAllLevelsInIndex=True, mergeFunction=np.mean):

        """Create a new Dataframe based on comparison of two existing Dataframes.
    
        Parameters:
            df: pandas.DataFrame
                Data to compare

            function: function, Default np.subtract 
                Other options are np.add, np.divide, or another <ufunc>.

            compareAllLevelsInIndex: boolean, Default True
                Whether to compare all levels in index.
                If False only "source" and "id" will be compared

            mergeFunction: function, Default np.mean
                Input Dataframes are merged with this function, 
                i.e. np.mean (default), np.median, np.max, or another <ufunc>.

        Returns:
            DataFrame or None
                Processed data

        Usage:
            df_data = df_dataH2.compareTwoTimeSeries(df_dataH1, function=np.subtract, compareAllLevelsInIndex=False, mergeFunction=np.median)
        """

        if self.index.names!=df.index.names:
            errMsg = 'Index of Dataframe 1 is not of the same shape as index of Dataframe 2!'
            print(errMsg)
            return errMsg

        if compareAllLevelsInIndex:
            df1_grouped, df2_grouped = self, df
        else:
            def aggregate(df):
                return df.groupby(level=['source', 'id']).agg(mergeFunction)

            df1_grouped, df2_grouped = aggregate(self), aggregate(df)

        index = pd.MultiIndex.from_tuples(list(set(df1_grouped.index.values).intersection(set(df2_grouped.index.values))), 
                                            names=df1_grouped.index.names)

        return function(df1_grouped.loc[index], df2_grouped.loc[index])

    def imputeMissingWithMedian(self, axis=1, inplace=False):

        """Normalize signals to unity.
    
        Parameters:
            axis: int, Default 1
                Axis to apply trasnformation along

            inplace: boolean, Default False
                Whether to modify data in place or return a new one

        Returns:
            Dataframe or None
                Processed data

        Usage:
            df_data = df_data.imputeMissingWithMedian()

            or

            df_data.imputeMissingWithMedian(inplace=True)
        """

        def tempFunction(data):

            data[np.isnan(data)] = np.median(data[np.isnan(data) == False])

            return data

        new_data = self.apply(tempFunction, axis=axis)

        if inplace:
            self._update_inplace(new_data)
        else:
            return self._constructor(new_data).__finalize__(self)

        return data

def mergeDataframes(listOfDataframes, axis=0):

    """Merge a list of Dataframes (outer join).
    
    Parameters:
        listOfDataframes: list
            List of pandas.DataFrames

        axis: int, Default 0
            Merge direction. 0 to stack vertically, 1 to stack horizontally

    Returns:
        pandas.Dataframe
            Processed data

    Usage:
        df_data = mergeDataframes([df_data1, df_data2])
    """

    if len(listOfDataframes)==0:
        return None
    elif len(listOfDataframes)==1:
        return listOfDataframes[0]

    df = pd.concat(listOfDataframes, sort=False, axis=axis)

    return DataFrame(df)

def getLombScarglePeriodogramOfDataframe(df_data, NumberOfCPUs=4, parallel=True):

    """Calculate Lomb-Scargle periodogram of DataFrame.
    
    Parameters:
        df: pandas.DataFrame
            Data to process

        parallel: boolean, Default True
            Whether to calculate in parallel mode (>1 process)

        NumberOfCPUs: int, Default 4
            Number of processes to create if parallel is True

    Returns:
        pandas.Dataframe
            Lomb-Scargle periodograms

    Usage:
        df_periodograms = getLombScarglePeriodogramOfDataframe(df_data)
    """

    if parallel:

        results = utilityFunctions.runCPUs(NumberOfCPUs, coreFunctions.pLombScargle, [(series.index[~np.isnan(series)].values, series[~np.isnan(series)].values, df_data.columns.values) for index, series in df_data.iterrows()])

        df_periodograms = pd.DataFrame(data=results[1::2], index=df_data.index, columns=results[0])

    else:
        frequencies = None
        intensities = []

        for index, series in df_data.iterrows():
            values = series[~np.isnan(series)].values
            times = series.index[~np.isnan(series)].values

            tempFrequencies, tempIntensities = coreFunctions.LombScargle(times, values, series.index.values, OversamplingRate=1)

            if frequencies is None:
                frequencies = tempFrequencies

            intensities.append(tempIntensities)

        df_periodograms = pd.DataFrame(data=np.vstack(intensities), index=df_data.index, columns=frequencies)

    return DataFrame(df_periodograms)

def getRandomSpikesCutoffs(df_data, p_cutoff, NumberOfRandomSamples=10**3):

    """Calculate spikes cuttoffs from a bootstrap of provided data,
    gived the significance cutoff p_cutoff.

    Parameters:
        df_data: pandas.DataFrame 
            Data where rows are normalized signals

        p_cutoff: float
            p-Value cutoff, e.g. 0.01

        NumberOfRandomSamples: int, Default 1000
            Size of the bootstrap distribution

    Returns:
        dictionary
            Dictionary of spike cutoffs.

    Usage:
        cutoffs = getSpikesCutoffs(df_data, 0.01)
    """

    data = np.vstack([np.random.choice(df_data.values[:,i], size=NumberOfRandomSamples, replace=True) for i in range(len(df_data.columns.values))]).T

    df_data_random = DataFrame(pd.DataFrame(data=data, index=range(NumberOfRandomSamples), columns=df_data.columns))
    df_data_random.filterOutFractionZeroSignals(0.75, inplace=True)
    df_data_random.normalizeSignalsToUnity(inplace=True)
    df_data_random.removeConstantSignals(0., inplace=True)

    data = df_data_random.values
    counts_non_missing = np.sum(~np.isnan(data), axis=1)
    data[np.isnan(data)] = 0.

    cutoffs = {}

    for i in list(range(data.shape[1]+1)):
        idata = data[counts_non_missing==i]
        if len(idata)>0:
            cutoffs.update({i : (np.quantile(np.max(idata, axis=1), 1.-p_cutoff, interpolation='lower'),
            np.quantile(np.min(idata, axis=1), p_cutoff, interpolation='lower'))} )

    return cutoffs

def getRandomAutocorrelations(df_data, NumberOfRandomSamples=10**5, NumberOfCPUs=4, fraction=0.75, referencePoint=0):

    """Generate autocorrelation null-distribution from permutated data using Lomb-Scargle Autocorrelation.
    NOTE: there should be already no missing or non-numeric points in the input Series or Dataframe

    Parameters:
        df_data: pandas.Series or pandas.Dataframe

        NumberOfRandomSamples: int, Default 10**5
            Size of the distribution to generate

        NumberOfCPUs: int, Default 4
            Number of processes to run simultaneously

    Returns:
        pandas.DataFrame
            Dataframe containing autocorrelations of null-distribution of data.

    Usage:
        result = getRandomAutocorrelations(df_data)
    """

    data = np.vstack([np.random.choice(df_data.values[:,i], size=NumberOfRandomSamples, replace=True) for i in range(len(df_data.columns.values))]).T

    df_data_random = DataFrame(pd.DataFrame(data=data, index=range(NumberOfRandomSamples), columns=df_data.columns))
    df_data_random.filterOutFractionZeroSignals(fraction, inplace=True)
    df_data_random.normalizeSignalsToUnity(inplace=True, referencePoint=referencePoint)
    df_data_random.removeConstantSignals(0., inplace=True)

    print('\nCalculating autocorrelations of %s random samples (sampled with replacement)...'%(df_data_random.shape[0]))

    results = utilityFunctions.runCPUs(NumberOfCPUs, coreFunctions.pAutocorrelation, [(df_data_random.iloc[i].index.values.copy(), 
                                                                                       df_data_random.iloc[i].values.copy(), 
                                                                                       df_data.columns.values.copy()) for i in range(df_data_random.shape[0])])
    
    return pd.DataFrame(data=results[1::2], columns=results[0])

def getRandomPeriodograms(df_data, NumberOfRandomSamples=10**5, NumberOfCPUs=4, fraction=0.75, referencePoint=0):

    """Generate periodograms null-distribution from permutated data using Lomb-Scargle function.

    Parameters:
        df_data: pandas.Series or pandas.Dataframe

        NumberOfRandomSamples: int, Default 10**5
            Size of the distribution to generate

        NumberOfCPUs: int, Default 4
            Number of processes to run simultaneously

    Returns:
        pandas.DataFrame
            Dataframe containing periodograms

    Usage:
        result = getRandomPeriodograms(df_data)
    """

    data = np.vstack([np.random.choice(df_data.values[:,i], size=NumberOfRandomSamples, replace=True) for i in range(len(df_data.columns.values))]).T

    df_data_random = DataFrame(pd.DataFrame(data=data, index=range(NumberOfRandomSamples), columns=df_data.columns))
    df_data_random.filterOutFractionZeroSignals(fraction, inplace=True)
    df_data_random.normalizeSignalsToUnity(inplace=True, referencePoint=referencePoint)
    df_data_random.removeConstantSignals(0., inplace=True)

    print('\nCalculating periodograms of %s random samples (sampled with replacement)...'%(df_data_random.shape[0]))

    return getLombScarglePeriodogramOfDataframe(df_data_random, NumberOfCPUs=NumberOfCPUs)
