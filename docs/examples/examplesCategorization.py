
import pyiomica as pio

from pyiomica import categorizationFunctions as cf

# Unzip example data
with pio.zipfile.ZipFile(pio.os.path.join(pio.ConstantPyIOmicaExamplesDirectory, 'SLV.zip'), "r") as zipFile:
    zipFile.extractall(path=pio.ConstantPyIOmicaExamplesDirectory)

# Process sample dataset SLV_Hourly1 
# Name of the fisrt data set 
dataName = 'SLV_Hourly1TimeSeries'

# Define a directory name where results are be saved
saveDir = pio.os.path.join('results', dataName, '')

# Directory name where example data is (*.csv files)
dataDir = pio.os.path.join(pio.ConstantPyIOmicaExamplesDirectory, 'SLV')

# Read the example data into a DataFrame
df_data = pio.pd.read_csv(pio.os.path.join(dataDir, dataName + '.csv'), index_col=[0,1,2], header=0)

# Calculate time series categorization
cf.calculateTimeSeriesCategorization(df_data, dataName, saveDir, NumberOfRandomSamples = 10**4)

# Cluster the time series categorization results
cf.clusterTimeSeriesCategorization(dataName, saveDir)

# Make plots of the clustered time series categorization
cf.visualizeTimeSeriesCategorization(dataName, saveDir)


# Process sample dataset SLV_Hourly2, in the same way as SLV_Hourly1 above
dataName = 'SLV_Hourly2TimeSeries'
saveDir = pio.os.path.join('results', dataName, '')
dataDir = pio.os.path.join(pio.ConstantPyIOmicaExamplesDirectory, 'SLV')
df_data = pd.read_csv(pio.os.path.join(dataDir, dataName + '.csv'), index_col=[0,1,2], header=0)
cf.calculateTimeSeriesCategorization(df_data, dataName, saveDir, NumberOfRandomSamples = 10**3)
cf.clusterTimeSeriesCategorization(dataName, saveDir)
cf.visualizeTimeSeriesCategorization(dataName, saveDir)


# Import data storage submodule to read results of processing sample datasets SLV_Hourly1 and SLV_Hourly2
from pyiomica import dataStorage as ds

# Use results from processing sample datasets SLV_Hourly1 and SLV_Hourly2 to calculate "Delta"
df_data_processed_H1 = ds.read(dataName+'_df_data_transformed', hdf5fileName=pio.os.path.join('results',dataName,dataName+'.h5'))
df_data_processed_H2 = ds.read(dataName+'_df_data_transformed', hdf5fileName=pio.os.path.join('results',dataName,dataName+'.h5'))
dataName = 'SLV_Delta'
saveDir = pio.os.path.join('results', dataName, '')
df_data = df_data_processed_H2.compareTwoTimeSeries(df_data_processed_H1, compareAllLevelsInIndex=False, mergeFunction=np.median).fillna(0.)
cf.calculateTimeSeriesCategorization(df_data, dataName, saveDir, NumberOfRandomSamples = 5*10**3)
cf.clusterTimeSeriesCategorization(dataName, saveDir)
cf.visualizeTimeSeriesCategorization(dataName, saveDir)