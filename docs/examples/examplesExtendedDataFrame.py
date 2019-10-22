import importlib, sys
sys.path.append("../..")

# Import PyIOmica
import pyiomica as pio

# Import Extended DataFrame from PyIOmica
from pyiomica.extendedDataFrame import DataFrame

# Create a simple data for testing and demonstration
df_data = DataFrame(data=pio.np.array([[0.5,2,3],
                                      [0,2,6],
                                      [7,3,0],
                                      [2,2,8],
                                      [1,pio.np.nan,pio.np.nan],
                                      [6,0,0],
                                      [0,0,0],
                                      [3,3,3.1],
                                      [3,2,pio.np.nan],
                                      [4,pio.np.nan,4]]).astype(float),
                    index=['s1','s2','s3','s4','s5','s6','s7','s8','s9','s10'],
                    columns=['c1', 'c2', 'c3'])
print(df_data, '\n')

# Remove all-zero signals from the data
df_data.filterOutAllZeroSignals(inplace=True)
print(df_data, '\n')

# Remove firt-point-zero signals from the data
df_data.filterOutFirstPointZeroSignals(inplace=True)
print(df_data, '\n')

# Remove nearly-constant signals from the data
df_data.removeConstantSignals(0.2, inplace=True)
print(df_data, '\n')

# Remove signals with >75% non-zero points
df_data.filterOutFractionZeroSignals(0.6, inplace=True)
print(df_data, '\n')

# Remove signals with >75% non-zero points
df_data.filterOutFractionMissingSignals(0.8, inplace=True)
print(df_data, '\n')

# Add a signal with zeros
df_data.loc['s11'] = [2,0,6]
print(df_data, '\n')

# Replace any zeros with np.NaN (missing)
df_data.tagValueAsMissing(inplace=True)
print(df_data, '\n')

# Replace any missing values (np.NaN) with values
df_data.tagMissingAsValue(value=0, inplace=True)
print(df_data, '\n')

# Replace any values smaller than 'a' with 'b'
df_data.tagLowValues(1., 1., inplace=True)
print(df_data, '\n') 

# Calculate modified zscore (median-based) of data
df_data_zm = df_data.modifiedZScore()
print(df_data_zm, '\n') 

# Quantile normalize the data
df_data_qn = df_data.quantileNormalize()
print(df_data_qn, '\n') 

# Box-cox transform data
df_data_bc = df_data.boxCoxTransform()
print(df_data_bc, '\n')

# Normalize signals to unity
df_data_un = df_data.normalizeSignalsToUnity()
print(df_data_un, '\n')