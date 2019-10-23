#import sys
#sys.path.append("../..")

import pyiomica as pio

from pyiomica.visualizationFunctions import makeVisibilityGraph, makeVisibilityBarGraph
from pyiomica import dataStorage as ds
from pyiomica.extendedDataFrame import DataFrame

# Get an example of a clustering object from ConstantPyIOmicaExamplesDirectory
ExampleClusteringObject = ds.read(pio.os.path.join(pio.ConstantPyIOmicaExamplesDirectory, 'exampleClusteringObject_SLV_Delta_LAG1_Autocorr'))

# Use Group 1, Subgroup 2 data (chosen arbitrarily here)
df_data = ExampleClusteringObject[1][2]['data']

# Read intensities of all signals and the measurement time points (here positions)
intensities, positions = df_data.values, df_data.columns.values

# Normalize and aggregate the intensities to obtain one averaged signal
normalized_intensities = DataFrame(data=intensities).imputeMissingWithMedian().apply(lambda data: pio.np.sum(data[data > 0.0]) / len(data), axis=0).values

# Make normal visibility graphs on a 'cirle' and 'line' layouts
makeVisibilityGraph(normalized_intensities, positions, 'results', 'circle_VG', layout='circle')
makeVisibilityGraph(normalized_intensities, positions, 'results', 'line_VG', layout='line')

# Make horizontal visibility graphs on a 'cirle' and 'line' layouts
makeVisibilityGraph(normalized_intensities, positions, 'results', 'circle_VG', layout='circle', horizontal=True)
makeVisibilityGraph(normalized_intensities, positions, 'results', 'line_VG', layout='line', horizontal=True)

# Make horizontal and normal bar-style visibility graphs
makeVisibilityBarGraph(normalized_intensities, positions, 'results', 'barVG')
makeVisibilityBarGraph(normalized_intensities, positions, 'results', 'barVG', horizontal=True)