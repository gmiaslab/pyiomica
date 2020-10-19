#import sys
#sys.path.append("../..")

import pyiomica as pio

from pyiomica.visualizationFunctions import makeVisibilityGraph, makeVisibilityBarGraph
from pyiomica import dataStorage as ds
from pyiomica.extendedDataFrame import DataFrame

# Make an example of time series
# Set intensities of all signals and the measurement time points (here positions)
positions = pio.np.array(range(24))
intensities = pio.np.cos(pio.np.array(range(24))*0.25 + 1.)**2 + 0.5*(pio.np.random.random(24) - 0.5)
intensities[intensities < 0.0] = 0.

# Make normal visibility graphs on a 'cirle' and 'line' layouts
makeVisibilityGraph(intensities, positions, 'results', 'circle_VG', layout='circle')
makeVisibilityGraph(intensities, positions, 'results', 'line_VG', layout='line')

# Make horizontal visibility graphs on a 'cirle' and 'line' layouts
makeVisibilityGraph(intensities, positions, 'results', 'circle_VG', layout='circle', horizontal=True)
makeVisibilityGraph(intensities, positions, 'results', 'line_VG', layout='line', horizontal=True)

# Make horizontal and normal bar-style visibility graphs
makeVisibilityBarGraph(intensities, positions, 'results', 'barVG')
makeVisibilityBarGraph(intensities, positions, 'results', 'barVG', horizontal=True)