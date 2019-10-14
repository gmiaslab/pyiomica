
import pyiomica as pio

from pyiomica.visualizationFunctions import makeVisibilityGraph, makeVisibilityBarGraph
from pyiomica import dataStorage as ds
from pyiomica.extendedDataFrame import DataFrame

ExampleClusteringObject = ds.read(pio.os.path.join(pio.ConstantPyIOmicaExamplesDirectory, 'exampleClusteringObject_SLV_Delta_LAG1_Autocorr'))

intensities, positions = ExampleClusteringObject[1][2]['data'].values, ExampleClusteringObject[1][2]['data'].columns.values

normalized_intensities = DataFrame(data=intensities).imputeMissingWithMedian().apply(lambda data: pio.np.sum(data[data > 0.0]) / len(data), axis=0).values

makeVisibilityGraph(normalized_intensities, positions, 'results', 'circle_VG', layout='circle')
makeVisibilityGraph(normalized_intensities, positions, 'results', 'line_VG', layout='line')

makeVisibilityGraph(normalized_intensities, positions, 'results', 'circle_VG', layout='circle', horizontal=True)
makeVisibilityGraph(normalized_intensities, positions, 'results', 'line_VG', layout='line', horizontal=True)

makeVisibilityBarGraph(normalized_intensities, positions, 'results', 'barVG')
makeVisibilityBarGraph(normalized_intensities, positions, 'results', 'barVG', horizontal=True)