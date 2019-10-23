#import sys
#sys.path.append("../..")

import pyiomica as pio

from pyiomica.utilityFunctions import readMathIOmicaData

print(pio.ConstantPyIOmicaExamplesDirectory, '\n')
tempPath = pio.os.path.join(pio.ConstantPyIOmicaExamplesDirectory, 'MathIOmicaExamples')

rnaExample = readMathIOmicaData(pio.os.path.join(tempPath, 'rnaExample'))
print('rnaExample:', str(rnaExample)[:400], '\t. . .\n')

proteinClassificationExample = readMathIOmicaData(pio.os.path.join(tempPath, 'proteinClassificationExample'))
print('proteinClassificationExample:', str(proteinClassificationExample)[:400], '\t. . .\n')

proteinTimeSeriesExample = readMathIOmicaData(pio.os.path.join(tempPath, 'proteinTimeSeriesExample'))
print('proteinTimeSeriesExample:', str(proteinTimeSeriesExample)[:400], '\t. . .\n')

proteinExample = readMathIOmicaData(pio.os.path.join(tempPath, 'proteinExample'))
print('proteinExample:', str(proteinExample)[:400], '\t. . .\n')

metabolomicsPositiveModeExample = readMathIOmicaData(pio.os.path.join(tempPath, 'metabolomicsPositiveModeExample'))
print('metabolomicsPositiveModeExample:', str(metabolomicsPositiveModeExample)[:400], '\t. . .\n')

metabolomicsNegativeModeExample = readMathIOmicaData(pio.os.path.join(tempPath, 'metabolomicsNegativeModeExample'))
print('metabolomicsNegativeModeExample:', str(metabolomicsNegativeModeExample)[:400], '\t. . .\n')