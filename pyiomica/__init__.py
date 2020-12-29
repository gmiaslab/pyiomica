"""
PyIOmica is a general omics data analysis Python package, with a focus on the analysis and categorization of longitudinal datasets.

Usage:
    import pyiomica

Notes:
    For additional information visit: https://github.com/gmiaslab/pyiomica and https://mathiomica.org by G. Mias Lab
"""

print("Loading PyIOmica 1.2.4 (https://github.com/gmiaslab/pyiomica by G. Mias Lab)")


from .globalVariables import *

if printPackageGlobalDefaults:

    variables = locals().copy()

    for variable in variables.items():
        if variable[0][:2] != '__': 
            if not type(variable[1]) is  types.ModuleType:
                if not type(variable[1]) is types.FunctionType:
                    if not type(variable[1]) is types.ClassType:
                        print('\n', variable[0], ':\n', variable[1], '\n')