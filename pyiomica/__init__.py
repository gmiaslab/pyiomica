from .globalVariables import *

if printPackageGlobalDefaults:

    import types

    variables = locals().copy()

    for variable in variables.items():
        if variable[0][:2] != '__': 
            if not type(variable[1]) is  types.ModuleType:
                if not type(variable[1]) is types.FunctionType:
                    if not type(variable[1]) is types.ClassType:
                        print('\n', variable[0], ':\n', variable[1], '\n')