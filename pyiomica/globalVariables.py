'''This module contains global constants used in PyIOmica.
Some of the modules, classes and functions are imported in this module.'''

import io
import os
import types
import appdirs
import gzip
import copy
import shutil
import zipfile

import numpy as np
np.random.seed(0)

import pandas as pd
import networkx as nx

import scipy
import scipy.stats
import scipy.fftpack
import scipy.cluster.hierarchy as hierarchy

import sklearn


printPackageGlobalDefaults = False
'''Whether to print package global defaults listed in this module'''

PackageDirectory = os.path.dirname(__file__)
'''Package directory'''

UserDataDirectory = os.path.join(PackageDirectory, 'data')

if not os.path.exists(UserDataDirectory):
    print('Cannot access default site package directory, will instead use ', UserDataDirectory)
    UserDataDirectory = appdirs.user_data_dir('pyiomica', 'gmiaslab')

ConstantPyIOmicaDataDirectory = UserDataDirectory
'''ConstantPyIOmicaDataDirectory is a global variable pointing to the PyIOmica data directory.'''

ConstantPyIOmicaExamplesDirectory = os.path.join(UserDataDirectory, "ExampleData")
'''ConstantPyIOmicaExamplesDirectory is a global variable pointing to the PyIOmica example data directory.'''

ConstantPyIOmicaExampleVideosDirectory = os.path.join(UserDataDirectory, "ExampleVideos")
'''ConstantPyIOmicaExampleVideosDirectory is a global variable pointing to the PyIOmica example videos directory.'''

for path in [ConstantPyIOmicaDataDirectory, ConstantPyIOmicaExamplesDirectory, ConstantPyIOmicaExampleVideosDirectory]:
    if not os.path.exists(path):
        os.makedirs(path)
 
del(path)

ConstantGeneDictionary = None
'''ConstantGeneDictionary is a global gene/protein dictionary variable typically created by GetGeneDictionary.'''
