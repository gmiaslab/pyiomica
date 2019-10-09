'''This module contains global constants used in PyIOmica.
Most of the modules, classes and functions are imported in this module.'''


import numpy as np
import pandas as pd
import networkx as nx
import numba

import os
import appdirs
import zipfile
import pickle
import gzip
import copy
import multiprocessing
import urllib.request
import shutil
import h5py
import pymysql
import datetime
import json

import scipy
import scipy.signal
import scipy.stats
import scipy.cluster.hierarchy as hierarchy
from scipy.spatial.distance import pdist, squareform
from scipy.interpolate import UnivariateSpline

import sklearn
from sklearn.cluster import AgglomerativeClustering, KMeans, MiniBatchKMeans
import sklearn.preprocessing
from sklearn.metrics import silhouette_score, silhouette_samples, pairwise_distances, adjusted_rand_score
from sklearn.manifold import TSNE

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import matplotlib.patches
import matplotlib.collections
from matplotlib import cm

from importlib import resources

numba.config.NUMBA_DEFAULT_NUM_THREADS = 4

printPackageGlobalDefaults = True
'''Whether to print package global defaults listed in this module'''

PackageDirectory = os.path.split(__file__)[0]
'''Package directory'''

try:
    with resources.path('.data','__init__.py') as readIn:
        ConstantPyIOmicaDataDirectory = os.path.dirname(readIn)

    UserDataDirectory = os.path.join(PackageDirectory, "data")
except:
    UserDataDirectory = appdirs.user_data_dir('pyiomica', 'gmiaslab')

    print('Cannot access default site package directory, will instead use ', UserDataDirectory)

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
