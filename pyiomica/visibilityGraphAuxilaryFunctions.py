'''Functions to generate adjacency matrix of visibility graphs'''

import numba
numba.config.NUMBA_DEFAULT_NUM_THREADS = 4

from .globalVariables import *

@numba.jit(cache=True)
def getAdjacencyMatrixOfNVG(data, times):

    """Calculate adjacency matrix of visibility graph.
    JIT-accelerated version (a bit faster than NumPy-accelerated version).
    Allows use of Multiple CPUs.

    Parameters:
        data: 2d numpy.array
            Numpy array of floats

        times: 1d numpy.array
            Numpy array of floats

    Returns:
        2d numpy.array
            Adjacency matrix

    Usage:
        A = getAdjacencyMatrixOfNVG(data, times)
    """

    dimension = len(data)

    V = np.zeros((dimension,dimension))

    for i in range(dimension):
        for j in range(i + 1, dimension):
            V[i,j] = V[j,i] = (data[i] - data[j]) / (times[i] - times[j])

    A = np.zeros((dimension,dimension))

    for i in range(dimension):
        for j in range(i + 1, dimension):
            no_conflict = True

            for a in list(range(i+1,j)):
                if V[a,i] > V[j,i]:
                    no_conflict = False
                    break

            if no_conflict:
                A[i,j] = A[j,i] = 1

    return A

def getAdjacencyMatrixOfNVGbyNUMPY(data, times):

    """Calculate adjacency matrix of visibility graph.
    NumPy-accelerated version. Somewhat slower than JIT-accelerated version.
    Use in serial applications.

    Parameters:
        data: 2d numpy.array
            Numpy array of floats

        times: 1d numpy.array
            Numpy array of floats

    Returns:
        2d numpy.array
            Adjacency matrix

    Usage:
        A = getAdjacencyMatrixOfNVGbyNUMPY(data, times)
    """

    dimension = len(data)

    V = (np.subtract.outer(data, data))/(np.subtract.outer(times, times) + np.identity(dimension))

    A = np.zeros((dimension,dimension))

    for i in range(dimension):
        if i<dimension-1:
            A[i,i+1] = A[i+1,i] = 1

        for j in range(i + 2, dimension):
            if np.max(V[i+1:j,i])<=V[j,i]:
                A[i,j] = A[j,i] = 1

    return A

@numba.jit(cache=True)
def getAdjacencyMatrixOfHVG(data):

    """Calculate adjacency matrix of horizontal visibility graph.
    JIT-accelerated version (a bit faster than NumPy-accelerated version).
    Single-threaded beats NumPy up to 2k data sizes.
    Allows use of Multiple CPUs.

    Parameters:
        data: 2d numpy.array
            Numpy array of floats

    Returns:
        2d numpy.array
            Adjacency matrix

    Usage:
        A = getAdjacencyMatrixOfHVG(data)
    """

    A = np.zeros((len(data),len(data)))

    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            no_conflict = True

            for a in list(range(i+1,j)):
                if data[a] > data[i] or data[a] > data[j]:
                    no_conflict = False
                    break

            if no_conflict:
                A[i,j] = A[j,i] = 1

    return A

def getAdjacencyMatrixOfHVGbyNUMPY(data):

    """Calculate adjacency matrix of horizontal visibility graph.
    NumPy-accelerated version.
    Use with datasets larger than 2k.
    Use in serial applications.

    Parameters:
        data: 2d numpy.array
            Numpy array of floats

    Returns:
        2d numpy.array
            Adjacency matrix

    Usage:
        A = getAdjacencyMatrixOfHVGbyNUMPY(data)
    """

    dimension = len(data)

    A = np.zeros((dimension,dimension))

    for i in range(dimension):
        if i<dimension-1:
            A[i,i+1] = A[i+1,i] = 1

        for j in range(i + 2, dimension):
            if np.max(data[i+1:j])<=min(data[i], data[j]):
                A[i,j] = A[j,i] = 1

    return A


