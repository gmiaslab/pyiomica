import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib import cm
import networkx as nx
#import py2cytoscape
#from py2cytoscape.data.cyrest_client import CyRestClient
import os
import timeMeasure
import numba
numba.config.NUMBA_DEFAULT_NUM_THREADS = 1


'''
JIT-accelerated version (a bit faster than NumPy-accelerated version)
Allows use of Multiple CPUs (have not tested it)
'''
def getAdjecencyMatrixOfVisibilityGraph_serial(data, times):

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


'''
JIT-accelerated version (a bit faster than NumPy-accelerated version)
Allows use of Multiple CPUs (have not tested it)
'''
@numba.jit(cache=True)
def getAdjecencyMatrixOfVisibilityGraph(data, times):

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


'''
NumPy-accelerated version
Somewhat slower than JIT-accelerated version
Use in serial applications
'''
def getAdjecencyMatrixOfVisibilityGraph_NUMPY(data, times):

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


'''
JIT-accelerated version
Single-threaded beats NumPy up to 2k data sizes
Allows use of Multiple CPUs (have not tested it)
'''
@numba.jit(cache=True)
def getAdjecencyMatrixOfHorizontalVisibilityGraph(data):

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


'''
NumPy-accelerated version
Use with datasets larger than 2k
Use in serial applications
'''
def getAdjecencyMatrixOfHorizontalVisibilityGraph_NUMPY(data):

    dimension = len(data)

    A = np.zeros((dimension,dimension))

    for i in range(dimension):
        if i<dimension-1:
            A[i,i+1] = A[i+1,i] = 1

        for j in range(i + 2, dimension):
            if np.max(data[i+1:j])<=min(data[i], data[j]):
                A[i,j] = A[j,i] = 1

    return A


'''
# Helper funciton to generate adjacency matrix     
Written by Shuyue - 03 18 2019
Parameters:
t -- 1d time array
x -- 1d numpy array time series of data
'''
@numba.jit(cache=True)
def ad_mx_NUMBA(t, x, algo='nv'):    
    N = len(t)
    aj_mx = np.zeros((N, N)) # an zero matrix at size NxN
    for i in range(N):
        for j in range(i+1, N):
            c = True
            for k in range(i+1, j):
                if algo == 'nv':
                    if x[k] >= x[i] + (x[j]-x[i])*(t[k]-t[i])/(t[j]-t[i]):
                        c = False
                        continue
                if algo == 'hv':
                    if x[k] >= min(x[i], x[j]):
                        c = False
                        continue
            if c:
                aj_mx[i,j] = 1
                aj_mx[j,i] = 1 
    return aj_mx


'''
# Helper funciton to generate adjacency matrix     
Written by Shuyue - 03 18 2019
Parameters:
t -- 1d time array
x -- 1d numpy array time series of data
'''
def ad_mx(t, x, algo='nv'):    
    N = len(t)
    aj_mx = np.zeros((N, N)) # an zero matrix at size NxN
    for i in range(N):
        for j in range(i+1, N):
            c = True
            for k in range(i+1, j):
                if algo == 'nv':
                    if x[k] >= x[i] + (x[j]-x[i])*(t[k]-t[i])/(t[j]-t[i]):
                        c = False
                        continue
                if algo == 'hv':
                    if x[k] >= min(x[i], x[j]):
                        c = False
                        continue
            if c:
                aj_mx[i,j] = 1
                aj_mx[j,i] = 1 
    return aj_mx


def PlotVisibilityGraph(A, data, times, fileName, id):
    fig, ax = plt.subplots(figsize=(8,4))
    ax.bar(times, data, width = 0.03, color='r', align='center', zorder=-np.inf)
    ax.scatter(times, data, color='b')

    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if i>j and A[i,j] == 1:
                ax.annotate(s='', xy=(times[i],data[i]), xytext=(times[j],data[j]), 
                            arrowprops=dict(arrowstyle='<->', shrinkA=0, shrinkB=0,linestyle='--'))

    ax.set_title('%s Time Series'%(id), fontdict={'color': 'k'})
    ax.set_xlabel('Times', fontsize=8)
    ax.set_ylabel('Signal intensity', fontsize=8)
    ax.set_xticks(times)
    ax.set_xticklabels([str(item)[:-2]+' hr' for item in np.round(times,0)],fontsize=10, rotation=90)
    ax.set_yticks([])

    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(True)

    fig.tight_layout()
    fig.savefig(fileName, dpi=600)
    plt.close(fig)

def PlotHorizontaVisibilityGraph(A, data, times, fileName, id):
    fig, ax = plt.subplots(figsize=(8,4))
    ax.bar(times, data, width = 0.03, color='r', align='center', zorder=-np.inf)
    ax.scatter(times, data, color='b')

    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if i>j and A[i,j] == 1:
                level = np.min([data[i],data[j]])
                ax.annotate(s='', xy=(times[i],level), xytext=(times[j],level), 
                            arrowprops=dict(arrowstyle='<->', shrinkA=0, shrinkB=0,linestyle='--'))

    ax.set_title('%s Time Series'%(id), fontdict={'color': 'k'})
    ax.set_xlabel('Times', fontsize=8)
    ax.set_ylabel('Signal intensity', fontsize=8)
    ax.set_xticks(times)
    ax.set_xticklabels([str(item)[:-2]+' hr' for item in np.round(times,0)],fontsize=10, rotation=90)
    ax.set_yticks([])

    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(True)

    fig.tight_layout()
    fig.savefig(fileName, dpi=600)
    plt.close(fig)




def CreateAndSendGraphToCytoscape(cy, A, times, name):
            
    graph_nx = nx.from_numpy_matrix(A)
    nx.set_node_attributes(graph_nx, {i:str(np.round(c*24,0))[:-2]+' hr' for i,c in enumerate(times)}, 'Label')
    nx.set_node_attributes(graph_nx, {i:c*1000 for i,c in enumerate(times)}, 'X location')
    nx.set_node_attributes(graph_nx, {i:0. for i,c in enumerate(times)}, 'Y location')

    graph_cy = cy.network.create_from_networkx(graph_nx,name=name, collection='AstroIO Graphs')
    cy.style.apply(cy.style.create('Curved'), network=graph_cy)
    cy.layout.apply(name='force-directed', network=graph_cy)


if __name__ == '__main__':

    doVisibilityGraphs = True
    sendGraphsToCytoscape = False

    saveDir = 'results_SpaceHealth/VisibilityGraphs_ofClusterMeans/'

    if not os.path.exists(saveDir):
        os.makedirs(saveDir)


    if doVisibilityGraphs:

        for i in range(10):

            data = np.loadtxt('results_SpaceHealth/SLV_Hourly1TimeSeries_mean_cluster_%s.csv'%(i), delimiter=',')
            times = np.loadtxt('results_SpaceHealth/SLV_Hourly1TimeSeries_df_data_transformed_columns.csv', delimiter=',')

            A = getAdjecencyMatrixOfVisibilityGraph(data, times)
            AO = getAdjecencyMatrixOfVisibilityGraph_NUMPY(data, times)

            executionTimes = []

            sizeMax = 1000

            dataAll = np.random.rand(sizeMax)

            for size in range(1,sizeMax,50):

                print('__'*10,size)

                data = dataAll[:size] #np.random.rand(size)
                times = range(size)
                
                A = getAdjecencyMatrixOfVisibilityGraph(data, times)
                sT = timeMeasure.getStartTime()
                A = getAdjecencyMatrixOfVisibilityGraph(data, times)
                tA = timeMeasure.getElapsedTime(sT,digits=3)

                sT = timeMeasure.getStartTime()
                A = getAdjecencyMatrixOfVisibilityGraph_serial(data, times)
                tAser = timeMeasure.getElapsedTime(sT,digits=3)

                sT = timeMeasure.getStartTime()
                AO = getAdjecencyMatrixOfVisibilityGraph_NUMPY(data, times)
                tAO = timeMeasure.getElapsedTime(sT,digits=3)

                #ASN = ad_mx_NUMBA(times, data, algo='nv')
                #sT = timeMeasure.getStartTime()
                #ASN = ad_mx_NUMBA(times, data, algo='nv')
                #tASN = timeMeasure.getElapsedTime(sT,digits=3)

                #sT = timeMeasure.getStartTime()
                #AS = ad_mx(times, data, algo='nv')
                #tAS = timeMeasure.getElapsedTime(sT,digits=3)

                executionTimes.append([size, tA, tAser, tAO]) #, tASN, tAS

                #print('A==AO', (A==AO).all())
                #print('A==AS', (A==AS).all())
                #print('\n')

            executionTimes = np.array(executionTimes)

            fig, ax = plt.subplots()

            ax.plot(executionTimes.T[0], executionTimes.T[1], 'ko-', linewidth=3, markersize=3.5, label='SD serial loops (NUMBA JIT-accelerated)')
            ax.plot(executionTimes.T[0], executionTimes.T[2], 'ko--', linewidth=2, markersize=3, label='SD serial loops')
            ax.plot(executionTimes.T[0], executionTimes.T[3], 'bo-', linewidth=3, markersize=3.5, label='SD NUMPY-accelerated')
            #ax.plot(executionTimes.T[0], executionTimes.T[4], 'ro-', linewidth=3, markersize=3.5, label='SX serial loops (NUMBA JIT-accelerated)')
            #ax.plot(executionTimes.T[0], executionTimes.T[5], 'ro--', linewidth=2, markersize=3, label='SX serial loops')

            ax.set_xlabel('Number of data points in Time Series')
            ax.set_ylabel('Execution time, sec')

            ax.set_xlim((0, np.max(executionTimes.T[0])))
            ax.set_ylim((0, np.max(executionTimes.T[1:3])))

            plt.legend()
            plt.savefig('VG_compare_versions.png', dpi=600)
            plt.show()


            exit()


            AH = getAdjecencyMatrixOfHorizontalVisibilityGraph(data)

            PlotVisibilityGraph(A, saveDir + 'VisibilityGraph_%s.png'%(i), i)
            PlotHorizontaVisibilityGraph(AH, saveDir + 'HorizontalVisibilityGraph_%s.png'%(i), i)

            if sendGraphsToCytoscape:
                cy = CyRestClient()

                CreateAndSendGraphToCytoscape(cy, A, times, 'Visibility Graph')
                CreateAndSendGraphToCytoscape(cy, AH, times, 'Horizontal Visibility Graph')


            
