'''Clustering-related functions'''

import scipy.signal
import scipy.spatial.distance
from scipy.interpolate import UnivariateSpline

from sklearn.cluster import AgglomerativeClustering, KMeans, MiniBatchKMeans
import sklearn.metrics

from .globalVariables import *

from . import (coreFunctions,
               visibilityGraphAuxiliaryFunctions,
               visibilityGraphCommunityDetection,
               utilityFunctions)


def getEstimatedNumberOfClusters(data, cluster_num_min, cluster_num_max, trials_to_do, numberOfAvailableCPUs=4, plotID=None, printScores=False):

    """ Get estimated number of clusters using ARI with KMeans

    Parameters:
        data: 2d numpy.array
            Data to analyze

        cluster_num_min: int
            Minimum possible number of clusters

        cluster_num_max: int
            Maximum possible number of clusters

        trials_to_do: int
            Number of trials to do in ARI function

        numberOfAvailableCPUs: int, Default 4
            Number of processes to run in parallel

        plotID: str, Default None
            Label for the plot of peaks

        printScores: boolean, Default False
            Whether to print all scores

    Returns: 
        tuple
            Largest peak, other possible peaks.

    Usage:
        n_clusters = getEstimatedNumberOfClusters(data, 1, 20, 25)
    """

    def getPeakPosition(scores, makePlot=False, plotID=None):

        print()

        spline = UnivariateSpline(scores.T[0], scores.T[1])
        spline.set_smoothing_factor(0.005)
        xs = np.linspace(scores.T[0][0], scores.T[0][-1], 1000)
        data = np.vstack((xs, spline(xs))).T

        data_all = data.copy()
        data = data[data.T[0] > 4.]
        peaks = scipy.signal.find_peaks(data.T[1])[0]

        if len(peaks) == 0:
            selected_peak = 5
            print('WARNING: no peak found')
        else:
            selected_peak = np.round(data.T[0][peaks[np.argmax(data.T[1][peaks])]],0).astype(int)

        selected_peak_value = scores.T[1][np.argwhere(scores.T[0] == selected_peak)[0][0]]
        peaks = np.round(data.T[0][peaks],0).astype(int) if len(peaks) != 0 else peaks

        #if makePlot:
        #    makePlotOfPeak(data_all, scores, selected_peak, selected_peak_value, plotID)

        print(selected_peak, peaks)

        return selected_peak, peaks

    print('Testing data clustering in a range of %s-%s clusters' % (cluster_num_min,cluster_num_max))
                
    scores = utilityFunctions.runCPUs(numberOfAvailableCPUs, runForClusterNum, [(cluster_num, copy.deepcopy(data), trials_to_do) for cluster_num in range(cluster_num_min, cluster_num_max + 1)])

    if printScores: 
        print(scores)
                
    return getPeakPosition(scores, makePlot=True, plotID=plotID)[0]

def getNClustersFromLinkageElbow(Y):

    """ Get optimal number clusters from linkage.
    A point of the highest accelleration of the fusion coefficient of the given linkage.

    Parameters:
        Y: 2d numpy.array
            Linkage matrix

    Returns:
        int
            Optimal number of clusters

    Usage:
        n_clusters = getNClustersFromLinkageElbow(Y)
    """

    return np.diff(np.array([[nc, Y[-nc + 1][2]] for nc in range(2,min(50,len(Y)))]).T[1], 2).argmax() + 1 if len(Y) >= 5 else 1

def getNClustersFromLinkageSilhouette(Y, data, metric):

    """Determine the optimal number of cluster in data maximizing the Silhouette score.

    Parameters:
        Y: 2d numpy.array
            Linkage matrix

        data: 2d numpy.array
            Data to analyze

        metric: str or function
            Distance measure

    Returns:
        int
            Optimal number of clusters

    Usage:
        n_clusters = getNClustersFromLinkageSilhouette(Y, data, 'euclidean')
    """

    max_score = 0
    n_clusters = 1

    distmatrix = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(data, metric=metric))

    for temp_n_clusters in range(2,10):
        print(temp_n_clusters, end='a, ', flush=True)
        temp_clusters = scipy.cluster.hierarchy.fcluster(Y, t=temp_n_clusters, criterion='maxclust')

        print(temp_n_clusters, end='b, ', flush=True)
        temp_score = sklearn.metrics.silhouette_score(distmatrix, temp_clusters, metric=metric)

        if temp_score>max_score:
            max_score = temp_score
            n_clusters = temp_n_clusters

    return n_clusters - 1

def runForClusterNum(arguments):
    
    """Calculate Adjusted Rand Index of the data for a range of cluster numbers.

    Parameters:
        arguments: tuple
            A tuple of three parameters int the form (cluster_num, data_array, trials_to_do), where
                cluster_num: int
                    Maximum number of clusters

                data_array: 2d numpy.array
                    Data to test

                trials_to_do: int
                    Number of trials for each cluster number

    Returns:
        1d numpy.array
            Numpy array

    Usage:
        instPool = multiprocessing.Pool(processes = NumberOfAvailableCPUs)

        scores = instPool.map(runForClusterNum, [(cluster_num, copy.deepcopy(data), trials_to_do) for cluster_num in range(cluster_num_min, cluster_num_max + 1)])

        instPool.close()

        instPool.join()
    """

    np.random.seed()

    cluster_num, data_array, trials_to_do = arguments

    print(cluster_num, end=', ', flush=True)

    labels = [KMeans(n_clusters=cluster_num).fit(data_array).labels_ for i in range(trials_to_do)]

    agreement_matrix = np.zeros((trials_to_do,trials_to_do))

    for i in range(trials_to_do):
        for j in range(trials_to_do):
            agreement_matrix[i, j] = sklearn.metrics.adjusted_rand_score(labels[i], labels[j]) if agreement_matrix[j, i] == 0 else agreement_matrix[j, i]

    selected_data = agreement_matrix[np.triu_indices(agreement_matrix.shape[0],1)]

    return np.array((cluster_num, np.mean(selected_data), np.std(selected_data)))

def getGroupingIndex(data, n_groups=None, method='weighted', metric='correlation', significance='Elbow'):

    """Cluster data into N groups, if N is provided, else determine N
    return: linkage matrix, cluster labels, possible cluster labels.

    Parameters:
        data: 2d numpy.array
            Data to analyze

        n_groups: int, Default None
            Number of groups to split data into

        method: str, Default 'weighted'
            Linkage calculation method

        metric: str, Default 'correlation'
            Distance measure

        significance: str, Default 'Elbow'
            Method for determining optimal number of groups and subgroups

    Returns:
        tuple
            Linkage matrix, cluster index, possible groups

    Usage:
        x, y, z = getGroupingIndex(data, method='weighted', metric='correlation', significance='Elbow')
    """

    Y = hierarchy.linkage(data, method=method, metric=metric, optimal_ordering=False)

    if n_groups == None:
        if significance=='Elbow':
            n_groups = getNClustersFromLinkageElbow(Y)
        elif significance=='Silhouette':
            n_groups = getNClustersFromLinkageSilhouette(Y, data, metric)

    print('n_groups:', n_groups)

    labelsClusterIndex = scipy.cluster.hierarchy.fcluster(Y, t=n_groups, criterion='maxclust')

    groups = np.sort(np.unique(labelsClusterIndex))

    print([np.sum(labelsClusterIndex == group) for group in groups])

    return Y, labelsClusterIndex, groups

def makeClusteringObject(df_data, df_data_autocorr, method='weighted', metric='correlation', significance='Elbow'):

    """Make a clustering Groups-Subgroups dictionary object.

    Parameters:
        df_data: pandas.DataFrame
            Data to analyze in DataFrame format

        df_data_autocorr: pandas.DataFrame
            Autocorrelations or periodograms in DataFrame format

        method: str, Default 'weighted'
            Linkage calculation method

        metric: str, Default 'correlation'
            Distance measure

        significance: str, Default 'Elbow'
            Method for determining optimal number of groups and subgroups

    Returns:
        dictionary
            Clustering object

    Usage:
        myObj = makeClusteringObject(df_data, df_data_autocorr, significance='Elbow')
    """

    def getSubgroups(df_data, metric, significance):

        Y = hierarchy.linkage(df_data.values, method=method, metric=metric, optimal_ordering=True)
        leaves = hierarchy.dendrogram(Y, no_plot=True)['leaves']

        if significance=='Elbow':
            n_clusters = getNClustersFromLinkageElbow(Y)
        elif significance=='Silhouette':
            n_clusters = getNClustersFromLinkageSilhouette(Y, df_data.values, metric)

        print('n_subgroups:', n_clusters)

        clusters = scipy.cluster.hierarchy.fcluster(Y, t=n_clusters, criterion='maxclust')[leaves]

        return {cluster:df_data.index[leaves][clusters==cluster] for cluster in np.unique(clusters)}, Y

    ClusteringObject = {}

    try:
        grouping = getGroupingIndex(df_data_autocorr.values, method=method, metric=metric, significance=significance)
    except Exception as exception:
        print(exception)
        print('Returning None')
        return None
        
    ClusteringObject['linkage'], labelsClusterIndex, groups = grouping

    for group in groups:
        signals = df_data.index[labelsClusterIndex==group]

        ClusteringObject[group], ClusteringObject[group]['linkage'] = ({1: signals}, None) if len(signals)==1 else getSubgroups(df_data.loc[signals], coreFunctions.metricCommonEuclidean, significance)

        for subgroup in sorted([item for item in list(ClusteringObject[group].keys()) if not item=='linkage']):
            ClusteringObject[group][subgroup] = {'order':[np.where([temp==signal for temp in df_data.index.values])[0][0] for signal in list(ClusteringObject[group][subgroup])],
                                                 'data':df_data.loc[ClusteringObject[group][subgroup]], 
                                                 'dataAutocorr':df_data_autocorr.loc[ClusteringObject[group][subgroup]]}

    return ClusteringObject

def exportClusteringObject(ClusteringObject, saveDir, dataName, includeData=True, includeAutocorr=True):

    """Export a clustering Groups-Subgroups dictionary object to a SpreadSheet.
    Linkage data is not exported.

    Parameters:
        ClusteringObject: dictionary
            Clustering object

        saveDir: str
            Path of directories to save the object to

        dataName: str
            Label to include in the file name

        includeData: boolean, Default True
            Export data

        includeAutocorr: boolean, Default True
            Export autocorrelations of data

    Returns:
        str
            File name of the exported clustering object

    Usage:
        exportClusteringObject(myObj, '/dir1', 'myObj')
    """

    if not os.path.exists(saveDir):
        os.makedirs(saveDir)

    fileName = saveDir + dataName + '_GroupsSubgroups.xlsx'

    writer = pd.ExcelWriter(fileName)

    for group in sorted([item for item in list(ClusteringObject.keys()) if not item=='linkage']):
        for subgroup in sorted([item for item in list(ClusteringObject[group].keys()) if not item=='linkage']):

            df_data = ClusteringObject[group][subgroup]['data'].iloc[::-1]
            df_dataAutocorr = ClusteringObject[group][subgroup]['dataAutocorr'].iloc[::-1]

            if includeData==True and includeAutocorr==True:
                df = pd.concat((df_data,df_dataAutocorr), sort = False, axis=1)
            elif includeData==True and includeAutocorr==False:
                df = df_data
            elif includeData==False and includeAutocorr==True:
                df = df_dataAutocorr
            else:
                df = pd.DataFrame(index=df_data.index)

            df.index.name = 'Index'
            df.to_excel(writer, 'G%sS%s'%(group, subgroup))

    writer.save()

    print('Saved clustering object to:', fileName)

    return fileName

def getCommunitiesOfTimeSeries(data, times, minNumberOfCommunities=2, horizontal=False, method='WDPVG', direction='left',weight='distance'):

    '''Get communities of time series

    Parameters:
        data: 1d numpy.array
            Data array

        times: 1d numpy.array
            Times corresponding to data points

        minNumberOfCommunities: int, Default 2
            Number of communities to find depends on the number of splits.
            This parameter is ignored in methods that automatically
            estimate optimal number of communities.
        
        horizontal: boolean, Default False
            Whether to use horizontal of normal visibility graph

        method: str, Default 'betweenness_centrality'
            Name of the method to use:
                'Girvan_Newman': edge betweenness centrality based approach

                'betweenness_centrality': reflected graph node betweenness centrality based approach

                'WDPVG': weighted dual perspective visibility graph method (also set weight variable)
    
        direction:str, default 'left'
            The direction that nodes aggregate to communities:
                None: no specific direction, e.g. both sides.
                'left': nodes can only aggregate to the left side hubs, e.g. early hubs
                'right': nodes can only aggregate to the right side hubs, e.g. later hubs
        
        weight: str, Default 'distance'
            Type of weight for method='WDPVG':
                None: no weighted

                'time': weight = abs(times[i] - times[j])

                'tan': weight = abs((data[i] - data[j])/(times[i] - times[j])) + 10**(-8)

                'distance': weight = A[i, j] = A[j, i] = ((data[i] - data[j])**2 + (times[i] - times[j])**2)**0.5

    Returns:
        (list, graph)
            List of communities and a networkx graph

    Usage:
        res = getCommunitiesOfTimeSeries(data, times)
    '''

    if method=='betweenness_centrality':
        if horizontal:
            graph_nx = nx.from_numpy_matrix(visibilityGraphAuxiliaryFunctions.getAdjacencyMatrixOfHVG(data))
            graph_nx_inv = nx.from_numpy_matrix(visibilityGraphAuxiliaryFunctions.getAdjacencyMatrixOfHVG(-data))
        else:  
            graph_nx = nx.from_numpy_matrix(visibilityGraphAuxiliaryFunctions.getAdjacencyMatrixOfNVG(data, times))
            graph_nx_inv = nx.from_numpy_matrix(visibilityGraphAuxiliaryFunctions.getAdjacencyMatrixOfNVG(-data, times))
    
        def find_and_remove_node(graph_nx):
            bc = nx.betweenness_centrality(graph_nx)
            node_to_remove = list(bc.keys())[np.argmax(list(bc.values()))]
            graph_nx.remove_node(node_to_remove)
            return graph_nx, node_to_remove

        list_of_nodes = []
        for i in range(minNumberOfCommunities-1):
            graph_nx_inv, node = find_and_remove_node(graph_nx_inv)
            list_of_nodes.append(node)
        
        if not 0 in list_of_nodes:
            list_of_nodes.append(0)

        list_of_nodes.append(list(graph_nx.nodes)[-1] + 1)
        list_of_nodes.sort()

        communities = [list(range(list_of_nodes[i],list_of_nodes[i + 1])) for i in range(len(list_of_nodes) - 1)]

    elif method=='Girvan_Newman':
        if horizontal:
            graph_nx = nx.from_numpy_matrix(visibilityGraphAuxiliaryFunctions.getAdjacencyMatrixOfHVG(data))
        else:
            graph_nx = nx.from_numpy_matrix(visibilityGraphAuxiliaryFunctions.getAdjacencyMatrixOfNVG(data, times))
        generator_of_communities = nx.algorithms.community.centrality.girvan_newman(graph_nx)

        for i in range(minNumberOfCommunities-1):
            communities_for_level = next(generator_of_communities)
        
        communities = list(sorted(c) for c in communities_for_level)

    elif method=='WDPVG':
        if horizontal:
            graph_nx = visibilityGraphCommunityDetection.createVisibilityGraph(data, times, "dual_horizontal", weight=weight)[0]
        else:
            graph_nx = visibilityGraphCommunityDetection.createVisibilityGraph(data, times, "dual_natural", weight=weight)[0]

        communities = visibilityGraphCommunityDetection.communityDetectByPathLength(graph_nx, direction=direction, cutoff='auto')

    else:

        print('Unknown method: %s'%(method))
        
        return None

    return communities, graph_nx