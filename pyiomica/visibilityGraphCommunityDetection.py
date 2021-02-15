"""Visibility Graph Community detection functions.
"""
from .globalVariables import *
import numpy as np
import networkx as nx

def createVisibilityGraph(data, times, graph_type='natural', weight=None, withsign=False):

    """Calculate adjacency matrix of visibility graph, create the networkx.Graph network.

    Parameters:
        data: 2d numpy.array
            Data to process
        
        times: 1d numpy.array
            Times corresponding to provided data points
        
        weight: str, Default None
            Type of normalization:
                None: no weighted

                'time': weight = abs(times[i] - times[j])

                'tan': weight = abs((data[i] - data[j])/(times[i] - times[j])) + 10**(-8)

                'distance': weight = A[i, j] = A[j, i] = ((data[i] - data[j])**2 + (times[i] - times[j])**2)**0.5
            
        graph_type: str, Default 'natural'
            Type of the visibility graph:
                "horizontal", Horizontal Visibility Graph

                "natural", Natural Visibility Graph

                "dual_horizontal", Dual Perspective Horizontal Visibility Graph

                "dual_natural", Dual Perspective Natural Visibility Graph 
                
        withsign: boolean, Default False

            Whether to return the sign of adjacency matrix, 
            the link from normal perspective VG is positive,
            the link from reflected perspective VG is negative 


    Returns: tuple
        Tuple of two objects:
            networkx.Graph
                Graph of networkx type
            
            2d numpy.array
                Adjacency matrix

    Usage:
        A = createVisibilityGraph(data, times)
    """
    
    idx_nan = np.argwhere(np.isnan(data))
    ndata = np.delete(data, idx_nan)
    ntimes = np.delete(times, idx_nan)
    if graph_type not in ['horizontal','dual_horizontal','dual_natural','natural']:
        print('Unknown graph type: %s, adjust graph type to natural'%(graph_type))
    else:
        print('graph type is: %s' %(graph_type))
        
    if weight not in ['time', 'tan', 'distance', None]:
        print('Unknown weight type: %s, adjust weight to None'%(weight))
    else:
        print('weight is: %s' %(weight))
    
    if graph_type == "horizontal":
        AdMatrixOfVisibilityGraph = np.asmatrix(__getAdjacencyMatrixOfHorizontalVisibilityGraph(ndata, ntimes, weight=weight))
    elif graph_type == "dual_horizontal":
        AdMatrixOfVisibilityGraph = np.asmatrix(__getAdjacencyMatrixOfHorizontalVisibilityGraph_dual(ndata, ntimes, weight=weight, withsign=withsign))
    elif graph_type == "dual_natural":
        AdMatrixOfVisibilityGraph = np.asmatrix(__getAdjacencyMatrixOfVisibilityGraph_dual(ndata, ntimes, weight=weight, withsign=withsign))
    else:
        AdMatrixOfVisibilityGraph = np.asmatrix(__getAdjacencyMatrixOfVisibilityGraph(ndata, ntimes, weight=weight))
        
    G = nx.convert_matrix.from_numpy_matrix(abs(AdMatrixOfVisibilityGraph))
    
    labels = {}
    intensity = {} 
    for idx, node in enumerate(G.nodes()):
        labels[node] = str(np.round(ntimes[idx],2))
        intensity[node] = ndata[idx]
    nx.set_node_attributes(G, labels, 'timepoint')
    nx.set_node_attributes(G, intensity, 'intensity')
    
    
    return G,AdMatrixOfVisibilityGraph

def __getAdjacencyMatrixOfHorizontalVisibilityGraph(data, times, weight=None):

    """Calculate adjacency matrix of horizontal visibility graph.

    Parameters:
        data: 2d numpy.array
            Data to process
        
        times: 1d numpy.array
            Times corresponding to provided data points
        
        weight: str, Default None
            Type of normalization:
                None: no weighted

                'time': weight = abs(times[i] - times[j])

                'tan': weight = abs((data[i] - data[j])/(times[i] - times[j])) + 10**(-8)

                'distance': weight = A[i, j] = A[j, i] = ((data[i] - data[j])**2 + (times[i] - times[j])**2)**0.5

    Returns:
        2d numpy.array
            Adjacency matrix

    Usage:
        A = __getAdjacencyMatrixOfHorizontalVisibilityGraph(data, times)
    """

    dimension = len(data)

    A = np.zeros((dimension,dimension))

    for i in range(dimension):
        if i<dimension-1:
            if weight == 'time':
                A[i, i+1] = A[i+1, i] = abs(times[i] - times[i+1])
            elif weight == 'tan':
                A[i, i+1] = A[i+1, i] = abs((data[i] - data[i+1])/(times[i] - times[i+1])) + 10**(-8)
            elif weight == 'distance':
                A[i, i+1] = A[i+1, i] = ((data[i] - data[i+1])**2 + (times[i] - times[i+1])**2)**0.5
            else:
                A[i, i+1] = A[i+1, i] = 1


        for j in range(i + 2, dimension):
            if np.max(data[i+1:j]) < min(data[i], data[j]):
                if  weight == 'time':
                    A[i,j] = A[j,i] = abs(times[i] - times[j])
                elif weight == 'tan':
                    A[i, j] = A[j, i] = abs((data[i] - data[j])/(times[i] - times[j])) + 10**(-8)
                elif weight == 'distance':
                    A[i, j] = A[j, i] = ((data[i] - data[j])**2 + (times[i] - times[j])**2)**0.5
                else:
                    A[i, j] = A[j, i] = 1
    return A

def __getAdjacencyMatrixOfHorizontalVisibilityGraph_dual(data, times, weight=None, withsign=False):

    """Calculate adjacency matrix of dual perspective horizontal visibility graph.

    Parameters:
        data: 2d numpy.array
            Data to process
        
        times: 1d numpy.array
            Times corresponding to provided data points
        
        weight: str, Default None
            Type of normalization:
                None: no weighted

                'time': weight = abs(times[i] - times[j])

                'tan': weight = abs((data[i] - data[j])/(times[i] - times[j])) + 10**(-8)

                'distance': weight = A[i, j] = A[j, i] = ((data[i] - data[j])**2 + (times[i] - times[j])**2)**0.5
            
        withsign: boolean, Default False

            Whether to return the sign of adjacency matrix, the link from normal perspective VG is positive,
            the link from reflected perspective VG is negative        

        
    Returns:
        2d numpy.array
            Adjacency matrix

    Usage:
        A = __getAdjacencyMatrixOfHorizontalVisibilityGraph_dual(data, times)
    """

    A_posi = np.asmatrix(__getAdjacencyMatrixOfHorizontalVisibilityGraph(data, times, weight=weight))
    A_nega = np.asmatrix(__getAdjacencyMatrixOfHorizontalVisibilityGraph(-data, times, weight=weight))
    A_dual = np.where(A_posi >= A_nega, A_posi, -A_nega)
    
    if withsign == True:
        A_dual = A_dual
    else:
        A_dual = abs(A_dual)
    
    return A_dual

def __getAdjacencyMatrixOfVisibilityGraph(data, times, weight=None):

    """Calculate adjacency matrix of natural visibility graph.

    Parameters:
        data: 2d numpy.array
            Data to process
        
        times: 1d numpy.array
            Times corresponding to provided data points
        
        weight: str, Default None
            Type of normalization:
                None: no weighted

                'time': weight = abs(times[i] - times[j])

                'tan': weight = abs((data[i] - data[j])/(times[i] - times[j])) + 10**(-8)

                'distance': weight = A[i, j] = A[j, i] = ((data[i] - data[j])**2 + (times[i] - times[j])**2)**0.5

    Returns:
        2d numpy.array
            Adjacency matrix

    Usage:
        A = __getAdjacencyMatrixOfVisibilityGraph(data, times)
    """

    dimension = len(data)

    V = (np.subtract.outer(data, data))/(np.subtract.outer(times, times) + np.identity(dimension))
    

    A = np.zeros((dimension,dimension))

    for i in range(dimension):
        if i<dimension-1:
            if weight == 'time':
                A[i, i+1] = A[i+1, i] = abs(times[i] - times[i+1])
            elif weight == 'tan':
                A[i, i+1] = A[i+1, i] = abs((data[i] - data[i+1])/(times[i] - times[i+1])) + 10**(-8)
            elif weight == 'distance':
                A[i, i+1] = A[i+1, i] = ((data[i] - data[i+1])**2 + (times[i] - times[i+1])**2)**0.5
            else:
                A[i, i+1] = A[i+1, i] = 1
                
        for j in range(i + 2, dimension):
            if np.max(V[i+1:j,i]) < V[j,i]:
                if  weight == 'time':
                    A[i,j] = A[j,i] = abs(times[i] - times[j])
                elif weight == 'tan':
                    A[i, j] = A[j, i] = abs((data[i] - data[j])/(times[i] - times[j])) + 10**(-8)
                elif weight == 'distance':
                    A[i, j] = A[j, i] = ((data[i] - data[j])**2 + (times[i] - times[j])**2)**0.5
                else:
                    A[i, j] = A[j, i] = 1

    return A

def __getAdjacencyMatrixOfVisibilityGraph_dual(data, times, weight=None, withsign=False):

    """Calculate adjacency matrix of dual perspective natural visibility graph.

    Parameters:
        data: 2d numpy.array
            Data to process
        
        times: 1d numpy.array
            Times corresponding to provided data points
        
        weight: str, Default None
            Type of normalization:
                None: no weighted

                'time': weight = abs(times[i] - times[j])

                'tan': weight = abs((data[i] - data[j])/(times[i] - times[j])) + 10**(-8)

                'distance': weight = A[i, j] = A[j, i] = ((data[i] - data[j])**2 + (times[i] - times[j])**2)**0.5
            
        withsign: boolean, Default False

            Whether to return the sign of adjacency matrix, the link from normal perspective VG is positive,
            the link from reflected perspective VG is negative 
            

    Returns:
        2d numpy.array
            Adjacency matrix

    Usage:
        A = __getAdjacencyMatrixOfVisibilityGraph_dual(data, times)
    """

    A_posi = np.asmatrix(__getAdjacencyMatrixOfVisibilityGraph(data, times, weight=weight))
    A_nega = np.asmatrix(__getAdjacencyMatrixOfVisibilityGraph(-data, times, weight=weight))
    if min(data)<0:
        A_dual = np.where(A_posi >= A_nega, A_posi, -A_nega)
    else:
        A_dual = A_posi
    if withsign == True:
        A_dual = A_dual
    else:
        A_dual = abs(A_dual)

    return A_dual

def communityDetectByPathLength(G, setSourceTarget=None,direction=None, cutoff=None):
    
    """Calculate community structure by shortest path length algorithm.
    
    Parameters:
        G: networkx.Graph
            Graph of networkx type
    
        direction:str, default None
            The direction that nodes aggregate to communities:
                None: no specific direction, e.g. both sides.
        
                'left': nodes can only aggregate to the left side hubs, e.g. early hubs
        
                'right': nodes can only aggregate to the right side hubs, e.g. later hubs
    
        cutoff: int, float or str, Default None
            Cutoff is used to combine initial communities, e.g. whenever the shortest path length of two adjacent hub nodes is smaller than cutoff, the communities with the two hub nodes will be combined:
        
                int or float: the percentile of all shortest path length distribution, between 0 ~ 100
        
                'auto': use optimized cutoff
        
                None: no cutoff
        
    Returns:
        list of list
            Detected communities in the form of nested list.

    Usage:
        c = communityDetectByPathLength(G)
    """

    if direction not in ['left', 'right', None]:
        print('Unknown direction type: %s, adjust direction to None'%(direction))
        direction = None
    else:
        print('direction type is: %s' %(direction))
        
        
    PL_dict = dict(nx.all_pairs_dijkstra_path_length(G)) # get shortest path length
    value_PL_list = []
    
    Nodes_PL_dict = list(PL_dict.keys())
    for node1 in range(len(PL_dict)):
        pl_node1 = PL_dict[Nodes_PL_dict[node1]]
        for node2 in range((node1+1), len(PL_dict)):
            value_PL_list.append(pl_node1[Nodes_PL_dict[node2]]) # get all path length value list
    
    nodelist = list(G.nodes)
    nodelist.sort()
    if setSourceTarget != None:
        PL_node0_node_end = nx.dijkstra_path(G, setSourceTarget[0], setSourceTarget[-1])
    else: 
        PL_node0_node_end = nx.dijkstra_path(G, nodelist[0], nodelist[-1]) #get path length from start to end time point
    
    community = {}

    sort_spl = PL_node0_node_end[:]
    sort_spl.sort()
    nodes_list = list(set(G.nodes) - set(PL_node0_node_end))

    print('the shortest path is:', sort_spl)
    
    #build community choose core as the nodes on the shortest path 
    #from start time point to end time point 
    for pl_node in sort_spl:
        community[pl_node] = [pl_node]
    
    #add nodes to community if the path length from nodes to core shorter than cutoff 
    for node in nodes_list:
        if direction == 'left':
            temp_spl = [i for i in sort_spl if i < node]
        elif direction == 'right':
            temp_spl = [i for i in sort_spl if i > node]
        else:
            temp_spl = sort_spl
            

        pl_to = {key:PL_dict[node][key] for key in temp_spl}
        pl_to_value = list(set(pl_to.values()))
        
        if 0 in pl_to_value:
            shortest_pl = min(pl_to_value.remove(0))
        else:
            shortest_pl = min(pl_to_value)
        c_key = []
        
        for k, v in pl_to.items():
            if v == shortest_pl:
                c_key.append(k)
        c_id = min(c_key)
        community[c_id].append(node)
    
    
    #merge communities if the path length between the cores shorter then cutoff
    if type(cutoff) == int or type(cutoff) == float:
        if 0.0 <= cutoff and cutoff <= 100.0:            
            cut = np.percentile(value_PL_list,cutoff)
            print('current percentiles cutoff is:', cutoff)
        else:
            cut = None
            print('Cutoff %f is out of range, adjust cutoff to None'%(cutoff))
    elif cutoff == 'auto':
        cut = __optimize_cutoff(PL_dict, value_PL_list, sort_spl)
        if max(value_PL_list) != min(value_PL_list):
            percentiles = (cut - min(value_PL_list)) / (max(value_PL_list) - min(value_PL_list)) * 100
        else:
            percentiles = 0
        print('current cutoff is auto, the optimized percentiles cutoff is %f ' %(percentiles))
    elif cutoff is None:
        cut = None
        print('current cutoff is None')
    else:
        cut = None
        print('Unknown cutoff type: %s, adjust cutoff to None'%(cutoff))

    if cut is not None:
            
        sort_spl_backup = sort_spl[:]
        
        while len(sort_spl) > 1:
            A = sort_spl[0]
            for B in sort_spl[1:]:
                if PL_dict[A][B] <= cut:
                    community[A].extend(community[B])
                    sort_spl_backup.remove(B)
                    del community[B]
            sort_spl_backup.remove(A)
            sort_spl = sort_spl_backup[:]
        
    c = list(community.values())
    for row in c:
        row.sort()

        
    return c

def __optimize_cutoff(PL_dict, value_PL_list, sort_spl):

    """Calculate the optimized cutoff, which will be used to combine initial communities. This function is used internally only.
    """

    L = len(sort_spl)
    plvalue = [min(value_PL_list)]
    for i in range (0,L-1):
        n1 = sort_spl[i]
        n2 = sort_spl[i+1]
        plvalue.append(PL_dict[n1][n2])
    plvalue.append(max(value_PL_list))
    plvalue.sort()
    plv_diff = [j-i for i, j in zip(plvalue[:-1], plvalue[1:])]
    
    index = plv_diff.index(max(plv_diff))
    plv_diff_copy = plv_diff[:]
    tcut = plvalue[index]

    while tcut > np.percentile(value_PL_list,50) and len(plv_diff)>1:
        plv_diff.pop()
        index = plv_diff_copy.index(max(plv_diff))
        tcut = plvalue[index]        


    if tcut <= np.percentile(value_PL_list,50):
        cut = tcut
    else:
        cut = None
    
    return cut
