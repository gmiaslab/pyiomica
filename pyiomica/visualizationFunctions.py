'''Visualization functions'''


from .globalVariables import *

def makeDataHistograms(df, saveDir, dataName):

    """Make a histogram for each pandas Series (time point) in a pandas Dataframe.

    Parameters:
        df: pandas.DataFrame
            Data to visualize

        saveDir: str
            Path of directories to save the object to

        dataName: str
            Label to include in the file name

    Returns:
        None

    Usage:
        makeDataHistograms(df, '/dir1', 'myData')
    """

    if not os.path.exists(saveDir):
        os.makedirs(saveDir)

    for timePoint in df.columns[:]:

        timeLabel = (str(np.round(timePoint,3)) + '000000')[:5]

        subset = df[timePoint]
        subset = subset[~np.isnan(subset.values)].values

        N_bins = 100

        range_min = np.min(subset)
        range_max = np.max(subset)

        hist_of_subset = scipy.stats.rv_histogram(np.histogram(subset, bins=N_bins, range=(range_min,range_max)))
        hist_data = hist_of_subset._hpdf / N_bins
        hist_bins = hist_of_subset._hbins

        fig, ax = plt.subplots(figsize=(8,8))

        bar_bin_width = range_max / N_bins

        ax.bar(hist_bins, hist_data[:-1], width=0.9 * bar_bin_width, color='b', align='center')

        ax.set_title('Data @ timePoint: ' + timeLabel, fontdict={'color': 'b'})
        ax.set_xlabel('Gene expression', fontsize=8)
        ax.set_ylabel('Density', fontsize=8)

        ax.set_xlim(range_min - 0.5 * bar_bin_width, range_max + 0.5 * bar_bin_width)

        fig.tight_layout()
        fig.savefig(saveDir + dataName + '_' + timeLabel + '_histogram_of_expression.png', dpi=600)

        plt.close(fig)

    return None

def makeLombScarglePeriodograms(df, saveDir, dataName):
        
    """Make a combined plot of the signal and its Lomb-Scargle periodogram
    for each pandas Series (time point) in a pandas Dataframe.

    Parameters:
        df: pandas.DataFrame
            Data to visualize

        saveDir: str
            Path of directories to save the object to

        dataName: str
            Label to include in the file name

    Returns:
        None

    Usage:
        makeLombScarglePeriodograms(df, '/dir1', 'myData')
    """

    if not os.path.exists(saveDir):
        os.makedirs(saveDir)

    for geneIndex in range(len(df.index[:])):
        
        geneName = df.index[geneIndex].replace(':', '_')

        subset = df.iloc[geneIndex]
        subset = subset[subset > 0.]

        setTimes, setValues, inputSetTimes = subset.index.values, subset.values, df.columns.values

        if len(subset) < 5:

            print(geneName, ' skipped (only %s non-zero point%s), ' % (len(subset), 's' if len(subset) != 1 else ''), end=' ', flush=True)

            continue

        pgram = LombScargle(setTimes, setValues, inputSetTimes, OversamplingRate=100)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 5))

        ax1.plot(setTimes, setValues, 'bo', linewidth=3, markersize=5, markeredgecolor='k', markeredgewidth=2)

        zero_points = np.array(list(set(inputSetTimes) - set(setTimes)))
        ax1.plot(zero_points, np.zeros((len(zero_points),)), 'ro', linewidth=3, markersize=3, markeredgecolor='k', markeredgewidth=0)

        ax1.set_aspect('auto')

        minTime = np.min(inputSetTimes)
        maxTime = np.max(inputSetTimes)
        extraTime = (maxTime - minTime) / 10

        ax1.set_xlim(minTime - extraTime, maxTime + extraTime)
        ax1.set_title('TimeSeries Data')
    
        ax2.plot(2 * np.pi * pgram[0], pgram[1], 'r-', linewidth=1)

        ax2.set_aspect('auto')
        ax2.set_title('Lomb-Scargle periodogram')

        fig.tight_layout()

        fig.savefig(saveDir + dataName + '_' + geneName + '_Lomb_Scargle_periodogram.png', dpi=600)

        plt.close(fig)

    return None

def addVisibilityGraph(data, times, dataName='G1S1', coords=[0.05,0.95,0.05,0.95], 
                       numberOfVGs=1, groups_ac_colors=['b'], fig=None, numberOfCommunities=6, printCommunities=False, 
                       fontsize=None, nodesize=None, level=0.55, commLineWidth=0.5, lineWidth=1.0,
                       withLabel=True, withTitle=False, layout='circle', radius=0.07, noplot=False):

    """Draw a Visibility graph of data on a provided Matplotlib figure.

    Parameters:
        data: 2d numpy.array
            Array of data to visualize

        times: 1d numpy.array
            Times corresponding to each data point, used for labels

        dataName: str, Default 'G1S1'
            label to include in file name

        coords: list, Default [0.05,0.95,0.05,0.95]
            Coordinates of location of the plot on the figure

        numberOfVGs: int, Default 1
            Number of plots to add to this figure

        groups_ac_colors: list, Default ['b']
            Colors corresponding to different groups of graphs

        fig: matplotlib.figure, Default None
            Figure object

        numberOfCommunities: int, Default 6
            Number of communities

        printCommunities: boolean, Default False
            Whether to print communities details to screen

        fontsize: float, Default None
            Size of labels

        nodesize: float, Default None
            Size of nodes

        level: float, Default 0.55
            Distance of the community lines to nodes

        commLineWidth: float, Default 0.5
            Width of the community lines

        lineWidth: float, Default 1.0
            Width of the edges between nodes

        withLabel: boolean, Default True
            Whether to include label on plot

        withTitle: boolean, Default False
            Whether to include title on plot

        layout: str, Default 'circle'
            Type of the layout. Other option is 'line'

        radius: float, Default 0.07
            Radius of the circle

        noplot: boolean, Default False
            Whether to make a plot or only calculate communities

    Returns:
        tuple
            (graph_nx, data, communities)

    Usage:
        addVisibilityGraph(exampleData, exampleTimes, fig=fig, fontsize=16, nodesize=700, 
                            level=0.85, commLineWidth=3.0, lineWidth=2.0, withLabel=False)
    """

    def imputeWithMedian(data):

        data[np.isnan(data)] = np.median(data[np.isnan(data) == False])

        return data

    if len(data.shape)>1:
        data = pd.DataFrame(data=data).apply(imputeWithMedian, axis=1).apply(lambda data: np.sum(data[data > 0.0]) / len(data), axis=0).values

    graph_nx = nx.from_numpy_matrix(getAdjacencyMatrixOfVisibilityGraph(data, times))
    
    def find_and_remove_node(graph_nx):
        bc = nx.betweenness_centrality(graph_nx)
        node_to_remove = list(bc.keys())[np.argmax(list(bc.values()))]
        graph_nx.remove_node(node_to_remove)
        return graph_nx, node_to_remove

    list_of_nodes = []
    graph_nx_inv = nx.from_numpy_matrix(getAdjacencyMatrixOfVisibilityGraph(-data, times))
    for i in range(numberOfCommunities):
        graph_nx_inv, node = find_and_remove_node(graph_nx_inv)
        list_of_nodes.append(node)
        
    if not 0 in list_of_nodes:
        list_of_nodes.append(0)

    list_of_nodes.append(list(graph_nx.nodes)[-1] + 1)
    list_of_nodes.sort()

    communities = [list(range(list_of_nodes[i],list_of_nodes[i + 1])) for i in range(len(list_of_nodes) - 1)]

    if printCommunities:
        print(list_of_nodes, '\n')
        [print(community) for community in communities]
        print()

    if noplot:
        return graph_nx, data, communities

    group = int(dataName[:dataName.find('S')].strip('G'))

    if fontsize is None:
        fontsize = 4. * (8. + 5.) / (numberOfVGs + 5.)
    
    if nodesize is None:
        nodesize = 30. * (8. + 5.) / (numberOfVGs + 5.)

    (x1,x2,y1,y2) = coords
    
    axisVG = fig.add_axes([x1,y1,x2 - x1,y2 - y1])
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('GR', [(0, 1, 0), (1, 0, 0)], N=1000)

    if layout=='line':
        pos = {i:[float(i)/float(len(graph_nx)), 0.5] for i in range(len(graph_nx))}
    else:
        pos = nx.circular_layout(graph_nx)
        keys = np.array(list(pos.keys())[::-1])
        values = np.array(list(pos.values()))
        values = (values - np.min(values, axis=0))/(np.max(values, axis=0)-np.min(values, axis=0))
        keys = np.roll(keys, np.argmax(values.T[1]) - np.argmin(keys))
        pos = dict(zip(keys, values))

    keys = np.array(list(pos.keys()))
    values = np.array(list(pos.values()))

    shortest_path = nx.shortest_path(graph_nx, source=min(keys), target=max(keys))
    shortest_path_edges = [(shortest_path[i],shortest_path[i + 1]) for i in range(len(shortest_path) - 1)]

    if layout=='line':
        for edge in graph_nx.edges:
            l = np.array(pos[edge[0]])
            r = np.array(pos[edge[1]])

            if edge in shortest_path_edges:
                axisVG.add_artist(matplotlib.patches.Wedge((l+r)/2., 0.5*np.sqrt((l-r)[0]*(l-r)[0]+(l-r)[1]*(l-r)[1]), 0, 180, fill=False, edgecolor='y', linewidth=0.5*3.*lineWidth, alpha=0.7, width=0.001))

            axisVG.add_artist(matplotlib.patches.Wedge((l+r)/2., 0.5*np.sqrt((l-r)[0]*(l-r)[0]+(l-r)[1]*(l-r)[1]), 0, 180, fill=False, edgecolor='k', linewidth=0.5*lineWidth, alpha=0.7, width=0.001))

        nx.draw_networkx(graph_nx, pos=pos, ax=axisVG, node_color='y', edge_color='y', node_size=nodesize * 1.7, width=0., nodelist=shortest_path, edgelist=shortest_path_edges, with_labels=False)
        nx.draw_networkx(graph_nx, pos=pos, ax=axisVG, node_color=data, cmap=cmap, alpha=1.0, font_size=fontsize,  width=0., font_color='k', node_size=nodesize)
    else:
        nx.draw_networkx(graph_nx, pos=pos, ax=axisVG, node_color='y', edge_color='y', node_size=nodesize * 1.7, width=3.0*lineWidth, nodelist=shortest_path, edgelist=shortest_path_edges, with_labels=False)
        nx.draw_networkx(graph_nx, pos=pos, ax=axisVG, node_color=data, cmap=cmap, alpha=1.0, font_size=fontsize,  width=lineWidth, font_color='k', node_size=nodesize)

    if layout=='line':
        xmin, xmax = (-1.,1.)
        ymin, ymax = (-1.,1.)
    else:
        xmin, xmax = axisVG.get_xlim()
        ymin, ymax = axisVG.get_ylim()

    X, Y = np.meshgrid(np.arange(xmin, xmax, (xmax - xmin) / 300.), np.arange(ymin, ymax, (ymax - ymin) / 300.))

    def smooth(Z, N=7.):
        for ix in range(1,Z.shape[0]-1,1):
            Z[ix] = ((N-1.)*Z[ix] + (Z[ix-1] + Z[ix+1])/2.)/N
        return Z

    for icommunity, community in enumerate(communities):
        Z = np.exp(X ** 2 - Y ** 2) * 0.
        nX, nY = tuple(np.array([pos[node] for node in community]).T)
        for i in range(len(community)-1):
            p1, p2 = np.array([nX[i], nY[i]]), np.array([nX[i+1], nY[i+1]])

            for j in range(-2, 32):
                pm = p1 + (p2-p1)*float(j)/30.
                Z[np.where((X-pm[0])**2+(Y-pm[1])**2<=radius**2)] = 1.
        
        for _ in range(20):
            Z = smooth(smooth(Z).T).T

        CS = axisVG.contour(X, Y, Z, [level], linewidths=commLineWidth, alpha=0.8, colors=groups_ac_colors[group - 1])
        #axisVG.clabel(CS, inline=True,fontsize=4,colors=group_colors[group-1], fmt ={level:'C%s'%icommunity})

    if layout=='line':
        axisVG.set_xlim(-0.1,1.)
        axisVG.set_ylim(-0.1,1.)

    axisVG.spines['left'].set_visible(False)
    axisVG.spines['right'].set_visible(False)
    axisVG.spines['top'].set_visible(False)
    axisVG.spines['bottom'].set_visible(False)
    axisVG.set_xticklabels([])
    axisVG.set_yticklabels([])
    axisVG.set_xticks([])
    axisVG.set_yticks([])

    if withLabel:
        axisVG.text(axisVG.get_xlim()[1], (axisVG.get_ylim()[1] + axisVG.get_ylim()[0]) * 0.5, dataName, ha='left', va='center',
                    fontsize=8).set_path_effects([path_effects.Stroke(linewidth=0.4, foreground=groups_ac_colors[group - 1]),path_effects.Normal()])

    if withTitle:
        titleText = dataName + ' (size: ' + str(data.shape[0]) + ')' + ' min=%s max=%s' % (np.round(min(data),2), np.round(max(data),2))
        axisVG.set_title(titleText, fontsize=10)

    return graph_nx, data, communities

def makeDendrogramHeatmap(ClusteringObject, saveDir, dataName, AutocorrNotPeriodogr=True, textScale=1.0, vectorImage=True):

    """Make Dendrogram-Heatmap plot along with VIsibility graphs.

    Parameters:
        ClusteringObject: 
            Clustering object

        saveDir: str
            Path of directories to save the object to

        dataName: str
            Label to include in the file name

        AutocorrNotPeriodogr: boolean, Default True
            Whether to use autocorrelation method instead of periodograms

        textScale: float, Default 1.0
            scaling of text size

        vectorImage: boolean, Default True
            Whether to make vector graphics instead of raster 

    Returns:
        None

    Usage:
        makeDendrogramHeatmap(myObj, '/dir1', 'myData', AutocorrNotPeriodogr=True)
    """

    def addAutocorrelationDendrogramAndHeatmap(ClusteringObject, groupColors, fig, AutocorrNotPeriodogr=AutocorrNotPeriodogr):

        axisDendro = fig.add_axes([0.68,0.1,0.17,0.8], frame_on=False)

        n_clusters = len(ClusteringObject.keys()) - 1
        hierarchy.set_link_color_palette(groupColors[:n_clusters]) #gist_ncar #nipy_spectral #hsv
        origLineWidth = matplotlib.rcParams['lines.linewidth']
        matplotlib.rcParams['lines.linewidth'] = 0.5
        Y_ac = ClusteringObject['linkage']
        Z_ac = hierarchy.dendrogram(Y_ac, orientation='left',color_threshold=Y_ac[-n_clusters + 1][2])
        hierarchy.set_link_color_palette(None)
        matplotlib.rcParams['lines.linewidth'] = origLineWidth
        axisDendro.set_xticks([])
        axisDendro.set_yticks([])

        posA = ((axisDendro.get_xlim()[0] if n_clusters == 1 else Y_ac[-n_clusters + 1][2]) + Y_ac[-n_clusters][2]) / 2
        axisDendro.plot([posA, posA], [axisDendro.get_ylim()[0], axisDendro.get_ylim()[1]], 'k--', linewidth = 1)

        clusters = []
        order = []
        tempData = None
        for group in sorted([item for item in list(ClusteringObject.keys()) if not item=='linkage']):
            for subgroup in sorted([item for item in list(ClusteringObject[group].keys()) if not item=='linkage']):
                subgroupData = ClusteringObject[group][subgroup]['dataAutocorr'].values
                tempData = subgroupData if tempData is None else np.vstack((tempData, subgroupData))
                clusters.extend([group for _ in range(subgroupData.shape[0])])
                order.extend(ClusteringObject[group][subgroup]['order'])

        tempData = tempData[np.argsort(order),:][Z_ac['leaves'],:].T[1:].T
        clusters = np.array(clusters)
        cluster_line_positions = np.where(clusters - np.roll(clusters,1) != 0)[0]

        for i in range(n_clusters - 1):
            posB = cluster_line_positions[i + 1] * (axisDendro.get_ylim()[1] / len(Z_ac['leaves'])) - 5. * 0
            axisDendro.plot([posA, axisDendro.get_xlim()[1]], [posB, posB], '--', color='black', linewidth = 1.0)
        
        axisDendro.plot([posA, axisDendro.get_xlim()[1]], [0., 0.], '--', color='k', linewidth = 1.0)
        axisDendro.plot([posA, axisDendro.get_xlim()[1]], [-0. + axisDendro.get_ylim()[1], -0. + axisDendro.get_ylim()[1]], '--', color='k', linewidth = 1.0)


        axisMatrixAC = fig.add_axes([0.78 + 0.07,0.1,0.18 - 0.075,0.8])

        cmap = plt.cm.bwr
        imAC = axisMatrixAC.imshow(tempData, aspect='auto', vmin=np.min(tempData), vmax=np.max(tempData), origin='lower', cmap=cmap)
        for i in range(n_clusters - 1):
            axisMatrixAC.plot([-0.5, tempData.shape[1] - 0.5], [cluster_line_positions[i + 1] - 0.5, cluster_line_positions[i + 1] - 0.5], '--', color='black', linewidth = 1.0)

        axisMatrixAC.set_xticks([i for i in range(tempData.shape[1] - 1)])
        axisMatrixAC.set_xticklabels([i + 1 for i in range(tempData.shape[1] - 1)], fontsize=6*textScale)
        axisMatrixAC.set_yticks([])
        axisMatrixAC.set_xlabel('Lag' if AutocorrNotPeriodogr else 'Frequency', fontsize=axisMatrixAC.xaxis.label._fontproperties._size*textScale)
        axisMatrixAC.set_title('Autocorrelation' if AutocorrNotPeriodogr else 'Periodogram', fontsize=axisMatrixAC.title._fontproperties._size*textScale)

        axisColorAC = fig.add_axes([0.9 + 0.065,0.55,0.01,0.35])

        axisColorAC.tick_params(labelsize=6*textScale)
        plt.colorbar(imAC, cax=axisColorAC, ticks=[np.round(np.min(tempData),2),np.round(np.max(tempData),2)])

        return

    def addGroupDendrogramAndShowSubgroups(ClusteringObject, groupSize, bottom, top, group, groupColors, fig):

        Y = ClusteringObject[group]['linkage']

        n_clusters = len(sorted([item for item in list(ClusteringObject[group].keys()) if not item=='linkage']))
        print("Number of subgroups:", n_clusters)

        axisDendro = fig.add_axes([left, bottom, dx + 0.005, top - bottom], frame_on=False)

        hierarchy.set_link_color_palette([matplotlib.colors.rgb2hex(rgb[:3]) for rgb in cm.nipy_spectral(np.linspace(0, 0.5, n_clusters + 1))]) #gist_ncar #nipy_spectral #hsv
        origLineWidth = matplotlib.rcParams['lines.linewidth']
        matplotlib.rcParams['lines.linewidth'] = 0.5
        Z = hierarchy.dendrogram(Y, orientation='left',color_threshold=Y[-n_clusters + 1][2])
        hierarchy.set_link_color_palette(None)
        matplotlib.rcParams['lines.linewidth'] = origLineWidth
        axisDendro.set_xticks([])
        axisDendro.set_yticks([])

        posA = axisDendro.get_xlim()[0]/2 if n_clusters == groupSize else (Y[-n_clusters + 1][2] + Y[-n_clusters][2])/2 
        axisDendro.plot([posA, posA], [axisDendro.get_ylim()[0], axisDendro.get_ylim()[1]], 'k--', linewidth = 1)

        clusters = []
        for subgroup in sorted([item for item in list(ClusteringObject[group].keys()) if not item=='linkage']):
            clusters.extend([subgroup for _ in range(ClusteringObject[group][subgroup]['data'].values.shape[0])])

        clusters = np.array(clusters)

        cluster_line_positions = np.where(clusters - np.roll(clusters,1) != 0)[0]

        for i in range(n_clusters - 1):
            posB = cluster_line_positions[i + 1] * (axisDendro.get_ylim()[1] / len(Z['leaves'])) - 5. * 0
            axisDendro.plot([posA, axisDendro.get_xlim()[1]], [posB, posB], '--', color='black', linewidth = 1.0)
        
        axisDendro.plot([posA, axisDendro.get_xlim()[1]], [0., 0.], '--', color='k', linewidth = 1.0)
        axisDendro.plot([posA, axisDendro.get_xlim()[1]], [-0. + axisDendro.get_ylim()[1], -0. + axisDendro.get_ylim()[1]], '--', color='k', linewidth = 1.0)

        axisDendro.text(axisDendro.get_xlim()[0], 0.5 * axisDendro.get_ylim()[1], 
                        'G%s:' % group + str(groupSize), fontsize=14*textScale).set_path_effects([path_effects.Stroke(linewidth=1, foreground=groupColors[group - 1]),path_effects.Normal()])

        return n_clusters, clusters, cluster_line_positions

    def addGroupHeatmapAndColorbar(data_loc, n_clusters, clusters, cluster_line_positions, bottom, top, group, groupColors, fig):

        axisMatrix = fig.add_axes([left + 0.205, bottom, dx + 0.025 + 0.075, top - bottom])

        masked_array = np.ma.array(data_loc, mask=np.isnan(data_loc))

        cmap = matplotlib.colors.LinearSegmentedColormap.from_list('GR', [(0, 1, 0), (1, 0, 0)], N=1000) #plt.cm.prism #plt.cm.hsv_r #plt.cm.RdYlGn_r
        cmap.set_bad('grey')
        im = axisMatrix.imshow(masked_array, aspect='auto', origin='lower', vmin=np.min(data_loc[np.isnan(data_loc) == False]), vmax=np.max(data_loc[np.isnan(data_loc) == False]), cmap=cmap)

        for i in range(n_clusters - 1):
            posB = cluster_line_positions[i + 1]
            posBm = cluster_line_positions[i]
            axisMatrix.plot([-0.5, data_loc.shape[1] - 0.5], [posB - 0.5, posB - 0.5], '--', color='black', linewidth = 1.0)

        def add_label(pos, labelText):
            return axisMatrix.text(-1., pos, labelText, ha='right', va='center', fontsize=12.*textScale).set_path_effects([path_effects.Stroke(linewidth=0.4, foreground=groupColors[group - 1]),path_effects.Normal()])

        order = clusters[np.sort(np.unique(clusters,return_index=True)[1])] - 1


        for i in range(n_clusters - 1):
            if len(data_loc[clusters == i + 1]) >= 5.:
                try:
                    add_label((cluster_line_positions[np.where(order == i)[0][0]] + cluster_line_positions[np.where(order == i)[0][0] + 1]) * 0.5, 'G%sS%s:%s' % (group,i + 1,len(data_loc[clusters == i + 1])))
                except:
                    print('Label printing error!')
        if len(data_loc[clusters == n_clusters]) >= 5.:
            posC = axisMatrix.get_ylim()[0] if n_clusters == 1 else cluster_line_positions[n_clusters - 1]
            add_label((posC + axisMatrix.get_ylim()[1]) * 0.5, 'G%sS%s:%s' % (group,n_clusters,len(data_loc[clusters == n_clusters])))

        axisMatrix.set_xticks([])
        axisMatrix.set_yticks([])

        times = ClusteringObject[group][subgroup]['data'].columns.values

        if group == 1:
            axisMatrix.set_xticks(range(data_loc.shape[1]))
            axisMatrix.set_xticklabels([('' if (i%2==1 and textScale>1.3) else np.int(time)) for i, time in enumerate(np.round(times,1))], rotation=0, fontsize=6*textScale)
            axisMatrix.set_xlabel('Time (hours)', fontsize=axisMatrix.xaxis.label._fontproperties._size*textScale)

        if group == sorted([item for item in list(ClusteringObject.keys()) if not item=='linkage'])[-1]:
            axisMatrix.set_title('Transformed gene expression', fontsize=axisMatrix.title._fontproperties._size*textScale)

        axisColor = fig.add_axes([0.635 - 0.075 - 0.1 + 0.075,current_bottom + 0.01,0.01, max(0.01,(current_top - current_bottom) - 0.02)])
        plt.colorbar(im, cax=axisColor, ticks=[np.max(im._A),np.min(im._A)])
        axisColor.tick_params(labelsize=6*textScale)
        axisColor.set_yticklabels([np.round(np.max(im._A),2),np.round(np.min(im._A),2)])

        return

    signalsInClusteringObject = 0
    for group in sorted([item for item in list(ClusteringObject.keys()) if not item=='linkage']):
        for subgroup in sorted([item for item in list(ClusteringObject[group].keys()) if not item=='linkage']):
            signalsInClusteringObject += ClusteringObject[group][subgroup]['data'].shape[0]


    fig = plt.figure(figsize=(12,8))

    left = 0.02
    bottom = 0.1
    current_top = bottom
    dx = 0.2
    dy = 0.8

    groupColors = ['b','g','r','c','m','y','k']
    [groupColors.extend(groupColors) for _ in range(10)]

    addAutocorrelationDendrogramAndHeatmap(ClusteringObject, groupColors, fig)

    for group in sorted([item for item in list(ClusteringObject.keys()) if not item=='linkage']):

        tempData = None
        for subgroup in sorted([item for item in list(ClusteringObject[group].keys()) if not item=='linkage']):
            subgroupData = ClusteringObject[group][subgroup]['data'].values
            tempData = subgroupData if tempData is None else np.vstack((tempData, subgroupData))

        current_bottom = current_top
        current_top += dy * float(len(tempData)) / float(signalsInClusteringObject)

        if len(tempData)==1:
            n_clusters, clusters, cluster_line_positions = 1, np.array([]), np.array([])
        else:
            n_clusters, clusters, cluster_line_positions = addGroupDendrogramAndShowSubgroups(ClusteringObject, len(tempData), current_bottom, current_top, group, groupColors, fig)

        addGroupHeatmapAndColorbar(tempData, n_clusters, clusters, cluster_line_positions, current_bottom, current_top, group, groupColors, fig)

    data_list = []
    data_names_list = []
    for group in sorted([item for item in list(ClusteringObject.keys()) if not item=='linkage']):
        for subgroup in sorted([item for item in list(ClusteringObject[group].keys()) if not item=='linkage']):
            if ClusteringObject[group][subgroup]['data'].shape[0] >= 5:
                data_list.append(ClusteringObject[group][subgroup]['data'].values)
                data_names_list.append('G%sS%s' % (group, subgroup))

    times = ClusteringObject[group][subgroup]['data'].columns.values

    for indVG, (dataVG, dataNameVG) in enumerate(zip(data_list, data_names_list)):
        
        x_min = 0.57
        x_max = 0.66
        y_min = 0.1
        y_max = 0.9

        numberOfVGs = len(data_list)
        height = min((y_max - y_min) / numberOfVGs, (x_max - x_min) * (12. / 8.))
        x_displacement = (x_max - x_min - height / 1.5) * 0.5
        y_displacement = (y_max - y_min - numberOfVGs * height) / numberOfVGs

        coords = [x_min + x_displacement, x_min + x_displacement + height / (12. / 8.), y_min + indVG * height + (0.5 + indVG) * y_displacement, y_min + (indVG + 1) * height + (0.5 + indVG) * y_displacement]

        addVisibilityGraph(dataVG, times, dataNameVG, coords, numberOfVGs, groupColors, fig)
    
    if vectorImage:
        fig.savefig(saveDir + dataName + '_DendrogramHeatmap.eps')
        fig.savefig(saveDir + dataName + '_DendrogramHeatmap.svg')
    else:
        fig.savefig(saveDir + dataName + '_DendrogramHeatmap.png', dpi=300)

    return None

def PlotVisibilityGraph(A, data, times, fileName, id):

    """Bar-plot style visibility graph.

    Parameters:
        A: 2d numpy.array
            Adjacency matrix

        data: 2d numpy.array
            Numpy array of floats

        times: 2d numpy.array
            Numpy array of floats

        fileName: str
            Name of the figure file to save

        id: str or int
            Label to add to the figure title

    Returns:
        None

    Usage:
        PlotVisibilityGraph(A, data, times, 'FIgure.png', 'Test Data')
    """

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

    return None

def PlotHorizontalVisibilityGraph(A, data, times, fileName, id):
    
    """Bar-plot style horizontal visibility graph.

    Parameters:
        A: 2d numpy.array
            Adjacency matrix

        data: 2d numpy.array
            Numpy array of floats

        times: 2d numpy.array
            Numpy array of floats

        fileName: str
            Name of the figure file to save

        id: str or int
            Label to add to the figure title

    Returns:
        None

    Usage:
        PlotHorizontalVisibilityGraph(A, data, times, 'FIgure.png', 'Test Data')
    """

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
    
    return None

def makePlotOfPeak(data_all, scores, selected_peak, selected_peak_value, plotID):

    fig, ax = plt.subplots()

    ax.plot(data_all.T[0], data_all.T[1], 'g', lw=3)
    ax.plot(scores.T[0], scores.T[1], 'ro', ms=5)
    ax.plot(selected_peak, selected_peak_value, 'bo', alpha=0.5, ms=10)

    fig.savefig('spline_%s.png' % ('' if plotID == None else str(plotID)), dpi=300)
    plt.close(fig)

    return
