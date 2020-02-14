import numpy as np
import networkx as nx

import pyiomica as pio
from pyiomica import visualizationFunctions
from pyiomica import visibilityGraphCommunityDetection
import matplotlib.pyplot as plt

### create time series
np.random.seed(11)
times = np.arange( 0, 2*np.pi, 0.35)
tp = list(range(len(times)))
data = 5*np.cos(times) + 2*np.random.random(len(times))

### plot time series
fig, ax = plt.subplots(figsize=(8,3))
ax.plot(tp,data)
ax.set_title('Time Series', fontdict={'color': 'k'},fontsize=20)
ax.set_xlabel('Times', fontsize=20)
ax.set_ylabel('Signal intensity', fontsize=20)
ax.set_xticks(tp)
ax.set_xticklabels([str(item) for item in np.round(tp,2)],fontsize=20, rotation=0)
ax.set_yticks([])

fig.tight_layout()
filename ='./A.eps'
fig.savefig(filename, dpi=600)
plt.close(fig)

### plot weighted natural visibility graph, weight is  Euclidean distance
g_nx_NVG, A_NVG = visibilityGraphCommunityDetection.createVisibilityGraph(data,tp,"natural", weight = 'distance')
filename = './B.eps'
visualizationFunctions.PlotNVGBarGraph_Dual(A_NVG, data, tp,fileName = filename,
                                            title = 'Natural Visibility Graph',fontsize=20,figsize=(8,3))

### plot reverse perspective weighted natural visibility graph, weight is  Euclidean distance
g_nx_revNVG, A_revNVG = visibilityGraphCommunityDetection.createVisibilityGraph(-data,tp,"natural", weight = 'distance')
filename = './C.eps'
visualizationFunctions.PlotNVGBarGraph_Dual(A_revNVG, -data, tp,fileName = filename,
                                            title='Reverse perspective Natural Visibility Graph',fontsize=20,figsize=(8,3))

### plot dual perspective natural visibility graph, weight is Euclidean distance
g_nx_dualNVG, A_dualNVG = visibilityGraphCommunityDetection.createVisibilityGraph(data,tp,"dual_natural", 
                                                                                  weight = 'distance', withsign=True)
filename = './D.eps'
visualizationFunctions.PlotNVGBarGraph_Dual(A_dualNVG, data, tp,fileName=filename,
                                            title='Dual perspective Natural Visibility Graph',fontsize=20,figsize=(10,4))

### plot line layout dual perspective natural visibility graph with community structure, weight is Euclidean distance
communities = visibilityGraphCommunityDetection.communityDetectByPathLength(g_nx_dualNVG, direction = None, cutoff='auto')
com = (communities, g_nx_dualNVG)
visualizationFunctions.makeVisibilityGraph(data, tp, './', 'E', layout='line',communities=com, 
                       level=0.8,figsize = (10,6), extension='.eps')
