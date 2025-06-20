## 1.4.0

- Updated codebase for compatibility with modern NumPy, pandas, and NetworkX (removed deprecated usages, updated aggregation and normalization logic).
- Fixed enrichment analysis input handling and report export: robust support for labeled, unlabeled, and mixed gene/compound input; ensured clean, correct output for "List of gene hits".
- Ensured all outputs and exports use plain Python types (not NumPy types).
- Fixed ValueErrors and shape issues from np.array on variable-length lists.
- Removed SyntaxErrors related to global variable usage and stray code/docstrings.
- Updated setup.py: refreshed dependency versions, removed python_requires restriction, and improved metadata.
- General code cleanup and suppression of unintended output in enrichment exports.

## 1.3.3

- updated functionality for visibility graph community detection for communityDetectByPathLength to output community times as an option

## 1.3.2

- updated code for ReactomeAnalysis, set header parameter headersGET = {'accept":'text/CSV'} for Reactome analysis service compatibility

## 1.3.1

- updated code for calculateTimeSeriesCategorization to output correct labels for Lag classifications when exporting to Excel files
- modified clueringFunctions exportClusteringObject to output components of clustering objects in the same order as they appear in heatmaps (i.e. top to bottom, instead of bottom to up, which was the previous behavior)

## 1.3.0

- updated internal code for annotate() function to use "text" versus "s"
- updated makeVisibilityGraph function
- updated addVisibilityGraph function with additional selection for community detection methods
- updated getCommunitiesOfTimeSeries to process horizontal and natural visibility graphs
- updated visualizeTimeSeriesCategorization for community detection options selection for displayed visibility graphs
- updated makeDendrogramHeatmapOfClusteringObject for community detection method options for displayed visibility graphs

## 1.2.9

- renamed inconsistencies:
  - PlotNVGBarGraphDual -> plotNVGBarGraphDual
  - PlotHVGBarGraphDual -> plotHVGBarGraphDual

## 1.2.8

- Documentation strings update for frequencySubjectMatch file - typos corrected.
- Functions changed to camelCase for consistency:
  - in module frequencySubjectMatch:
    - IOptimazeK -> optimizeK
    - get_community_genes_dict -> getCommunityGenesDict
    - split_genes -> splitGenes
    - get_community_top_genes_by_number -> getCommunityTopGenesByNumber
    - get_community_top_genes_by_frequency_ranking -> getCommunityTopGenesByFrequencyRanking
  - in module clusteringFunctions:
    - get_n_clusters_from_linkage_Silhouette -> getNClustersFromLinkageSilhouette
    - get_n_clusters_from_linkage_Elbow -> getNClustersFromLinkageElbow
  - in module visualizationFunctions
    - PlotNVGBarGraph_Dual -> PlotNVGBarGraphDual
    - PlotHVGBarGraphDual -> PlotHVGBarGraphDual
- example files were modified to reflect above changes
  - pyiomica_examples.ipynb
  - examplesVisibilityGraphCommunityDetection.py


- 1.2.7
   * Added new functionality in frequencySubjectMatch to enable comparison across subjects, utilizing spectra to identify common changing components, construct networks with such connections, and identify clusters of similar temporal behavior.

- 1.2.6

   * Updated autocorrelation computation function to improve handling of nan values.

- 1.2.5

   * Updated visibilityGraphCommunityDetection functions (correction for division by zero in community calculations).

- 1.2.4

   * Updated plotting heatmap functions (categorizationFunctions.visualizeTimeSeriesCategorization and visualizationFunctions.makeDendrogramHeatmapOfClusteringObject) to utilize optional custom strings for x-axis and plot labels.
   * Updated categorizationFunctions.visualizeTimeSeriesCategorization to avoid error for cases where the linkage array has only 1 row.


- 1.2.3

   * Applied minor fixes of numpy deprecation warnings.
   * Updated examples Jupyter notebook.
   * Updated enrichment report export function.
   * Updated PyIOmica dependency graph.

- 1.2.2

   * Applied minor fixes related to numpy upgrade.
   * Fixed typo in name of function extendedDataFrame.getLombScarglePeriodogramOfDataframe 

- 1.2.1

   * Added Reactome pathway overrepresentation analysis functions. 
   * Added Reactome analysis report export function. 
   * Added examples of Reactome analysis. 
   
- 1.2.0

   * Added new visibility graph based community detection functions. 
   * Added plotting functions. 
   * Added examples of visibility graph community detection. 
   * Updated examples Jupyter notebook. 


- 1.1.2

   * Small typographical fixes.


- 1.1.1

   * Updated examples Jupyter notebook. 
   * Small typographical fixes.


- 1.1.0

   * Restructured all modules. 
   * Developed ReadTheDocs documentation.


- 1.0.2

   * Updated setup dependencies for pip compatibility.


- 1.0.1 

   * Updated setup dependencies.


- 1.0.0

   * First Release.
