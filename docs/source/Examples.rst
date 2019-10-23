.. _examples-reference-label:

Examples
========

Examples different uses of PyIOmica package API.


Enrichment report
-----------------


Function ``ExportEnrichmentReport`` generates enrichment report in form of ".xlsx" file.


.. thumbnail:: https://raw.githubusercontent.com/gmiaslab/pyiomica/master/docs/examples/goExample1.PNG
    :title: GO analysis enrichment report
    :alt: Cannot load this photo
    :align: center
    :download: false

The are following columns in each Worksheet of the output ".xlsx" file

* ID: Annotation term identifier
* p-Value: Probability to find at least "Counts in members" number of genes when drawing "Counts in list" genes "Counts in family" times (without replacement) from "Counts in members" identifiers. Note: this is the behavior when using default setting, i.e. Hypergeometric distribution testing function
* BH-corrected p-Value:  p-Value corrected for false discovery rate (FDR) via Benjamini-Hochberg procedure
* Significant: Whether the "BH-corrected p-Value" is below the threshold (typically 0.05) specified for the enrichment analysis
* Counts in list: Number of genes in the input list
* Counts in family: Number of genes in a particular annotation term
* Total members: Number of members (e.g. UniprotIDs) in the IDs to annotation terms dictionary
* Counts in members: Number of annotation terms in which at least of gene from the input list appear
* Description: Description of a particular annotation term, e.g. details/type
* List of gene hits: Gene identifiers that are found in a particular annotation term. The identifiers are separated by a Vertical bar



Import of MathIOmica Objects
----------------------------

.. literalinclude:: ../examples/examplesImportMathiomicaObjects.py




Clustering object export example
--------------------------------

Example of a clustering object exported to ".xlsx" file.

.. thumbnail:: https://raw.githubusercontent.com/gmiaslab/pyiomica/master/docs/examples/ExampleClusteringObject_screen.png
    :title: Clustering Object export
    :alt: Cannot load this photo
    :align: center
    :download: false

The file contains one Spreadsheet for each subgroup in a clustering object. The spreadsheet is structrured as follows. The first column indicates the data source, the second column has gene identifiers, all the remaining colummns contain data and autocorrelations. Note, this is the structure when using the default settings. User can modify output by setting various options.



GO Analysis examples
--------------------

.. literalinclude:: ../examples/examplesGOAnalysis.py






KEGG Analysis examples
----------------------

.. literalinclude:: ../examples/examplesKEGGAnalysis.py





Visibility Graph examples
-------------------------

.. literalinclude:: ../examples/examplesVisibilityGraph.py


Normal (left) and Horizontal (right) visibility graph on a circular layout:

.. thumbnail:: https://raw.githubusercontent.com/gmiaslab/pyiomica/master/docs/examples/normal_circle_VG.png
    :title: Visibility graph
    :alt: Cannot load this photo
    :align: left
    :width: 300
    :height: 300
    :download: false
    :group: groupCirlceVG

.. thumbnail:: https://raw.githubusercontent.com/gmiaslab/pyiomica/master/docs/examples/horizontal_circle_VG.png
    :title: Visibility graph
    :alt: Cannot load this photo
    :align: right
    :width: 300
    :height: 300
    :download: false
    :group: groupCirlceVG

.. container:: clearfix

   .. stuff

Normal (left) and Horizontal (right) visibility graph on a linear layout:

.. thumbnail:: https://raw.githubusercontent.com/gmiaslab/pyiomica/master/docs/examples/normal_line_VG.png
    :title: Visibility graph
    :alt: Cannot load this photo
    :align: left
    :width: 300
    :height: 300
    :download: false
    :group: groupLineVG

.. thumbnail:: https://raw.githubusercontent.com/gmiaslab/pyiomica/master/docs/examples/horizontal_line_VG.png
    :title: Visibility graph
    :alt: Cannot load this photo
    :align: right
    :width: 300
    :height: 300
    :download: false
    :group: groupLineVG


.. container:: clearfix

   .. stuff


Normal (left) and Horizontal (right) bar-style visibility graph:

.. thumbnail:: https://raw.githubusercontent.com/gmiaslab/pyiomica/master/docs/examples/normal_barVG.png
    :title: Visibility graph
    :alt: Cannot load this photo
    :align: left
    :width: 300
    :height: 300
    :download: false
    :group: groupBarVG

.. thumbnail:: https://raw.githubusercontent.com/gmiaslab/pyiomica/master/docs/examples/horizontal_barVG.png
    :title: Visibility graph
    :alt: Cannot load this photo
    :align: right
    :width: 300
    :height: 300
    :download: false
    :group: groupBarVG


.. container:: clearfix

   .. stuff




Extended DataFrame
------------------

Usage of some of the functions added to a standard DataFrame.

.. literalinclude:: ../examples/examplesExtendedDataFrame.py



Time Series Categorization
--------------------------

.. literalinclude:: ../examples/examplesCategorization.py

.. thumbnail:: https://raw.githubusercontent.com/gmiaslab/pyiomica/master/docs/examples/results/results%20SLV%20Delta/SLV_Delta_LAG1_AutocorrelationsBased_DendrogramHeatmap.png
    :title: Categorization example
    :alt: Cannot load this photo
    :align: center
    :width: 600px
    :download: false
    :group: heatMapDendroACbased

.. thumbnail:: https://raw.githubusercontent.com/gmiaslab/pyiomica/master/docs/examples/results/results%20SLV%20Delta/SLV_Delta_LAG2_AutocorrelationsBased_DendrogramHeatmap.png
    :title: Categorization example
    :alt: Cannot load this photo
    :align: center
    :width: 600px
    :download: false
    :group: heatMapDendroACbased

.. thumbnail:: https://raw.githubusercontent.com/gmiaslab/pyiomica/master/docs/examples/results/results%20SLV%20Delta/SLV_Delta_LAG3_AutocorrelationsBased_DendrogramHeatmap.png
    :title: Categorization example
    :alt: Cannot load this photo
    :align: center
    :width: 600px
    :download: false
    :group: heatMapDendroACbased

.. thumbnail:: https://raw.githubusercontent.com/gmiaslab/pyiomica/master/docs/examples/results/results%20SLV%20Delta/SLV_Delta_SpikeMax_AutocorrelationsBased_DendrogramHeatmap.png
    :title: Categorization example
    :alt: Cannot load this photo
    :align: center
    :width: 600px
    :download: false
    :group: heatMapDendroACbased

.. thumbnail:: https://raw.githubusercontent.com/gmiaslab/pyiomica/master/docs/examples/results/results%20SLV%20Delta/SLV_Delta_SpikeMin_AutocorrelationsBased_DendrogramHeatmap.png
    :title: Categorization example
    :alt: Cannot load this photo
    :align: center
    :width: 600px
    :download: false
    :group: heatMapDendroACbased




