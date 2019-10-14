Enrichment analyses functions
=============================

Submodule **pyiomica.enrichmentAnalyses**


Example of use of a function described in this module:

.. code-block:: python
   :emphasize-lines: 1-

   
   import pyiomica as pio

   # Import functions necessary for this demo
   from pyiomica.enrichmentAnalyses import GOAnalysis, ExportEnrichmentReport

   # Specify a directory for output
   EnrichmentOutputDirectory = pio.os.path.join('results','EnrichmentOutputDirectory')

   # Let's do a GO analysis for a group of genes, annotated with their "Gene Symbol":
   goExample1 = GOAnalysis(["TAB1", "TNFSF13B", "MALT1", "TIRAP", "CHUK", 
                            "TNFRSF13C", "PARP1", "CSNK2A1", "CSNK2A2", "CSNK2B", "LTBR", 
                            "LYN", "MYD88", "GADD45B", "ATM", "NFKB1", "NFKB2", "NFKBIA", 
                            "IRAK4", "PIAS4", "PLAU"])

   # Export enrichment results in to .xlsx file
   ExportEnrichmentReport(goExample1, 
                          AppendString='goExample1', 
                          OutputDirectory=EnrichmentOutputDirectory + 'GOAnalysis/')

.. Note::
    
    Function ``ExportEnrichmentReport`` generates ".xlsx" file, described in 
    :ref:`examples-reference-label`.


.. automodule:: pyiomica.enrichmentAnalyses
    :members:
    :undoc-members:
    :show-inheritance:




