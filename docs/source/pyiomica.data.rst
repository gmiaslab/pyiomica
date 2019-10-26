Included data
=============

PyIOmica is supplied with a set of examples that require folder `data 
<https://github.com/gmiaslab/pyiomica/tree/master/pyiomica/data>`_
described here. This folder is created during installation of PyIOmica, see section Installation,
and its location is stored in PyIOmica's global variable ``ConstantPyIOmicaDataDirectory``.

Annotation and Enumeration functions such as Enrichment Analyses functions 
require various dictionaties and information files to be downloaded from the Internet.
We provide a set of such dictionaries and files (in PyIOmica's data directory described above) to make it possible for user run PyIOmica 
in the abscence of the Internet connection. However the user can always override these 
files with updates ones by specifying proper flags to the functions that use the dictionaries.
