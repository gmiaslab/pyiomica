![logo](https://raw.githubusercontent.com/gmiaslab/pyiomica/master/pyiomica/data/PyIOmica.png)

[![release](https://img.shields.io/github/v/release/gmiaslab/pyiomica?logo=github)](https://github.com/gmiaslab/pyiomica)
[![pypi version](https://img.shields.io/pypi/v/pyiomica?logo=pypi)](https://pypi.org/project/pyiomica)
[![readthedocs](https://readthedocs.org/projects/pyiomica/badge/?version=latest&style=flat)](https://pyiomica.readthedocs.io)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5733381.svg)](https://doi.org/10.5281/zenodo.5733381)
[![pypi license](https://img.shields.io/pypi/l/pyiomica)](https://pypi.org/project/pyiomica)

# PyIOmica (pyiomica)
This repository contains PyIOmica, a Python package that provides bioinformatics utilities for analyzing (dynamic) omics datasets. PyIOmica extends MathIOmica usage to Python and implements new visualizations and computational tools for graph analyses. The documentation is available at Read the Docs: https://pyiomica.readthedocs.io/en/latest/

# PyIOmica Installation Instructions

## A. INSTALLATION 
  
### Pre-Installation Requirements

     To install PyIOmica on any platform you need Python. Required package dependencies are listed in the setup.py file. The software has been tested with Python 3.13.5. Compatibility with earlier Python 3.x versions depends on the minimum requirements of the dependencies listed in setup.py.
  
### Installation Instructions

1. To install the current release from PyPI (Python Package Index) use pip:

```bash
pip install pyiomica
```

Alternatively, you can install directly from github using:
```bash
pip install git+https://github.com/gmiaslab/pyiomica/
```

or

```bash
git clone https://github.com/gmiaslab/pyiomica/
python setup.py install
```


## B. RUNNING PyIOmica

After installation you can run:

```python
>>> import pyiomica
```

## C. DOCUMENTATION

Documentation for PyIOmica is built-in and is available through the help() functionality in Python.
Also the documentation is available at Read the Docs: https://pyiomica.readthedocs.io/en/latest/

## D. ADDITIONAL INFORMATION

* PyIOmica is a multi-omics analysis framework distributed as a Python package that aims to assist in bioinformatics.
* The most current version of the package is maintained at
<https://github.com/gmiaslab/pyiomica>
* News are distributed via twitter (@mathiomica)

## E. LICENSING

PyIOmica is released under an MIT License. Please also consult the folder LICENSES distributed with PyIOmica regarding Licensing information for use of external associated content.

## F. OTHER CONTACT INFORMATION

* G. Mias Lab (https://georgemias.org)
* e-mail: mathiomica@gmail.com
* twitter: @mathiomica

## G. FUNDING

PyIOmica development and associated research were supported by the Translational Research Institute 
for Space Health through NASA Cooperative Agreement NNX16AO69A (Project Number T0412, PI: Mias). 
The content is solely the responsibility of the authors and does not necessarily 
represent the official views of the supporting funding agencies.

## I. CITATIONS
- If you use PyIOmica in your work please use the following citation:

   - Sergii Domanskyi, Carlo Piermarocchi and George I Mias, *PyIOmica: longitudinal omics analysis and trend identification*. Bioinformatics, 36(7), 2306–2307 (2020). https://doi.org/10.1093/bioinformatics/btz896

- If you use PyIOmica's visibility graph functionality, please also consider the following citation:

   - Minzhang Zheng, Sergii Domanskyi, Carlo Piermarocchi, and George I Mias, *Visibility graph based temporal community detection with applications in biological time series*, Sci Rep 11, 5623 (2021). https://doi.org/10.1038/s41598-021-84838-x

