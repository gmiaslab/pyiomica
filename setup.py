"""
setup.py for pyiomica package
"""
from setuptools import setup, find_packages
from codecs import open
from os import path
here = path.abspath(path.dirname(__file__))
# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description=f.read()

setup(
    name='pyiomica',
    packages=find_packages(),
    version='1.4.0',
    description='Omics Analysis Tool Suite',
    long_description_content_type="text/markdown",
    long_description=long_description,
    include_package_data=True,
    author='S. Domanskyi, C. Piermarocchi, G. Mias',
    author_email='gmiaslab@gmail.com',
    license='MIT',
    url='https://github.com/gmiaslab/pyiomica',
    download_url='https://github.com/gmiaslab/pyiomica/archive/1.4.0.tar.gz',
    keywords=['omics', 'longitudinal','bioinformatics'],
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',
        'Intended Audience :: Developers',
        'Intended Audience :: End Users/Desktop',
        'Operating System :: MacOS',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: Unix',
        'Topic :: Education',
        'Topic :: Utilities',
        ],
    install_requires=[
        'appdirs>=1.4.4',
        'h5py>=3.14.0',
        'matplotlib>=3.10.3',
        'networkx>=3.5',
        'numba>=0.61.2',
        'numpy>=2.2.6',
        'openpyxl>=3.1.5',
        'pandas>=2.3.0',
        'pymysql>=1.1.1',
        'requests>=2.32.4',
        'scikit-learn>=1.7.0',
        'scipy>=1.15.3',
        'tables>=3.10.2',
        'xlsxwriter>=3.2.5'],
    zip_safe=False
)
