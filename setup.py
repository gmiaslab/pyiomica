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
    version='1.3.3',
    description='Omics Analysis Tool Suite',
    long_description_content_type="text/markdown",
    long_description=long_description,
    include_package_data=True,
    author='S. Domanskyi, C. Piermarocchi, G. Mias',
    author_email='gmiaslab@gmail.com',
    license='MIT',
    url='https://github.com/gmiaslab/pyiomica',
    download_url='https://github.com/gmiaslab/pyiomica/archive/1.3.3.tar.gz',
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
        'h5py>=2.9.0',
        'matplotlib>=3.1.0',
        'networkx>=2.3',
        'numba>=0.44.1',
        'numpy>=1.16.4',
        'openpyxl>=2.6.2',
        'pandas>=0.24.2',
        'pymysql>=0.9.3',
        'tables>=3.5.2',
        'requests>=2.22.0',
        'scikit-learn>=0.21.2',
        'scipy>=1.2.1',
        'xlsxwriter>=1.1.8',
        'appdirs>=1.4.3'],
    zip_safe=False
)
