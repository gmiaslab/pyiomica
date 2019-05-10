"""
setup.py for pyiomica package
"""
from setuptools import setup, find_packages
from codecs import open
from os import path
here = path.abspath(path.dirname(__file__))
# Get the long description from the README file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description=f.read()

setup(
    name='pyiomica',
    packages=find_packages(),
    version='1.0.0',
    description='Omics Analysis Tool Suite',
    long_description=long_description,
    include_package_data=True,
    author='S. Domanskyi, C. Piermarocchi, G. Mias',
    author_email='gmiaslab@gmail.com',
    license='MIT',
    url='https://github.com/gmiaslab/pyiomica',
    download_url='https://github.com/gmiaslab/pyiomica/archive/1.0.0.tar.gz',
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
        'pandas>=0.23.3',
        'numpy>=1.15.3',
        'scipy>=1.1.0'],
    zip_safe=False
)
