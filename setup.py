#! /usr/bin/env python
from ez_setup import use_setuptools
use_setuptools()
from setuptools import setup, find_packages


setup(name='pyDeltaRCM',
      version='0.1.0',
      author='Mariela Perignon',
      author_email='perignon@colorado.edu',
      license='MIT',
      description="Python version of Man Liang's DeltaRCM, in Matlab",
      long_description=open('README.md').read(),
      packages=find_packages(exclude=['*.tests']),
      url='https://github.com/mperignon/pyDeltaRCM_WMT',
)
