#! /usr/bin/env python
from setuptools import setup, find_packages

from pyDeltaRCM import utils

setup(name='pyDeltaRCM',
      version=utils._get_version(),
      author='The DeltaRCM Team',
      license='MIT',
      description="Python version of original Matlab DeltaRCM",
      long_description=open('README.rst').read(),
      packages=find_packages(exclude=['*.tests']),
      include_package_data=True,
      url='https://github.com/DeltaRCM/pyDeltaRCM',
      install_requires=['matplotlib', 'netCDF4',
                        'basic-modeling-interface', 'scipy', 'numpy',
                        'pyyaml', 'numba'],
      )
