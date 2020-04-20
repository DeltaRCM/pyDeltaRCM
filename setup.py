#! /usr/bin/env python
from setuptools import setup, find_packages


setup(name='pyDeltaRCM',
      version='0.1.0',
      author='The DeltaRCM Team',
      license='MIT',
      description="Python version of original Matlab DeltaRCM",
      long_description=open('README.md').read(),
      packages=find_packages(exclude=['*.tests']),
      url='https://github.com/DeltaRCM/pyDeltaRCM',
      install_requires=['matplotlib', 'netCDF4',
                        'basic-modeling-interface', 'scipy', 'numpy'],
      )
