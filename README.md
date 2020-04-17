# pyDeltaRCM_WMT

[![Build Status](https://travis-ci.com/DeltaRCM/pyDeltaRCM_WMT.svg?branch=develop)](https://travis-ci.org/DeltaRCM/pyDeltaRCM_WMT)

[![Coverage Status](https://coveralls.io/repos/github/DeltaRCM/pyDeltaRCM_WMT/badge.svg?branch=develop)](https://coveralls.io/github/DeltaRCM/pyDeltaRCM_WMT?branch=develop)

pyDeltaRCM is the Python version of the [Matlab deltaRCM](http://csdms.colorado.edu/wiki/Model:DeltaRCM) model by Man Liang. 
This model has been turned into a component for WMT, the [Community Surface Dynamics Modeling System (CSDMS)](http://csdms.colorado.edu/wiki/Main_Page) Web Modeling Interface.

The pyDeltaRCM scripts in this repository can be run as a stand-alone model following the instructions below.

## Installation

To install this package into an existing Python 2.x environment, download or clone the repository and run:

``` $ python setup.py install```

## Execution

The example script `run_pyDeltaRCM.py` with:

``` $ python run_pyDeltaRCM.py```

It reads the input file `deltaRCM.yaml` and runs a 2000 timestep simulation. This script will create an output folder in the working directory and save a PNG file of the parameter `eta` (surface elevation) every 50 timesteps.

Change the parameters in the YAML file to customize the simulation.
