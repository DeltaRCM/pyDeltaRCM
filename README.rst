**************
pyDeltaRCM_WMT
**************


.. image:: https://api.travis-ci.com/DeltaRCM/pyDeltaRCM_WMT.svg?branch=develop
    :target: https://travis-ci.com/DeltaRCM/pyDeltaRCM_WMT

.. image:: https://coveralls.io/repos/github/DeltaRCM/pyDeltaRCM_WMT/badge.svg?branch=develop
    :target: https://coveralls.io/github/DeltaRCM/pyDeltaRCM_WMT?branch=develop


pyDeltaRCM is the Python version of the [Matlab deltaRCM](http://csdms.colorado.edu/wiki/Model:DeltaRCM) model by Man Liang. 
The pyDeltaRCM scripts in this repository can be run as a stand-alone model following the instructions below.


Documentation
#############

`Find the full documentation here <https://deltarcm.org/pyDeltaRCM_WMT/index.html>`_.



Installation
############

To install this package into an existing Python 3.x environment, download or clone the repository and run:

.. code::bash
    $ python setup.py install

Or for a developer installation run:

.. code::bash
    $ pip install -e .


Executing the model
###################

We recommend you check out our **`pyDeltaRCM in 10 minutes tutorial <https://deltarcm.org/pyDeltaRCM_WMT/guides/10min.html>`_**, which is part of our documentation.

Additionally, the model can be run with example script ``run_pyDeltaRCM.py``:

.. code::bash
    $ python run_pyDeltaRCM.py

This reads the input file ``tests/test.yaml`` and runs a 1 timestep simulation. 
This script will create an output folder in the working directory and save a PNG file of the parameter ``eta`` (surface elevation) every 50 timesteps.
