**************
pyDeltaRCM
**************

| .. image:: https://github.com/DeltaRCM/pyDeltaRCM/workflows/build/badge.svg
    :target: https://github.com/DeltaRCM/pyDeltaRCM/actions
.. image:: https://badge.fury.io/py/pyDeltaRCM.svg
    :target: https://badge.fury.io/py/pyDeltaRCM


|.. image:: https://codecov.io/gh/DeltaRCM/pyDeltaRCM/branch/develop/graph/badge.svg
    :target: https://codecov.io/gh/DeltaRCM/pyDeltaRCM
.. image:: https://app.codacy.com/project/badge/Grade/1c137d0227914741a9ba09f0b00a49a7
    :target: https://www.codacy.com/gh/DeltaRCM/pyDeltaRCM?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=DeltaRCM/pyDeltaRCM&amp;utm_campaign=Badge_Grade

pyDeltaRCM is the Python version of the `Matlab deltaRCM <https://csdms.colorado.edu/wiki/Model:DeltaRCM>`_ model by Man Liang. 
The pyDeltaRCM scripts in this repository can be run as a stand-alone model following the instructions below.

This model repository no longer includes support for the `Basic Model Interface (BMI) <https://bmi.readthedocs.io/en/latest/?badge=latest>`_.
We have separated BMI support for pyDeltaRCM to another repository (`the BMI_pyDeltaRCM model <https://deltarcm.org/BMI_pyDeltaRCM/>`_).


Documentation
#############

`Find the full documentation here <https://deltarcm.org/pyDeltaRCM/index.html>`_.



Installation
############

pyDeltaRCM can be installed from the Pypi package respository.
To install this package into an existing Python 3.x environment:

.. code:: bash

    $ pip install pyDeltaRCM

For the latest version of pyDeltaRCM, download or clone the repository and run:

.. code:: bash

    $ python setup.py install

Or for a developer installation run:

.. code:: bash

    $ pip install -e .


Executing the model
###################

We recommend you check out our `pyDeltaRCM in 10 minutes tutorial <https://deltarcm.org/pyDeltaRCM/guides/10min.html>`_, which is part of our documentation.

Additionally, the model can be run with example script ``run_pyDeltaRCM.py``:

.. code::bash
    $ python run_pyDeltaRCM.py

This reads the input file ``tests/test.yaml`` and runs a 1 timestep simulation. 
This script will create an output folder in the working directory and save a PNG file of the parameter ``eta`` (surface elevation) every 50 timesteps.
