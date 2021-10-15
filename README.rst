**************
pyDeltaRCM
**************

.. image:: https://badge.fury.io/py/pyDeltaRCM.svg
    :target: https://badge.fury.io/py/pyDeltaRCM

.. image:: https://joss.theoj.org/papers/10.21105/joss.03398/status.svg
   :target: https://doi.org/10.21105/joss.03398

.. image:: https://github.com/DeltaRCM/pyDeltaRCM/actions/workflows/build.yml/badge.svg
    :target: https://github.com/DeltaRCM/pyDeltaRCM/actions
    
.. image:: https://codecov.io/gh/DeltaRCM/pyDeltaRCM/branch/develop/graph/badge.svg
  :target: https://codecov.io/gh/DeltaRCM/pyDeltaRCM

.. image:: https://app.codacy.com/project/badge/Grade/1c137d0227914741a9ba09f0b00a49a7
    :target: https://www.codacy.com/gh/DeltaRCM/pyDeltaRCM?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=DeltaRCM/pyDeltaRCM&amp;utm_campaign=Badge_Grade    
    



*pyDeltaRCM* is a computationally efficient, free and open source, and easy-to-customize numerical delta model based on the original DeltaRCM model design (`Matlab deltaRCM <https://csdms.colorado.edu/wiki/Model:DeltaRCM>`_ model by Man Liang; `Liang et al., 2015 <https://doi.org/10.5194/esurf-3-67-2015>`_).
*pyDeltaRCM* delivers improved model stability and capabilities, infrastructure to support exploration with minimal boilerplate code, and establishes an approach to extending model capabilities that ensures reproducible and comparable studies.


.. figure:: https://deltarcm.org/pyDeltaRCM/_images/cover.png
    
    Weighted random walks for 20 water parcels, in a *pyDeltaRCM* model run with default parameters.


Documentation
#############

`Find the complete documentation here <https://deltarcm.org/pyDeltaRCM/index.html>`_.

Documentation includes an `installation guide <https://deltarcm.org/pyDeltaRCM/meta/installing.html>`_, a thorough `guide for users <https://deltarcm.org/pyDeltaRCM/guides/user_guide.html>`_, detailed `API documentation for developers <https://deltarcm.org/pyDeltaRCM/reference/index.html>`_, a `plethora of examples <https://deltarcm.org/pyDeltaRCM/examples/index.html>`_ to use and develop pyDeltaRCM in novel scientific experiments, and more!


Installation
############

See our complete `installation guide <https://deltarcm.org/pyDeltaRCM/meta/installing.html>`_, especially if you are a developer planning to modify or contribute code (`developer installation guide <https://deltarcm.org/pyDeltaRCM/meta/installing.html#developer-installation>`_), or if you are new to managing Python `venv` or `conda` environments.

For a quick installation into an existing Python 3.x environment:

.. code:: console

    $ pip install pyDeltaRCM


Executing the model
###################

We recommend you check out our `pyDeltaRCM in 10 minutes tutorial <https://deltarcm.org/pyDeltaRCM/guides/10min.html>`_, which is part of our documentation.

Beyond that breif tutorial, we have a comprehensive `User Documentation <https://deltarcm.org/pyDeltaRCM/index.html#user-documentation>`_ and `Developer Documentation <https://deltarcm.org/pyDeltaRCM/index.html#developer-documentation>`_ to check out.


Additional notes
################

This repository no longer includes the `Basic Model Interface (BMI) <https://bmi.readthedocs.io/en/latest/?badge=latest>`_ wrapper to the DeltaRCM model.
*pyDeltaRCM* maintains BMI compatibility through another repository (`the BMI_pyDeltaRCM model <https://deltarcm.org/BMI_pyDeltaRCM/>`_).
