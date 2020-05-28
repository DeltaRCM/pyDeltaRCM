**********
User Guide
**********

Guide to users!


==============================
Configuring an input YAML file
==============================

The configuration for a pyDeltaRCM run is set up by a parameter set, described in the ``YAML`` markup format.
To configure a run, you should create a file called, for example, ``run_parameters.yml``. 
Inside this file you can specify parameters for your run, with each parameter on a new line. For example, if ``run_parameters.yml`` contained the line: 

.. code-block:: yaml

    S0: 0.005

then a :obj:`~pyDeltaRCM.deltaRCM_driver.pyDeltaRCM` model instance initialized with this file specified as ``input_file`` will have a slope of 0.005.
Multiple parameters can be specified line by line.

Default values are substituted for any parameter not explicitly given in the ``input_file`` ``.yml`` file.
Default values of the YAML configuration are listed in the :doc:`../reference/deltaRCM_driver/yaml_defaults`.


===================
Starting model runs
===================

There are three ways to interact with the pyDeltaRCM model. 
Two methods are part of the "high-level" model API, and these approaches each consist of a single line of code.
The third method is to create a model instance via the "low-level" model API and manually handle timestepping and model finalization.


High-level model API
====================

The high-level API is accessed via either a shell prompt or python script, and is invoked directly if a YAML configuration file includes the ``timesteps`` variable.

.. note::
    To use the high-level API, the configuration YAML file must contain the ``timesteps`` variable.

Bash API
--------

For example, to invoke a model run from the bash prompt using the YAML file ``model_configuration.yml`` which looks like:

.. code-block:: yaml

    timesteps: 500

we would simply invoke at the bash prompt:

.. code:: bash
    
    python -m pyDeltaRCM --config model_configuration.yml

or equivalently:

.. code:: bash
    
    run_pyDeltaRCM --config model_configuration.yml

These invokations will run the pyDeltaRCM with the parameters specified in the ``model_configuration.yml`` file, and automatically :obj:`~pyDeltaRCM.deltaRCM_driver.pyDeltaRCM.update` the model 500 times.


Python API
----------

Alternatively, calling the ``run_model`` method from a python script, with the :meth:`input file <pyDeltaRCM.deltaRCM_driver.pyDeltaRCM.__init__>` parameter specified as the same ``model_configuration.yml`` file above, would run the pyDeltaRCM model, and automatically :obj:`~pyDeltaRCM.deltaRCM_driver.pyDeltaRCM.update` the model 500 times.



Low-level model API
===================

iinteract with the model by creating your own script, and manipulating model outputs at the desired level. The simplest case is to do

.. code::

    delta = pyDeltaRCM(input_file='model_configuration.yml')

    for time in range(0,1):
        delta.update()

    delta.finalize()

However, you can also inspect/modify the :obj:`~pyDeltaRCM.deltaRCM_driver.pyDeltaRCM.update` method, and change the order or add operations as desired.


=============================
Advanced model configurations
=============================

Configuring multiple model runs from a single YAML file
==============================================================

todo
