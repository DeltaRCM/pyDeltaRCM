**********
User Guide
**********

.. rubric:: Preface

All of the documentation in this page, and elsewhere assumes you have imported the `pyDeltaRCM` package as ``pyDeltaRCM``:

.. code:: python

    >>> import pyDeltaRCM

Additionally, the documentation frequently refers to the `numpy` and `matplotlib` packages.
We may not always explicitly import these packages throughout the documentation; we may refer to these packages by their common shorthand as well.

.. code:: python

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt


=================
Running the model
=================

Running a `pyDeltaRCM` model is usually accomplished via a combination of configuration files and scripts.
In the simplest case, actually, none of these are required; the following command (executed at a console) runs a `pyDeltaRCM` model with the default set of model parameters for 10 timesteps:

.. code:: console

    $ pyDeltaRCM --timesteps 10

Internally, this console command calls a Python `function` that runs the model via a `method`.
Python code and scripts are the preferred and most common model use case.
Let's do the exact same thing (run a model for 10 timesteps), but do so with Python code.

Python is an object-oriented programming language, which means that the `pyDeltaRCM` model is set up to be manipulated *as an object*.
In technical jargon, the `pyDeltaRCM` model is a `class`, specifically, it is a class called `DeltaModel`; stated explicitly, **the actual model is created by instantiating** (making an instance of a class) **the `DeltaModel`.**
We can instantiate a Python object by calling from a python script (or at the Python console):

.. code:: python

    >>> mdl = pyDeltaRCM.DeltaModel()

.. plot::
    :context: reset

    >>> with pyDeltaRCM.shared_tools._docs_temp_directory() as output_dir:
    ...     mdl = pyDeltaRCM.DeltaModel(out_dir=output_dir)

Here, `mdl` is an instance of the `DeltaModel` class; `mdl` is *an actual model that we can run*.
Based on the :doc:`default parameters </reference/model/yaml_defaults>`, the model domain is configured as an open basin, with an inlet centered along one wall.
Default parameters are set to sensible values, and generally follow those described in [1]_.

.. plot::
    :context:
    
    >>> fig, ax = plt.subplots(figsize=(4, 3))
    >>> ax.imshow(mdl.bed_elevation, vmax=-3)
    >>> plt.show()

To run the model, we use the :obj:`update` method of the `DeltaModel` instance (i.e., call `update()` on `mdl`):

.. code:: python

    >>> for _ in range(0, 5):
    ...     mdl.update()

Finally, we need to tie up some loose ends when we are done running the model.
We use the :obj:`finalize` method of the `DeltaModel` instance (i.e., call `finalize()` on `mdl`):

.. code:: python

    >>> mdl.finalize()

And putting the above t all together as a complete minimum working example script:

.. code-block:: python

    >>> mdl = DeltaModel()

    >>> for _ in range(0, 5):
    ...    mdl.update()

    >>> mdl.finalize()

Now, we probably don't want to just run the model with default parameters, so the remainder of the guide will cover other use cases; first up is changing model parameter values via a ``YAML`` configuration file.



==============================
Configuring an input YAML file
==============================

The configuration for a pyDeltaRCM run is set up by a parameter set, usually described in the ``YAML`` markup format.
To configure a run, you should create a file called, for example, ``run_parameters.yml``.
Inside this file you can specify parameters for your run, with each parameter on a new line. For example, if ``run_parameters.yml`` contained the line:

.. code-block:: yaml

    S0: 0.005
    seed: 42

then a :obj:`~pyDeltaRCM.DeltaModel` model instance initialized with this file specified as ``input_file`` will have a slope of 0.005, and will use a random seed of 42.
Multiple parameters can be specified line by line.

Default values are substituted for any parameter not explicitly given in the ``input_file`` ``.yml`` file.
Default values of the YAML configuration are listed in the :doc:`/reference/model/yaml_defaults`.


===================
Starting model runs
===================

There are two API levels at which you can interact with the pyDeltaRCM model.
There is a "high-level" model API, which takes as argument a YAML configuration file, and will compose a list of jobs as indicated in the YAML file; the setup can be configured to automatically execute the job list, as well.
The "low-level" API consists of creating a model instance from a YAML configuration file and manually handling the timestepping, or optionally, augmenting operations of the model to implement new features.



.. _high_level_api:

High-level model API
====================

The high-level API is accessed via either a shell prompt or python script, and handles setting up the model configuration and running the model a specified duration.

For the following high-level API demonstrations, consider a YAML input file named ``model_configuration.yml`` which looks like:

.. code-block:: yaml

    Length: 5000
    Width: 2000
    timesteps: 500


Command line API
----------------

To invoke a model run from the command line using the YAML file ``model_configuration.yml`` defined above,
we would simply call:

.. code:: bash

    pyDeltaRCM --config model_configuration.yml

or equivalently:

.. code:: bash

    python -m pyDeltaRCM --config model_configuration.yml

These invocations will run the pyDeltaRCM :obj:`preprocessor <pyDeltaRCM.preprocessor.PreprocessorCLI>` with the parameters specified in the ``model_configuration.yml`` file.
If the YAML configuration indicates multiple jobs (:ref:`via matrix expansion or ensemble specification <configuring_multiple_jobs>`), the jobs will each be run automatically by calling :obj:`~pyDeltaRCM.DeltaModel.update` on the model 500 times.



Python API
----------

The Python high-level API is accessed via the :obj:`~pyDeltaRCM.Preprocessor` object.
First, the `Preprocessor` is instantiated with a YAML configuration file (e.g., ``model_configuration.yml``):

.. code:: python

    >>> pp = preprocessor.Preprocessor(p)

which returns an object containing the list of jobs to run.
Jobs are then run with:

.. code:: python

    >>> pp.run_jobs()


Model simulation duration in the high-level API
-----------------------------------------------

The duration of a model run configured with the high-level API can be set up with a number of different configuration parameters.

.. note:: see the complete description of "time" in the model: :doc:`../info/modeltime`.

Using the high-level API, you can specify the duration to run the model by two mechanisms: 1) the number of timesteps to run the model, or 2) the duration of time to run the model.

The former case is straightforward, insofar that the model determines the timestep duration and the high-level API simply iterates for the specified number of timestep iterations.
To specify the number of timesteps to run the model, use the argument ``--timesteps`` at the command line (or ``timesteps:`` in the configuration YAML file, or ``timesteps=`` with the Python :obj:`~pyDeltaRCM.Preprocessor`).

.. code:: bash
    
    $ pyDeltaRCM --config model_configuration.yml --timesteps 5000

The second case is more complicated, because the time specification is converted to model time according to a set of additional parameters.
In this case, the model run end condition is that the elapsed model time is *equal to or greater than* the specified input time.
Importantly, this means that the duration of the model run is unlikely to exactly match the input condition, because the model timestep is unlikely to be a factor of the specified time.
Again, refer to the complete description of model time :doc:`../info/modeltime` for more information.

To specify the duration of time to run the model in *seconds*, simply use the argument ``--time`` at the command line (or ``time:`` in the configuration YAML file, or ``time=`` with the Python :obj:`~pyDeltaRCM.Preprocessor`).
It is also possible to specify the input run duration in units of years with the similarly named argument ``--time_years`` (``time_years:``, ``time_years=``).

.. code:: bash
    
    $ pyDeltaRCM --config model_configuration.yml --time 31557600
    $ pyDeltaRCM --config model_configuration.yml --time_years 1

would each run a simulation for :math:`(86400 * 365.25)` seconds, or 1 year.

.. important::

    Do not specify both time arguments, or specify time arguments with the timesteps argument.
    In the case of multiple argument specification, precedence is given in the order `timesteps` > `time` > `time_years`.

When specifying the time to run the simulation, an additional parameter determining the intermittency factor (:math:`I_f`) may be specified ``--If`` at the command line (``If:`` in the YAML configuration file, ``If=`` with the Python :obj:`~pyDeltaRCM.Preprocessor`).
This argument will scale the specified time-to-model-time, such that the *scaled time* is equal to the input argument time.
Specifying the :math:`I_f`  value is essential when using the model duration run specifications.
See :doc:`../info/modeltime` for complete information on the scaling between model time and elapsed simulation time.

Running simulations in parallel
-------------------------------

The high-level API provides the ability to run simulations in parallel on Linux environments.
This option is only useful in the case where you are running multiple jobs with the :ref:`matrix expansion <matrix_expansion_tag>`, :ref:`ensemble expansion <ensemble_expansion_tag>`, or :ref:`set expansion <set_expansion_tag>` tools.

To run jobs in parallel simply specify the `--parallel` flag to the command line interface.
Optionally, you can specify the number of simulations to run at once by following the flag with a number.

.. code:: bash
    
    $ pyDeltaRCM --config model_configuration.yml --timesteps 5000 --parallel
    $ pyDeltaRCM --config model_configuration.yml --timesteps 5000 --parallel 6


Low-level model API
===================

The low-level API is the same as that described at the beginning of this guide.
Interact with the model by creating your own script, and manipulating model outputs at the desired level.
The simplest case to use the low-level API is to do

.. code::

    >>> delta = DeltaModel(input_file='model_configuration.yml')

    >>> for _ in range(0, 5000):
    ...    delta.update()

    >>> delta.finalize()

However, you can also inspect/modify the :obj:`~pyDeltaRCM.DeltaModel.update` method, and change the order of operations, or add operations, as desired; see the :ref:`guide to customizing the model <customize_the_model>` below.
If you are working with the low-level API, you can optionally pass any valid key in the YAML configuration file as a keyword argument during model instantiation. 
For example:

.. code::

    >>> delta = DeltaModel(input_file='model_configuration.yml',
    ...                    SLR=1e-9)


Keyword arguments supplied at this point will supersede values specified in the YAML configuration.
See :ref:`our guide for model customization <customize_the_model>` for a complete explanation and demonstration for how to modify model behavior.

..
    Advanced model configurations
    =============================
    ** Advanced model configuration guide is imported from another file. **

.. include:: advanced_configuration_guide.inc


..
    Working with Subsidence
    =======================
    ** Subsidence guide is imported from another file. **

.. include:: subsidence_guide.inc


Supporting documentation and files
==================================

Model reference:

    * :doc:`/reference/model/yaml_defaults`
    * :doc:`/reference/model/model_hooks`

Examples:

    * :doc:`/examples/slight_slope`
    * :doc:`/examples/subsidence_region`
    * :doc:`/examples/custom_saving`

References
==========

.. [1] A reduced-complexity model for river delta formation – Part 1: Modeling
       deltas with channel dynamics, M. Liang, V. R. Voller, and C. Paola, Earth
       Surf. Dynam., 3, 67–86, 2015. https://doi.org/10.5194/esurf-3-67-2015
