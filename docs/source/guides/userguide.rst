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
    seed: 42

then a :obj:`~pyDeltaRCM.DeltaModel` model instance initialized with this file specified as ``input_file`` will have a slope of 0.005, and will use a random seed of 42.
Multiple parameters can be specified line by line.

Default values are substituted for any parameter not explicitly given in the ``input_file`` ``.yml`` file.
Default values of the YAML configuration are listed in the :doc:`../reference/model/yaml_defaults`.


===================
Starting model runs
===================

There are two API levels at which you can interact with the pyDeltaRCM model.
There is a "high-level" model API, which takes as argument a YAML configuration file, and will compose a list of jobs as indicated in the YAML file; the setup can be configured to automatically execute the job list, as well.
The "low-level" API consists of creating a model instance from a YAML configuration file and manually handling the timestepping, or optionally, augmenting operations of the model to implement new features.


High-level model API
====================

The high-level API is accessed via either a shell prompt or python script, and is invoked directly if a YAML configuration file includes the ``timesteps`` variable.

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

These invokations will run the pyDeltaRCM :obj:`preprocessor <pyDeltaRCM.preprocessor.PreprocessorCLI>` with the parameters specified in the ``model_configuration.yml`` file.
If the YAML configuration indicates multiple jobs (:ref:`via matrix expansion or ensemble specification <configuring_multiple_jobs>`), the jobs will each be run automatically by calling :obj:`~pyDeltaRCM.DeltaModel.update` on the model 500 times.



Python API
----------

The Python high-level API is accessed via the :obj:`~pyDeltaRCM.Preprocessor` object.
First, the `Preprocessor` is instantiated with a YAML configuration file (e.g., ``model_configuration.yml``):

.. code::

    >>> pp = preprocessor.Preprocessor(p)

which returns an object containing the list of jobs to run.
Jobs are then run with:

.. code::

    >>> pp.run_jobs()



Low-level model API
===================

iinteract with the model by creating your own script, and manipulating model outputs at the desired level. The simplest case is to do

.. code::

    >>> delta = DeltaModel(input_file='model_configuration.yml')

    >>> for _ in range(0, 1):
    ...    delta.update()

    >>> delta.finalize()

However, you can also inspect/modify the :obj:`~pyDeltaRCM.DeltaModel.update` method, and change the order of operations, or add operations, as desired.


=============================
Advanced model configurations
=============================

.. _configuring_multiple_jobs:

Configuring multiple model runs from a single YAML file
==============================================================

Multiple model runs (referred to as "jobs") can be configured by a single `.yml` configuration file, by using the `matrix` and `ensemble` configuration keys.

Matrix expansion
----------------

To use matrix expansion to configure multiple model runs, the dimensions of the matrix (i.e., the variables you want to run) should be listed below the `matrix` key. For example, the following configuration is a one-dimensional matrix with the variable `f_bedload`:

.. code:: yaml

    out_dir: 'out_dir'
    dx: 2.0
    h0: 1.0

    matrix:
      f_bedload:
        - 0.5
        - 0.2

This configuation would produce two model runs, one with bedload fraction (`f_bedload`) 0.5 and another with bedload fraction 0.2, and both with grid spacing (`dx`) 2.0 and basin depth (`h0`) 1.0.
The matrix expansions will create two folders at `./out_dir/job_000` and `./out_dir/job_001` that each correspond to a created job.
Each folder will contain a copy of the configuration file used for that job; for example, the full configuration for `job_000` is:

.. code:: yaml

    out_dir: 'out_dir/job_000'
    dx: 2.0
    h0: 1.0
    f_bedload: 0.5

Additionally, a log file for each job is located in the output folder, and any output grid files or images specified by the input configuration will be located in the respective job output folder.

.. note:: You must specify the `out_dir` key in the input YAML configuation to use matrix expansion.

Multiple dimensional matrix expansion is additionally supported. For example, the following configuation produces six jobs:

.. code:: yaml

    out_dir: 'out_dir'

    matrix:
      f_bedload:
        - 0.5
        - 0.4
        - 0.2
      h0:
        - 1
        - 5


Ensemble expansion
------------------

Ensemble expansion creates replicates of specified model configurations with different random seed values.
Like the matrix expansion, the `out_dir` key must be specified in the input configuration file.
The `ensemble` key can be added to any configuration file that does not explicitly define the random seed.
As an example, two model runs can be generated with the same input sediment fraction using the following configuration `.yml`:

.. code:: yaml

    out_dir: 'out_dir'

    f_bedload: 0.5
    ensemble: 2

This configuration file would produce two model runs that share the same parameters, but have different initial random seed values.
The ensemble expansion can be applied to configuration files that include a matrix expansion as well:

.. code:: yaml

    out_dir: 'out_dir'

    ensemble: 3

    matrix:
      h0:
        - 1.0
        - 2.0

The above configuration file would produce 6 model runs, 3 with a basin depth (`h0`) of 1.0, and 3 with a basin depth of 2.0.
