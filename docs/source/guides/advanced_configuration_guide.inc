=============================
Advanced model configurations
=============================

.. _configuring_multiple_jobs:

Configuring multiple model runs from a single YAML file
==============================================================

Multiple model runs (referred to as "jobs") can be configured by a single `.yml` configuration file, by using the `matrix` and `ensemble` configuration keys.

.. _matrix_expansion_tag:

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

.. _ensemble_expansion_tag:

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

.. _set_expansion_tag:

Set expansion
-------------

Set expansion enables user-configured parameter sets to take advantage of the :obj:`~pyDeltaRCM.Preprocessor` infrastructure (such as the job output preparation and ability to run jobs in parallel), while also enabling flexible configurations for parameter sets than cannot be configured via `matrix` expansion.
For example, to vary `Qw0` while holding `Qs0` fixed requires modifying both `C0_percent` and some water-discharge-controlling parameter *simultaneously*; i.e., this cannot be achieved with `matrix` expansion. 

To use set expansion, add the `set` key to a configuration file, and define a *list* of *dictionaries* which set the parameters of each run to be completed.
For example, to configure two model runs, the first with parameters ``u0: 1.0`` and ``h0: 1.0``, and the second with parameters ``u0: 1.2`` and ``h0: 1.2``:

.. code:: yaml

    set:
      - {u0: 1.0, h0: 1.0}
      - {u0: 1.2., h0: 1.2}


All jobs in the `set` specification must have the exact same set of keys.
Moreover, additional `ensemble` or `matrix` specifications are not supported with the `set` specification. 


Customizing model operations with subclasses and hooks
======================================================

.. _customize_the_model:

The :obj:`~pyDeltaRCM.DeltaModel` is designed for flexibility and extension by users, and to support arbitrary and imaginative changes to the model routine.
For example, one could easily extend the model to include additional delta controls (such as vegetation or permafrost development), or modify the model domain boundary conditions (such as imposing a sloped receiving basin).
This flexibility is achieved by "subclassing" the `DeltaModel` to create a custom model object, and using "hooks" in the model to achieve the desired modifications.

Subclassing is a standard concept in object-oriented programming, whereby a `subclass` obtains all the functionality of the `parent` object, and then adds/modifies existing functionality of the parent, to create a new class of object (i.e., the `subclass`).
To subclass the :obj:`~pyDeltaRCM.model.DeltaModel` we simply create a new Python object class, which inherits from the model class:

.. doctest::

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import pyDeltaRCM

    >>> class SubclassDeltaModel(pyDeltaRCM.DeltaModel):
    ...     def __init__(self, input_file=None):
    ...
    ...         # inherit base DeltaModel methods
    ...         super().__init__(input_file)

We then can initialize our new model type, and see that this model has all of the attributes and functionality of the original `DeltaModel`, but its type is our subclass.

.. doctest::

    >>> mdl = SubclassDeltaModel()

    >>> mdl
    <SubclassDeltaModel object at 0x...>
    
    # for example, the subclass `mdl` has the `update` method
    >>> hasattr(mdl, 'update')
    True

Hooks are methods in the model sequence that do nothing by default, but can be augmented to provide arbitrary desired behavior in the model.
Hooks have been integrated throughout the model initialization and update sequences, to allow the users to achieve complex behavior at various stages of the model sequence.
For example, ``after_init()`` is a hook which occurs after the default model initialization is complete.
To utilize the hooks, we simply define a method in our subclass with the name corresponding to the hook we want to augment to achieve the desired behavior.

So, as an example we don't recommend trying out, change the number of steps a water or sediment parcel could take by changing the model definition from above:

.. doctest::

    >>> class TenStepDeltaModel(pyDeltaRCM.DeltaModel):
    ...     def __init__(self, input_file=None):
    ...
    ...         # inherit base DeltaModel methods
    ...         super().__init__(input_file)
    ...
    ...     def after_init(self):
    ...
    ...         # default is (2 x (L + W))
    ...         self.itmax = 10  # always 10

Now, during the initializing of a `TenStepDeltaModel` instance, our implementation of `after_init()` will be called and `self.itmax` will be adjusted accordingly. 

.. doctest::

    >>> mdl = TenStepDeltaModel()
    >>> mdl.itmax
    10

This is a somewhat contrived example to give you a sense of how you can implement changes to the model to achieve desired behavior. 

A complete list of model hooks is available here, but model hooks all follow the convention of beginning with the prefix `hook_` and include the name of the method they *immediately precede* in the model; for example `hook_run_water_iteration` is called immediately before `run_water_iteration` is called.

There may be cases where hooks are insufficient for the modifications you need to make to the model.
In this case


Subclassing examples
--------------------

We maintain a number of :doc:`examples of pyDeltaRCM use</examples/index>`, which show  DeltaModel subclasses implementing various hooks, and reimplementing existing methods, to achieve complex behavior.