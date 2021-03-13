******************
10-minute tutorial
******************

Use pyDeltaRCM in ten minutes!
This simple guide will show you the absolute basics of getting a `pyDeltaRCM` model running, and give you some direction on where to go from there.


A default model
---------------

You can get a model running with three simple lines of code.
First, we instantiate the main :obj:`~pyDeltaRCM.deltaRCM_driver.pyDeltaRCM` model object.

.. code:: python

    >>> import pyDeltaRCM

    >>> default_delta = pyDeltaRCM.DeltaModel()

Instantiating the :obj:`~pyDeltaRCM.model.DeltaModel()` without any arguments will use a set of :doc:`default parameters <../reference/model/yaml_defaults>` to configure the model run.
The default options are a reasonable set for exploring some controls of the model, and would work perfectly well for a simple demonstration here.


The model with set parameters
-----------------------------

To run a simulation with a non-default set of parameters, we can use a configuration file written in the YAML markup language named `10min_tutorial.yaml`.
For example, we can specify where we would like the output file to be placed with the `out_dir` parameter, ensure that our simulation is easily reproducible by setting the random `seed` parameter, and examine what is the effect of a high fraction of bedload:

.. code:: yaml

    out_dir: '10min_tutorial'
    seed: 451220118313
    f_bedload: 0.9

Now, we can create a second instance of the :obj:`~pyDeltaRCM.model.DeltaModel()`, this time using the input yaml file.

.. code::

    >>> delta = pyDeltaRCM.DeltaModel(input_file='10min_tutorial.yaml')

Next, since this is just a simple demo, we will run for a few short timesteps.
The delta model is run forward with a call to the :meth:`~pyDeltaRCM.DeltaModel.update()` method of the delta model.
So we loop the `update` function, and then finalize the model:

.. code:: python

    >>> for _ in range(0, 5):
    ...     delta.update()

    >>> delta.finalize()

That's it! You ran the pyDeltaRCM model for five timesteps. 

We can visualize the delta bed elevation, though it's not very exciting after only five timesteps...

.. code:: python

    >>> import matplotlib.pyplot as plt

    >>> fig, ax = plt.subplots()
    >>> ax.imshow(delta.bed_elevation, vmax=-3)
    >>> plt.show()

.. plot:: 10min/model_run_visual.py


Resources
---------

Consider reading through the :doc:`User Guide <user_guide>` as a first action, and determine how to set up the model to complete your experiment, including tutorials and examples for customizing the model to achieve any arbitrary behavior you need!

* :doc:`user_guide`
* :doc:`/reference/model/index`
