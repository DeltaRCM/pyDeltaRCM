******************
10-minute tutorial
******************

Use pyDeltaRCM in ten minutes!


You can get a model running with three simple lines of code.
First, we instantiate the main :obj:`~pyDeltaRCM.deltaRCM_driver.pyDeltaRCM` model object.

.. code:: python

    >>> import pyDeltaRCM

    >>> default_delta = pyDeltaRCM.DeltaModel()

Instantiating the :obj:`~pyDeltaRCM.model.DeltaModel()` without any arguments will use a set of :doc:`default parameters <../reference/model/yaml_defaults>` to configure the model run.
The default options are a reasonable set for exploring some controls of the model, and would work perfectly well for a simple demonstration here.
However, to run a simulation with a non-default set of parameters, we can use a configuration file written in the YAML markup language named `10min_tutorial.yaml`.
For example, we can specify where we would like the output file to be placed with the `out_dir` parameter, and ensure that our simulation is easily reproducible by setting the random `seed` parameter:

.. code:: yaml

    out_dir: '10min_tutorial'
    seed: 451220118313

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
