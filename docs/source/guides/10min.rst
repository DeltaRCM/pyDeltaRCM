******************
10-minute tutorial
******************

Use pyDeltaRCM in ten minutes!
This simple guide will show you the absolute basics of getting a `pyDeltaRCM` model running, and give you some direction on where to go from there.

.. important::

    If you haven't already, be sure to follow the :doc:`installation guide </meta/installing>` to get *pyDeltaRCM* set up properly on your computer.


A default model
---------------

You can get a model running with five simple lines of code.
First, we instantiate the main :obj:`~pyDeltaRCM.DeltaModel` model object.

.. code:: python

    >>> import pyDeltaRCM

    >>> default_delta = pyDeltaRCM.DeltaModel()

Instantiating the :obj:`~pyDeltaRCM.model.DeltaModel()` without any arguments will use a set of :doc:`default parameters <../reference/model/yaml_defaults>` to configure the model run.
The default options are a reasonable set for exploring some controls of the model, and would work perfectly well for a simple demonstration here.

The delta model is run forward with a call to the :meth:`~pyDeltaRCM.DeltaModel.update()` method.
So, we simply create a `for` loop, and call the `update` function, and then wrap everything up with a call to :meth:`~pyDeltaRCM.DeltaModel.finalize` the model:

.. code:: python

    >>> for _ in range(0, 5):
    ...     delta.update()

    >>> delta.finalize()

That's it! You ran the pyDeltaRCM model for five timesteps, with just five lines of code. 

We can visualize the delta bed elevation, though it's not very exciting after only five timesteps...

.. code:: python

    >>> import matplotlib.pyplot as plt

    >>> fig, ax = plt.subplots()
    >>> ax.imshow(delta.bed_elevation, vmax=-3)
    >>> plt.show()

.. plot:: guides/10min_demo.py


The model with set parameters
-----------------------------

To run a simulation with a non-default set of parameters, we use a configuration file written in the YAML markup language named `10min_tutorial.yaml`.
For example, we can specify where we would like the output file to be placed with the `out_dir` parameter, ensure that our simulation is easily reproducible by setting the random `seed` parameter, and examine what is the effect of a high fraction of bedload:

.. code:: yaml

    out_dir: '10min_tutorial'
    seed: 451220118313
    f_bedload: 0.9

Now, we can create a second instance of the :obj:`~pyDeltaRCM.model.DeltaModel()`, this time using the input yaml file.

.. code::

    >>> second_delta = pyDeltaRCM.DeltaModel(input_file='10min_tutorial.yaml')

and repeat the same `for` loop operation as above:

.. code:: python

    >>> for _ in range(0, 5):
    ...     second_delta.update()

    >>> second_delta.finalize()


Resources
---------

Consider reading through the :doc:`User Guide <user_guide>` as a first action, and determine how to set up the model to complete your experiment, including tutorials and examples for customizing the model to achieve any arbitrary behavior you need!

* :doc:`user_guide`
* :doc:`/reference/model/index`
