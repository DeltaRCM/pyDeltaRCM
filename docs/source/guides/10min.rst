******************
10-minute tutorial
******************

Use pyDeltaRCM in ten minutes!
This simple guide will show you the absolute basics of getting a `pyDeltaRCM` model running, and give you some direction on where to go from there.

.. important::

    If you haven't already, be sure to follow the :doc:`installation guide </meta/installing>` to get *pyDeltaRCM* set up properly on your computer.


A default model
---------------

You can get a model running with five simple lines of Python code. Note that you can run *pyDeltaRCM* in either a standalone script or part of an interactive session.
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
    ...     default_delta.update()

    >>> default_delta.finalize()

.. note::

    Additional calls to update the model can be called up until the model is finalized.

That's it! You ran the pyDeltaRCM model for five timesteps, with just five lines of code.

We can visualize the delta bed elevation, though it's not very exciting after only five timesteps...

.. code:: python

    >>> import matplotlib.pyplot as plt

    >>> fig, ax = plt.subplots()
    >>> ax.imshow(default_delta.bed_elevation, vmax=-3)
    >>> plt.show()

.. plot:: guides/10min_demo.py


The model with set parameters
-----------------------------

To run a simulation with a non-default set of parameters, we use a configuration file written in the YAML markup language named `10min_tutorial.yaml`.
The markup file allows us to specify model boundary conditions and input and output settings, where anything set in the file will override the :doc:`default parameters <../reference/model/yaml_defaults>` for the model, and anything *not* specified will take the default value.

.. important::

    The best practice for model configurations is to create a YAML file with only the settings you want to change specified.

The YAML configuration file is central to managing *pyDeltaRCM* simulations, so we did not create this file for you; you will need to create the YAML file yourself.
To create the YAML file, open up your favorite plain-text editing application (e.g., gedit, notepad).
YAML syntax is pretty simple for basic configurations, essentially amounting to each line representing a parameter-value pair, separated by a colon.
For this example, let's specify three simulation controls: where we want the output file to be placed via the `out_dir` parameter, we will ensure that our simulation is easily reproducible by setting the random `seed` parameter, and we can examine what is the effect of a high fraction of bedload with the `f_bedload` parameter.
Enter the following in your text editor, and save the file as ``10min_tutorial.yaml``, making sure to place the file in a location accessible to your interpreter.

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
