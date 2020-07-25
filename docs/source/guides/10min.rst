******************
10-minute tutorial
******************

Use pyDeltaRCM in ten minutes!


You can get a model running with three simple lines of code.
First, we instantiate the main :obj:`~pyDeltaRCM.deltaRCM_driver.pyDeltaRCM` model object.

.. doctest:: 

    >>> import pyDeltaRCM

    >>> delta = pyDeltaRCM.DeltaModel()

Next, since this is just a simple demo, we will run for a few short timesteps.
The delta model is run forward with a call to the :meth:`~pyDeltaRCM.DeltaModel.update()` method of the delta model.
So we loop the `update` function, and then finalize the model:

.. doctest::

    >>> for _ in range(0, 5):
    ...     delta.update()

    >>> delta.finalize()

That's it! You ran the pyDeltaRCM model for five timesteps. 

We can visualize the delta bed elevation, though it's not very exciting after only five timesteps...

.. code::

    >>> import matplotlib.pyplot as plt

    >>> fig, ax = plt.subplots()
    >>> ax.imshow(delta.bed_elevation, vmax=-4)
    >>> plt.show()

.. plot:: 10min/model_run_visual.py
