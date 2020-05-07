******************
10-minute tutorial
******************

Use pyDeltaRCM in ten minutes!


You can get a model running with three simple lines of code.
First, we instantiate the main :obj:`~pyDeltaRCM.deltaRCM_driver.pyDeltaRCM` model object.

.. code:: 

    >>> import pyDeltaRCM

    >>> delta = pyDeltaRCM.deltaRCM_driver.pyDeltaRCM()

Next, since this is just a simple demo, we will run for a few short timesteps.
The delta model is run forward with a call to the :meth:`~pyDeltaRCM.deltaRCM_driver.pyDeltaRCM.update()` method of the delta model.
So we loop a few times and then finalize the model:

.. code::

    >>> for _t in range(0, 1):
    >>>     delta.update()

    >>> delta.finalize()

That's it! You ran the pyDeltaRCM model for one timestep. 

We can visualize the delta bed elevation, though it's not very exciting after only one timestep...

.. code::

    >>> fig, ax = plt.subplots()
    >>> ax.imshow(delta.bed_elevation, vmax=-4.5)
    >>> plt.show()

.. plot:: 10min/model_run_visual.py
