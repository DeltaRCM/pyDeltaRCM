Slightly sloping basin
======================

Consider the case where we are a researcher seeking to explore the effects of a receiving basin that is sloped perpendicular to the channel outlet. 
This researcher asks: does this sloped basin cause channels to steer towards the deeper water, where compensation is higher?

The researcher can easily use subclassing and model hooks to achieve the desired effect.
Recall that anything added to the end of the the subclass' `__init__` method will be called during instantiation of the subclass.

.. plot::
    :context: reset
    :include-source:

    class SlightSlopeModel(pyDeltaRCM.DeltaModel):
        """A subclass of DeltaModel with sloping basin.
    
        This subclass simply modifies the basin geometry
        before any computation has occurred.
        """
        def __init__(self, input_file=None, **kwargs):
    
             # inherit base DeltaModel methods
            super().__init__(input_file, **kwargs)

             # modify the basin
            slope = 0.0005  # cross basin slope
            eta_line = slope * np.arange(0, self.Width,
                                          step=self.dx)
            eta_grid = np.tile(eta_line, (self.L - self.L0, 1))
            eta_grid = eta_grid - ((slope * self.Width)/2)  # center at inlet
            self.eta[self.L0:, :] += eta_grid

Next, we instantiate the model class.

.. code::

    mdl = SlightSlopeModel()


.. plot::
    :context:

    with pyDeltaRCM.shared_tools._docs_temp_directory() as output_dir:
        mdl = SlightSlopeModel(out_dir=output_dir)


And finally, make a plot of the initial condition using the :obj:`~pyDeltaRCM.debug_tools.debug_tools.show_attribute` method.

.. plot::
    :context:
    :include-source:

    fig, ax = plt.subplots()
    mdl.show_attribute('eta', grid=False)
    plt.show()

You can try this out for yourself, and even complete the model run.
Are the channels steered by the basin slope?

.. important:: 

    In this example, we did not take care to update the model `stage` or `depth` fields. In this simple case it works out fine, because after a single timestep, the fields are correctly computed relative to the modified bed. However, take caution when modifying `DeltaModel` fields directly, and be sure to change *all* relevant fields too.
