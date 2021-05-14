Time-varying bedload
====================

The following example demonstrates how to configure a subclassing model with a time-varying parameter.
In this example, the time-varying behavior arises by managing two "switches", ``self._changed`` and ``self._changed_back``, which change the state of the :obj:`f_bedload` parameters at a predetermined time in the mode sequence.

The following codes produce two runs (using `matrix` expansion from the Preprocessor), which has a baseline `f_bedload` value of either ``0.3`` or ``0.7``, and for a period in the middle of the run, the `f_bedload` values are mirrored by ``self.f_bedload = (1 - self.f_bedload)``, i.e., briefly switching the bedload values.


.. plot::
    :context: reset
    :include-source:

    class VariableBedloadModel(pyDeltaRCM.DeltaModel):

        def __init__(self, input_file=None, **kwargs):

            super().__init__(input_file, **kwargs)

            self._changed = False
            self._changed_back = False

        def hook_solve_water_and_sediment_timestep(self):
            """Change the state depending on the _time.
            """
            # check if the state has been changed, and time to change it
            if (not self._changed) and (self._time > 459909090):
                self.f_bedload = (1 - self.f_bedload)
                self._changed = True

                _msg = 'Bedload changed to {f_bedload}'.format(
                    f_bedload=self.f_bedload)
                self.log_info(_msg, verbosity=0)

            # check if the state has been changed back, and time to change it back
            if (not self._changed_back) and (self._time > 714545454):
                self.f_bedload = (1 - self.f_bedload)
                self._changed_back = True

                _msg = 'Bedload changed back to {f_bedload}'.format(
                    f_bedload=self.f_bedload)
                self.log_info(_msg, verbosity=0)


.. rubric:: Checking on the state-change effect

To demonstrate how this works, let's loop through time and check the model state.
Here, we will change the model time value directly, so that we can verify that the model is working as intended, but you should never do this in practice.

.. plot::
    :context:

    with pyDeltaRCM.shared_tools._docs_temp_directory() as output_dir:
        mdl_muddy = VariableBedloadModel(f_bedload=0.3,
                                         out_dir=output_dir)
        mdl_sandy = VariableBedloadModel(f_bedload=0.7)

.. code:: python

    mdl_muddy = VariableBedloadModel(f_bedload=0.3,
                                     out_dir=output_dir)
    mdl_sandy = VariableBedloadModel(f_bedload=0.7)

.. important::

    You should never modify the model time via ``self._time`` directly when working with the model.


.. plot::
    :context:
    :include-source:

    # create a figure
    fig, ax = plt.subplots()

    # set the "simulation" range
    _times = np.linspace(0, 1e9, num=100)
    fb_mdl_muddy = np.zeros_like(_times)
    fb_mdl_sandy = np.zeros_like(_times)

    # loop through time, change the model time and grab f_bedload values
    for i, _time in enumerate(_times):
        # change the model time directly
        mdl_muddy._time = _time  # you should never do this
        mdl_sandy._time = _time  # you should never do this

        # run the hooked method
        mdl_muddy.hook_solve_water_and_sediment_timestep()
        mdl_sandy.hook_solve_water_and_sediment_timestep()

        # grab the state of the `f_bedload` parameter
        fb_mdl_muddy[i] = mdl_muddy.f_bedload  # get the value
        fb_mdl_sandy[i] = mdl_sandy.f_bedload  # get the value

    # add it to the plot
    ax.plot(_times, fb_mdl_muddy, '-', c='saddlebrown', lw=2, label='muddy')
    ax.plot(_times, fb_mdl_sandy, '--', c='goldenrod', lw=2, label='sandy')
    ax.legend()

    # clean up
    ax.set_ylim(0, 1)
    ax.set_ylabel('f_bedload')
    ax.set_xlabel('model time (s)')

    plt.show()


.. rubric:: Running the model for real

Given a yaml file (``variable_bedload.yaml``):

.. code:: yaml

    Length: 5000.
    Width: 10000.
    dx: 50
    N0_meters: 500
    C0_percent: 0.05
    SLR: 1.5e-8
    h0: 2
    u0: 1.1
    coeff_U_dep_mud: 0.5
    parallel: True
    
    matrix:
      f_bedload:
        - 0.3
        - 0.7


and a script to run the code:

.. code:: python
    
    if __name__ == '__main__':

        # base yaml configuration
        base_yaml = 'variable_bedload.yaml'

        pp = pyDeltaRCM.Preprocessor(
                base_yaml,
                timesteps=12000)

        # run the jobs
        pp.run_jobs(DeltaModel=VariableBedloadModel)

