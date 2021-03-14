Time-varying bedload
====================

The following example demonstrates how to configure a subclassing model with a time-varying parameter.
In this example, the time-varying behavior arises by managing two "switches", ``self._changed`` and ``self._changed_back``, which change the state of the :obj:`f_bedload` parameters at a predetermined time in the mode sequence.

The following codes produce two runs (using `matrix` expansion from the Preprocessor), which has a baseline `f_bedload` value of either ``0.3`` or ``0.7``, and for a period in the middle of the run, the `f_bedload` values are mirrored by ``self.f_bedload = (1 - self.f_bedload)``, i.e., briefly switching the bedload values.


.. code:: python

    class VariableBedloadModel(pyDeltaRCM.DeltaModel):

        def __init__(self, input_file, **kwargs):

            super().__init__(input_file, **kwargs)

            self._changed = False
            self._changed_back = False

        def hook_run_one_timestep(self):
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

