Constraining subsidence to part of the domain
=============================================

One case that has been explored in the literature with the DeltaRCM model is the case of subsidence limited to one region of the model domain [1]_.
This model configuration can be readily achieved with model subclassing.

Setting up the custom subclass
------------------------------

.. plot::
    :context: reset
    :include-source:

    class ConstrainedSubsidenceModel(pyDeltaRCM.DeltaModel):
        """A simple subclass of DeltaModel with subsidence region constrained.
    
        This subclass *overwrites* the `init_subsidence` method to
        constrain subsidence to only one region of the model domain.
        """
        def __init__(self, input_file=None, **kwargs):
    
             # inherit base DeltaModel methods
            super().__init__(input_file, **kwargs)

        def init_subsidence(self):
            """Initialize subsidence pattern constrained to a tighter region.

            Uses theta1 and theta2 to set the angular bounds for the
            subsiding region. theta1 and theta2 are set in relation to the
            inlet orientation. The inlet channel is at an angle of 0, if
            theta1 is -pi/3 radians, this means that the angle to the left of
            the inlet that will be included in the subsiding region is 30
            degrees. theta2 defines the right angular bounds for the subsiding
            region in a similar fashion.
            """
            _msg = 'Initializing custom subsidence field'
            self.log_info(_msg, verbosity=1)

            if self._toggle_subsidence:

                theta1 = -(np.pi / 3)
                theta2 = 0

                R1 = 0.3 * self.L  # radial limits (fractions of L)
                R2 = 0.8 * self.L

                Rloc = np.sqrt((self.y - self.L0)**2 + (self.x - self.W / 2.)**2)

                thetaloc = np.zeros((self.L, self.W))
                thetaloc[self.y > self.L0 - 1] = np.arctan(
                    (self.x[self.y > self.L0 - 1] - self.W / 2.)
                    / (self.y[self.y > self.L0 - 1] - self.L0 + 1))
                self.subsidence_mask = ((R1 <= Rloc) & (Rloc <= R2) &
                                        (theta1 <= thetaloc) &
                                        (thetaloc <= theta2))
                self.subsidence_mask[:self.L0, :] = False

                self.sigma = self.subsidence_mask * self.subsidence_rate * self.dt


Now, initialize the model and look at the field.
Note that the colorscale depicts the magnitude of subsidence in the model *per timestep* (`sigma`, which has units meters).

.. code:: python

    mdl = ConstrainedSubsidenceModel(toggle_subsidence=True)

.. plot::
    :context:

    with pyDeltaRCM.shared_tools._docs_temp_directory() as output_dir:
        mdl = ConstrainedSubsidenceModel(toggle_subsidence=True,
                                         out_dir=output_dir)

.. plot::
    :context:
    :include-source:

    fig, ax = plt.subplots()
    mdl.show_attribute('sigma', grid=False)
    plt.show()


Using the custom subclass with the preprocessor
-----------------------------------------------

We can configure a :obj:`Preprocessor` to handle a set of custom runs in conjunction with out custom `pyDeltaRCM` model subclass.
For example, in [1]_, the authors explore the impact of subsidence at various rates: 3 mm/yr, 6 mm/yr, 10 mm/yr, 25 mm/yr, 50 mm/yr, and 100 mm/yr.
We can scale these rates, assuming a model :doc:`intermittency factor </info/modeltime>` of 0.019, representing 7 of 365 days of flooding per year, by using the convenience function :obj:`~pyDeltaRCM.preprocessor.scale_relative_sea_level_rise_rate`:

.. plot::
    :context: close-figs
    :include-source:

    from pyDeltaRCM.preprocessor import scale_relative_sea_level_rise_rate

    subsidence_mmyr = np.array([3, 6, 10, 25, 50, 100])
    subsidence_scaled = scale_relative_sea_level_rise_rate(subsidence_mmyr, If=0.019)

Now, we use :ref:`matrix expansion <matrix_expansion_tag>` to set up the runs with a preprocessor.
For example, in a Python script, following the definition of the subclass above, define a dictionary with a `matrix` key and supply to the `Preprocessor`:

.. plot::
    :context:
    :include-source:

    # add a matrix with subsidence to the dict
    param_dict = {}
    param_dict['matrix'] = {'subsidence_rate': subsidence_scaled}

    # add other configurations
    param_dict.update(
        {'out_dir': 'liang_2016_reproduce',
         'toggle_subsidence': True,
         'parallel': 3})  # we can take advantage of parallel jobs

.. code::

    # create the preprocessor
    pp = pyDeltaRCM.Preprocessor(
        param_dict,
        timesteps=10000)

And finally run the jobs by specifying the model subclass as the class to use when instantiating the jobs with the preprocessor.

.. below, we overwrite the above, to make sure we only run for one timestep
.. plot::
    :context:

    with pyDeltaRCM.shared_tools._docs_temp_directory() as output_dir:
        param_dict['out_dir'] = output_dir
        pp = pyDeltaRCM.Preprocessor(
            param_dict,
            parallel=False,
            timesteps=1)
        pp.run_jobs(DeltaModel=ConstrainedSubsidenceModel)

.. code:: python

    # run the jobs
    pp.run_jobs(DeltaModel=ConstrainedSubsidenceModel)

We can check whether the runs were set up, as expected:

.. plot::
    :context:
    :include-source:

    from matplotlib.colors import Normalize

    fig, ax = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(10, 4))
    norm = Normalize(vmin=3, vmax=100)

    for i, job in enumerate(pp.job_list):
        # first convert the field to a rate
        subsidence_rate_field = (job.deltamodel.sigma / job.deltamodel.dt)

        # now convert to mm/yr
        subsidence_rate_field = (subsidence_rate_field * 1000 *
            pyDeltaRCM.shared_tools._scale_factor(If=0.019, units='years'))

        # and display
        im = ax.flat[i].imshow(subsidence_rate_field, norm=norm)

    fig.colorbar(im, ax=ax.ravel().tolist())
    plt.show()


.. [1] Liang, M., Kim, W., and Passalacqua, P. (2016), How much subsidence is
   enough to change the morphology of river deltas?, Geophysical Research Letters, 43, 10,266--10,276, doi:10.1002/2016GL070519.
