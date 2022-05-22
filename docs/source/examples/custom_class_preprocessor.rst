Using custom classes with the Preprocessor
==========================================

Here, we set up three jobs to run as an ensemble of a single custom class.


.. plot::
    :context: reset
    :include-source:

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt

    >>> import pyDeltaRCM

    >>> class CustomRandomModel(pyDeltaRCM.DeltaModel):
    ...     """
    ...     A subclass of DeltaModel that runs fast for this example.
    ... 
    ...     Just for the sake of this example, we are implementing a 
    ...     custom class that runs very quickly. We override the 
    ...     `solve_water_and_sediment_timestep` method of the model to 
    ...     simply add random gaussian blobs to the surface on each step.
    ...     """
    ...     def __init__(self, input_file=None, **kwargs):
    ...     
    ...         # inherit base DeltaModel methods
    ...         super().__init__(input_file, **kwargs)
    ...     
    ...         self.noise_patch = int(25)
    ...         self.noise_size = 5     # x y scale 
    ...         self.noise_scale = 200  # z scale
    ...         self._half_ns = self.noise_patch // 2
    ...         self.noise_x, self.noise_y = np.meshgrid(
    ...             np.linspace(-self._half_ns, self._half_ns, num=self.noise_patch),
    ...             np.linspace(-self._half_ns, self._half_ns, num=self.noise_patch))
    ...     
    ...     def solve_water_and_sediment_timestep(self):
    ...         """Overwrite method for documentation demonstration.
    ...     
    ...         This method now simply adds random gaussian noise on each step.
    ...         """         
    ...         # get a random x and y value
    ...         #   important: use get_random_uniform for reproducibility!
    ...         x, y, z = [pyDeltaRCM.shared_tools.get_random_uniform(1) for _ in range(3)]
    ...         
    ...         # rescale to fit inside domain
    ...         x = int(x * (self.L - self.noise_patch))
    ...         y = int(y * (self.W - self.noise_patch))
    ...     
    ...         # generate the blob
    ...         sx = sy = self.noise_size
    ...         exp = np.exp(-((self.noise_x)**2. / (2. * sx**2.) + (self.noise_y)**2. / (2. * sy**2.)))
    ...         blob = (1. / (2. * np.pi * sx * sy) * exp * self.noise_scale)
    ...         
    ...         # place into domain
    ...         self.eta[x:x+self.noise_patch, y:y+self.noise_patch] += blob


Then, we pass this custom subclass to the :obj:`~pyDeltaRCM.preprocessor.Preprocessor.run_jobs` method.

.. plot::
    :context:

    with pyDeltaRCM.shared_tools._docs_temp_directory() as output_dir:
        param_dict = dict(
            out_dir=output_dir,
            ensemble=3,
            timesteps=50
            )

        # let the preprocessor set up the jobs for you from checkpoint
        pp = pyDeltaRCM.Preprocessor(
            param_dict)

        # run the jobs
        pp.run_jobs(DeltaModel=CustomRandomModel)

.. code::

    >>> # set up dictionary for parameters and create a `Preprocessor`
    >>> param_dict = dict(
    ...     ensemble=3,
    ...     timesteps=50
    ...     )

    >>> # preprocessor set up the jobs
    >>> pp = pyDeltaRCM.Preprocessor(
    ...     param_dict)

    >>> # run the jobs with custom class!
    >>> pp.run_jobs(DeltaModel=CustomRandomModel)

.. plot::
    :context:
    :include-source:

    >>> fig, ax = plt.subplots(
    ...     1, len(pp.job_list),
    ...     figsize=(12, 4.8))
    >>> for i in range(len(pp.job_list)):
    ...     ax[i].imshow(pp.job_list[i].deltamodel.eta)
    >>> plt.tight_layout()
    >>> plt.show()
