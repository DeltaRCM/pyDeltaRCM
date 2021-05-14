class ChangingVelocityModel(pyDeltaRCM.DeltaModel):
    """Model with changing flow velocity.

    Create a model that changes the inlet flow velocity throughout the run.
    In this example, the velocity is changed on each timestep, and the value
    it is set to is interpolated from a **predetermined timeseries** of
    velocities.
    """
    def __init__(self, input_file=None, end_time=86400, **kwargs):

        # inherit from the base model
        super().__init__(input_file, **kwargs)

        # set up the attributes for interpolation
        self._time_array, self._velocity_array = create_velocity_array(
            end_time)  # use default shape parameters for the array

    def hook_solve_water_and_sediment_timestep(self):
        """Change the velocity."""

        # find the new velocity and set it to the model
        self.u0 = np.interp(self._time, self._time_array, self._velocity_array)

        # log the new value
        _msg = 'Changed velocity value to {u0}'.format(
            u0=self.u0)
        self.log_info(_msg, verbosity=0)