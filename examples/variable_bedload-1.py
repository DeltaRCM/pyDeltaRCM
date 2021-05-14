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