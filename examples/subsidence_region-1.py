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