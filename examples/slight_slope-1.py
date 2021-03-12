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