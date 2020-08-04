import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1 as axtk

import numpy as np

import pyDeltaRCM


class SlightSlopeModel(pyDeltaRCM.DeltaModel):
    def __init__(self, input_file):
        super().__init__(input_file)

    def after_init(self):
        """Called at end of initialization."""
        _slope = 0.0005
        lin = _slope * np.arange(0, self.Width, step=self.dx)
        grid = np.tile(lin, (self.L - self.L0, 1)) - ((_slope*self.Width)/2)
        self.eta[self.L0:, :] = self.eta[self.L0:, :] + grid


mdl = SlightSlopeModel('./slight_slope.yml')

fig, ax = plt.subplots(figsize=(5, 3))
im = ax.pcolormesh(mdl.X, mdl.Y, mdl.bed_elevation,
                   shading='flat', vmin=-10, vmax=1)
ax.set_xlim((0, mdl.Width))
ax.set_ylim((0, mdl.Length))
ax.set_aspect('equal', adjustable='box')
divider = axtk.axes_divider.make_axes_locatable(ax)
cax = divider.append_axes("right", size="2%", pad=0.05)
cb = plt.colorbar(im, cax=cax)
cb.ax.tick_params(labelsize=7)
ax.use_sticky_edges = False
ax.margins(y=0.2)
ax.text(0.95, 0.875, 'initial bed elevation\n(m)', fontsize=8,
        horizontalalignment='right', verticalalignment='center',
        transform=ax.transAxes)
plt.show()
