import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import pyDeltaRCM


# filter out the warning raised about no netcdf being found
warnings.filterwarnings("ignore", category=UserWarning)


n = 20
cm = matplotlib.cm.get_cmap('tab10')


# init delta model
with pyDeltaRCM.shared_tools._docs_temp_directory() as output_dir:

    delta = pyDeltaRCM.DeltaModel(
        out_dir=output_dir,
        resume_checkpoint='../../_resources/checkpoint')


_shp = delta.eta.shape


# manually call only the necessary paths
delta.init_water_iteration()
delta.run_water_iteration()


# define a function to fill in the walks of given idx
def _plot_idxs_walks_to_step(delta_inds, _step, _idxs, _ax):
    for i in range(len(_idxs)):
        iidx = _idxs[i]
        walk = delta_inds[iidx, :]
        walk = walk[:_step]
        pyDeltaRCM.debug_tools.plot_line(
            walk, shape=_shp, color='r', alpha=0.5,
            nozeros=True)
        yend, xend = pyDeltaRCM.shared_tools.custom_unravel(
            walk[-1], _shp)
        _ax.plot(xend, yend,
                 marker='o', ms=3, color='r', alpha=0.5)


# declare the idxs to use:
np.random.seed(0)  # make it reproducible
idxs = np.random.randint(low=0, high=delta._Np_water, size=n)
ps = 100

# set up axis
fig, ax = plt.subplots(figsize=(6, 3))
vmin, vmax = delta.eta.min(), delta.eta.max()

# fill in axis2
ax.imshow(
    delta.eta, vmin=vmin, vmax=vmax, cmap='cividis')
ax.set_xticks([])
ax.set_yticks([])
_plot_idxs_walks_to_step(
    delta.free_surf_walk_inds, _step=ps, _idxs=idxs, _ax=ax)

plt.tight_layout()
plt.show()
