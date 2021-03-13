import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import pyDeltaRCM


# filter out the warning raised about no netcdf being found
warnings.filterwarnings("ignore", category=UserWarning)


n = 10
cm = matplotlib.cm.get_cmap('tab10')


# init delta model
with pyDeltaRCM.shared_tools._docs_temp_directory() as output_dir:
    delta = pyDeltaRCM.DeltaModel(
        out_dir=output_dir)

    delta_later = pyDeltaRCM.DeltaModel(
        out_dir=output_dir,
        resume_checkpoint='../../_resources/checkpoint')


_shp = delta_later.eta.shape


# manually call only the necessary paths
delta.init_water_iteration()
delta.run_water_iteration()

delta_later.init_water_iteration()
delta_later.run_water_iteration()


# define a function to fill in the walks of given idx
def _plot_idxs_walks_to_step(delta_inds, _step, _idxs, _ax):
    for i in range(len(_idxs)):
        iidx = _idxs[i]
        walk = delta_inds[iidx, :]
        walk = walk[:_step]
        pyDeltaRCM.debug_tools.plot_line(
            walk, shape=_shp, color=cm(i),
            nozeros=True)
        yend, xend = pyDeltaRCM.shared_tools.custom_unravel(
            walk[-1], _shp)
        _ax.plot(xend, yend,
                 marker='o', ms=3, color=cm(i))


# declare the idxs to use:
idxs = np.random.randint(low=0, high=delta._Np_water, size=n)
ps = [5, 25, 75]

# set up axis
fig, ax = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(12, 4))
vmin, vmax = delta.eta.min(), delta.eta.max()

# fill in axis0
pyDeltaRCM.debug_tools.plot_domain(
    delta.eta, ax=ax[0, 0], grid=False, cmap='cividis')
_plot_idxs_walks_to_step(
    delta.free_surf_walk_inds, _step=ps[0], _idxs=idxs, _ax=ax[0, 0])
ax[0, 0].set_title('after {} steps'.format(ps[0]))
pyDeltaRCM.debug_tools.plot_domain(
    delta_later.eta, ax=ax[1, 0], grid=False, vmin=vmin, vmax=vmax, cmap='cividis')
_plot_idxs_walks_to_step(
    delta_later.free_surf_walk_inds, _step=ps[0], _idxs=idxs, _ax=ax[1, 0])
# ax[1, 0].set_title('after {} steps'.format(ps[0]))


# fill in axis1
pyDeltaRCM.debug_tools.plot_domain(
    delta.eta, ax=ax[0, 1], grid=False, cmap='cividis')
_plot_idxs_walks_to_step(
    delta.free_surf_walk_inds, _step=ps[1], _idxs=idxs, _ax=ax[0, 1])
ax[0, 1].set_title('after {} steps'.format(ps[1]))
pyDeltaRCM.debug_tools.plot_domain(
    delta_later.eta, ax=ax[1, 1], grid=False, vmin=vmin, vmax=vmax, cmap='cividis')
_plot_idxs_walks_to_step(
    delta_later.free_surf_walk_inds, _step=ps[1], _idxs=idxs, _ax=ax[1, 1])
# ax[1, 1].set_title('after {} steps'.format(ps[1]))


# fill in axis2
pyDeltaRCM.debug_tools.plot_domain(
    delta.eta, ax=ax[0, 2], grid=False, cmap='cividis')
_plot_idxs_walks_to_step(
    delta.free_surf_walk_inds, _step=ps[2], _idxs=idxs, _ax=ax[0, 2])
ax[0, 2].set_title('after {} steps'.format(ps[2]))
pyDeltaRCM.debug_tools.plot_domain(
    delta_later.eta, ax=ax[1, 2], grid=False, vmin=vmin, vmax=vmax, cmap='cividis')
_plot_idxs_walks_to_step(
    delta_later.free_surf_walk_inds, _step=ps[2], _idxs=idxs, _ax=ax[1, 2])
# ax[1, 3].set_title('after {} steps'.format(ps[3]))


plt.tight_layout()
plt.show()
