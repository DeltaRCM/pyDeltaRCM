import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import pyDeltaRCM


# filter out the warning raised about no netcdf being found
warnings.filterwarnings("ignore", category=UserWarning)


n = 1
cm = matplotlib.cm.get_cmap('tab10')


# init delta model
with pyDeltaRCM.shared_tools._docs_temp_directory() as output_dir:
    delta = pyDeltaRCM.DeltaModel(
        out_dir=output_dir,
        resume_checkpoint='../../_resources/checkpoint')
_shp = delta.eta.shape


delta.init_water_iteration()
delta.run_water_iteration()


# define a function to fill in the walks of given idx
def _plot_idxs_walks_to_step(delta_inds, _step, _idxs, _ax):
    for i in range(len(_idxs)):
        walk = delta_inds[_idxs[i], :]
        walk = walk[:_step]
        # print(walk)
        pyDeltaRCM.debug_tools.plot_line(
            walk, shape=_shp, color='c',
            multiline=True, nozeros=True)
        yend, xend = pyDeltaRCM.shared_tools.custom_unravel(walk[-1], _shp)
        _ax.plot(xend, yend,
                 marker='o', ms=3, color='c')


# declare the idxs to use:
idxs = np.random.randint(low=0, high=delta._Np_water, size=n)
pidx = 85


# set up axis
# fig, ax = plt.subplots(1, 4, figsize=(10, 5))
fig = plt.figure(constrained_layout=True)
gs = fig.add_gridspec(2, 2)


# fill in axis0
ax0 = fig.add_subplot(gs[0, 0])
pyDeltaRCM.debug_tools.plot_domain(
    delta.eta, ax=ax0, grid=False, cmap='cividis')
_plot_idxs_walks_to_step(
    delta.free_surf_walk_inds, _step=pidx, _idxs=idxs, _ax=ax0)
ax0.set_title('bed elevation')


# fill in axis0
ax1 = fig.add_subplot(gs[0, 1])
pyDeltaRCM.debug_tools.plot_domain(
    delta.stage, ax=ax1, grid=False)
_plot_idxs_walks_to_step(delta.free_surf_walk_inds, _step=pidx, _idxs=idxs, _ax=ax1)
ax1.set_title('water surface (stage)')


ax2 = fig.add_subplot(gs[1, :])
ax2.axhline(y=0, xmin=0, xmax=pidx+1, ls='--', color='0.6')
for i in range(n):
    walk = delta.free_surf_walk_inds[idxs[i], :]
    walk = walk[:pidx]

    ax2.plot(delta.eta.flat[walk], 'k-')
    ax2.plot(delta.stage.flat[walk], '-', color=cm(i))

ax2.set_ylabel('elevation')
ax2.set_xlabel('steps along parcel path')

plt.tight_layout()

plt.show()
