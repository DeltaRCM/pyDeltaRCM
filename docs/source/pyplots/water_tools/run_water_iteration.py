import matplotlib.pyplot as plt
import numpy as np
import pyDeltaRCM
import matplotlib


n = 10
cm = matplotlib.cm.get_cmap('tab10')


# init delta model
delta = pyDeltaRCM.DeltaModel()
_shp = delta.eta.shape

# init delta model
delta_later = pyDeltaRCM.DeltaModel(
    '../../_resources/checkpoint.yaml',
    resume_checkpoint='../../_resources/deltaRCM_Output')
_shp = delta_later.eta.shape



# manually call only the necessary paths
delta.init_water_iteration()
delta.run_water_iteration()

delta_later.init_water_iteration()
delta_later.run_water_iteration()


# define a function to fill in the walks of given idx
def _plot_idxs_walks_to_step(delta_inds, _step, _idxs, _ax):
    for i in range(len(_idxs)):
        _hld = np.zeros((_step, 2))
        iidx = _idxs[i]
        walk = delta_inds[iidx, :]
        walk = walk[:_step]
        pyDeltaRCM.debug_tools.plot_line(walk, shape=_shp, color=cm(i))
        yend, xend = pyDeltaRCM.shared_tools.custom_unravel(walk[-1], _shp)
        _ax.plot(xend, yend,
                 marker='o', ms=3, color=cm(i))


# declare the idxs to use:
idxs = np.random.randint(low=0, high=delta._Np_water, size=n)
ps = [5, 15, 40, 60]

# set up axis
fig, ax = plt.subplots(2, 4, sharex=True, sharey=True, figsize=(10, 5))
vmin, vmax = delta.eta.min(), delta.eta.max()

# fill in axis0
pyDeltaRCM.debug_tools.plot_domain(
    delta.eta, ax=ax[0, 0], grid=False)
_plot_idxs_walks_to_step(delta.free_surf_walk_inds, _step=ps[0], _idxs=idxs, _ax=ax[0, 0])
ax[0, 0].set_title('after {} steps'.format(ps[0]))
pyDeltaRCM.debug_tools.plot_domain(
    delta_later.eta, ax=ax[1, 0], grid=False, vmin=vmin, vmax=vmax)
_plot_idxs_walks_to_step(delta_later.free_surf_walk_inds, _step=ps[0], _idxs=idxs, _ax=ax[1, 0])
# ax[1, 0].set_title('after {} steps'.format(ps[0]))


# fill in axis1
pyDeltaRCM.debug_tools.plot_domain(
    delta.eta, ax=ax[0, 1], grid=False)
_plot_idxs_walks_to_step(
    delta.free_surf_walk_inds, _step=ps[1], _idxs=idxs, _ax=ax[0, 1])
ax[0, 1].set_title('after {} steps'.format(ps[1]))
pyDeltaRCM.debug_tools.plot_domain(
    delta_later.eta, ax=ax[1, 1], grid=False, vmin=vmin, vmax=vmax)
_plot_idxs_walks_to_step(
    delta_later.free_surf_walk_inds, _step=ps[1], _idxs=idxs, _ax=ax[1, 1])
# ax[1, 1].set_title('after {} steps'.format(ps[1]))

# fill in axis2
pyDeltaRCM.debug_tools.plot_domain(
    delta.eta, ax=ax[0, 2], grid=False)
_plot_idxs_walks_to_step(
    delta.free_surf_walk_inds, _step=ps[2], _idxs=idxs, _ax=ax[0, 2])
ax[0, 2].set_title('after {} steps'.format(ps[2]))
pyDeltaRCM.debug_tools.plot_domain(
    delta_later.eta, ax=ax[1, 2], grid=False, vmin=vmin, vmax=vmax)
_plot_idxs_walks_to_step(
    delta_later.free_surf_walk_inds, _step=ps[2], _idxs=idxs, _ax=ax[1, 2])
# ax[1, 2].set_title('after {} steps'.format(ps[2]))


# fill in axis3
pyDeltaRCM.debug_tools.plot_domain(
    delta.eta, ax=ax[0, 3], grid=False)
_plot_idxs_walks_to_step(
    delta.free_surf_walk_inds, _step=ps[3], _idxs=idxs, _ax=ax[0, 3])
ax[0, 3].set_title('after {} steps'.format(ps[3]))
pyDeltaRCM.debug_tools.plot_domain(
    delta_later.eta, ax=ax[1, 3], grid=False, vmin=vmin, vmax=vmax)
_plot_idxs_walks_to_step(
    delta_later.free_surf_walk_inds, _step=ps[3], _idxs=idxs, _ax=ax[1, 3])
# ax[1, 3].set_title('after {} steps'.format(ps[3]))


plt.tight_layout()
if __name__ == '__main__':
    plt.savefig('run_water_iteration.png', transparent=True, dpi=300)
else:
    plt.show()
