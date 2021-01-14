import matplotlib.pyplot as plt
import numpy as np
import pyDeltaRCM
import matplotlib

n = 10
cm = matplotlib.cm.get_cmap('tab10')

# init delta model
delta = pyDeltaRCM.DeltaModel()

# manually call only the necessary paths
delta.init_water_iteration()
delta.run_water_iteration()

# define a function to fill in the walks of given idx
_shp = delta.eta.shape
def _plot_idxs_walks_to_step(_step, _idxs, _ax):
	# _step = delta.free_surf_walk_indices.shape[1]
	for i in range(len(_idxs)):
	    _hld = np.zeros((_step, 2))
	    iidx = _idxs[i]
	    for j in range(_step):
	        walk = delta.free_surf_walk_indices[iidx, j]
	        _hld[j, 1], _hld[j, 0] = pyDeltaRCM.shared_tools.custom_unravel(walk, _shp)
	    _hld[_hld[:, 0] == 0, :] = np.nan
	    _ax.plot(_hld[:, 0], _hld[:, 1], color=cm(i))
	    idx_last = np.where(~np.isnan(_hld[:, 0]))[0][-1]
	    _ax.plot(_hld[idx_last, 0], _hld[idx_last, 1], marker='o', ms=3, color=cm(i))

# declare the idxs to use:
idxs = np.random.randint(low=0, high=delta._Np_water, size=n)

# set up axis
fig, ax = plt.subplots(1, 4, sharex=True, sharey=True, figsize=(12, 3))

# fill in axis0
pyDeltaRCM.debug_tools.plot_domain(delta.eta, ax=ax[0], grid=False)
_plot_idxs_walks_to_step(_step=2, _idxs=idxs, _ax=ax[0])
ax[0].set_title('after 2 steps')

# fill in axis1
pyDeltaRCM.debug_tools.plot_domain(delta.eta, ax=ax[1], grid=False)
_plot_idxs_walks_to_step(_step=5, _idxs=idxs, _ax=ax[1])
ax[1].set_title('after 5 steps')

# fill in axis2
pyDeltaRCM.debug_tools.plot_domain(delta.eta, ax=ax[2], grid=False)
_plot_idxs_walks_to_step(_step=10, _idxs=idxs, _ax=ax[2])
ax[2].set_title('after 10 steps')

# fill in axis3
pyDeltaRCM.debug_tools.plot_domain(delta.eta, ax=ax[3], grid=False)
_plot_idxs_walks_to_step(_step=15, _idxs=idxs, _ax=ax[3])
ax[3].set_title('after 15 steps')

plt.tight_layout()
plt.show()