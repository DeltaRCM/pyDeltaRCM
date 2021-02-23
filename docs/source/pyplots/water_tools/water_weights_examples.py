import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from mpl_toolkits import mplot3d

import pyDeltaRCM
from pyDeltaRCM import water_tools
from pyDeltaRCM import shared_tools

_delta = pyDeltaRCM.DeltaModel()

ivec_flat = _delta.ivec_flat
jvec_flat = _delta.jvec_flat
distances_flat = _delta.distances_flat

ct_nbrs = np.ones((3, 3))
theta_water = 1
gamma = 0.05
dry_depth = 0.1
ind = (100, 100)


def _get_back_weights(eta_nbrs, depth_nbrs, stage_nbrs, _qx, _qy):

    _eta_nbrs = eta_nbrs.ravel()
    _depth_nbrs = depth_nbrs.ravel()
    _stage_nbrs = stage_nbrs.ravel()
    _stage = _stage_nbrs[4]

    weight_sfc, weight_int = shared_tools.get_weight_sfc_int(
        _stage, _stage_nbrs,
        _qx, _qy, ivec_flat, jvec_flat,
        distances_flat)

    water_weights = water_tools._get_weight_at_cell_water(
        ind, weight_sfc, weight_int,
        _depth_nbrs, ct_nbrs.ravel(),
        dry_depth, gamma, theta_water)

    return water_weights


# stage_nbrs = pad_stage[i - 1 + 1:i + 2 + 1, j - 1 + 1:j + 2 + 1]
# depth_nbrs = pad_depth[i - 1 + 1:i + 2 + 1, j - 1 + 1:j + 2 + 1]
# ct_nbrs = pad_cell_type[i - 1 + 1:i + 2 + 1, j - 1 + 1:j + 2 + 1]

eta_nbrs = np.array([[2,    1.25, 1.5],
                     [1.25, 1.0,  1.5],
                     [1.25, 1.0, 0.75]])
depth_nbrs = np.array([[3, 3.75, 2.5],
                       [2.5, 2,  2.],
                       [2.5, 2, 2]])
stage_nbrs = eta_nbrs + depth_nbrs

qx = 1
qy = 0

# wts = _get_back_weights(eta_nbrs, depth_nbrs, stage_nbrs, qx, qy)


def _plot_array(_arr, _ax, colspec, **kwargs):
    if colspec == 'bed':
        cm = matplotlib.cm.get_cmap('viridis', 10)
    elif colspec == 'stage':
        cm = matplotlib.cm.get_cmap('viridis', 10)

    for i in range(_arr.shape[0]):
        for j in range(_arr.shape[0]):
            # create x,y
            xx, yy = np.meshgrid(np.arange(i, i+2), np.arange(j, j+2))
            # _arr[i:i+2, j:j+2]
            _p_arr = np.ones((2, 2)) * _arr[i, j]
            x = _arr[i, j]
            v = (10-0)*((x-np.min(_arr))/(np.max(_arr)-np.min(_arr)))+0  # rescale value in 1-100
            _ax.plot_surface(xx, yy, _p_arr, color=cm(int(v)), **kwargs)


# fig, ax = plt.subplots()
fig = plt.figure()
ax = plt.axes(projection='3d')
# ax.imshow(wts.reshape(3, 3))
# _plot_array(eta_nbrs, _ax=ax, colspec='bed')

_xx, _yy = np.meshgrid(np.arange(3), np.arange(3))
x, y = _xx.ravel(), _yy.ravel()

top = eta_nbrs.ravel()
bottom = np.zeros_like(top)
width = depth = 1
ax.bar3d(x, y, bottom, width, depth, top, shade=True)


# _plot_array(stage_nbrs, _ax=ax, colspec='stage')
_xx, _yy = np.meshgrid(np.arange(4), np.arange(4))
ax.plot_surface(_xx, _yy, stage_nbrs)
ax.set_zlim(0, 20)
plt.show()
