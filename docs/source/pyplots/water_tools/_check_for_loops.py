import warnings

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import pyDeltaRCM
from pyDeltaRCM import water_tools


# filter out the warning raised about no netcdf being found
warnings.filterwarnings("ignore", category=UserWarning)


n = 10
cm = matplotlib.cm.get_cmap('tab10')


# init delta model
with pyDeltaRCM.shared_tools._docs_temp_directory() as output_dir:
    delta = pyDeltaRCM.DeltaModel(
        out_dir=output_dir,
        resume_checkpoint='../../_resources/checkpoint')
_shp = delta.eta.shape


# determine the index to plot at
pidx = 40
Npts = 5
jidx = np.random.randint(0, delta._Np_water, Npts)

# run an interation from the checkpoint
delta.init_water_iteration()
delta.run_water_iteration()

# here, we recreate some steps of each iteration, in order to set up the
#   arrays needed to display the action of _check_for_loops
#
#   1. extract the water walks
current_inds = delta.free_surf_walk_inds[:, pidx-1]
#
#   2. use water weights and random pick to determine d8 direction
water_weights_flat = delta.water_weights.reshape(-1, 9)
new_direction = water_tools._choose_next_directions(
    current_inds, water_weights_flat)
new_direction = new_direction.astype(np.int)
#
#   3. use the new directions for each parcel to determine the new ind for
#   each parcel
new_inds = water_tools._calculate_new_inds(
                current_inds,
                new_direction,
                delta.ravel_walk_flat)
new_inds0 = np.copy(new_inds)

# choose N indices randomly to jump
new_inds[jidx] = delta.free_surf_walk_inds[jidx, pidx-2]

# copy inputs and then run the function to get new outputs
new_inds, looped = water_tools._check_for_loops(
    delta.free_surf_walk_inds[:, :pidx-1], new_inds, pidx + 1, delta.L0,
    delta.CTR, delta.stage - delta.H_SL)


# make the figure
fig, ax = plt.subplots(1, 2, figsize=(10, 4))

delta.show_attribute('eta', ax=ax[0], grid=False, cmap='cividis')
delta.show_attribute('eta', ax=ax[1], grid=False, cmap='cividis')


def _fill_out_an_axis(_ax, _sc=1):
    for j in range(len(jidx)):
        walk = delta.free_surf_walk_inds[jidx[j], :]
        walk = walk[:pidx-2]
        yend, xend = pyDeltaRCM.shared_tools.custom_unravel(
            walk[-1], _shp)
        ynew, xnew = pyDeltaRCM.shared_tools.custom_unravel(
            new_inds[jidx[j]], _shp)
        pyDeltaRCM.debug_tools.plot_line(
            walk, shape=_shp, color=cm(j),
            multiline=True, nozeros=True, lw=1.5*_sc, ax=_ax)
        _ax.plot(xend, yend,
                 marker='o', ms=3*_sc, color=cm(j))
        _ax.plot(xnew, ynew,
                 marker='o', ms=3*_sc, color=cm(j))

    # return the last set
    return yend, xend, ynew, xnew


# fill out both axes with the same info
_fill_out_an_axis(_ax=ax[0])
yend, xend, ynew, xnew = _fill_out_an_axis(_ax=ax[1], _sc=2)

# sub region of the original image
_f = int(delta.W / delta.L)  # ensure x-y scale same
_s = 10
x1, x2, y1, y2 = xend-(_s*_f), xnew+(_s*_f), yend-(_s), ynew+(_s)
ax[1].set_xlim(x1, x2)
ax[1].set_ylim(y2, y1)
ax[1].set_xticks([])
ax[1].set_yticks([])

_r = Rectangle((x1, y2), _s*_f*2, -_s*2, ec='k', fc='none')
ax[0].add_patch(_r)

plt.show()
