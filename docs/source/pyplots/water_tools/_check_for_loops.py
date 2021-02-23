import matplotlib.pyplot as plt
import numpy as np
import matplotlib

import pyDeltaRCM
from pyDeltaRCM import water_tools
from pyDeltaRCM import shared_tools


n = 10
cm = matplotlib.cm.get_cmap('tab10')


# init delta model
delta = pyDeltaRCM.DeltaModel(
    '../../_resources/checkpoint.yaml',
    resume_checkpoint='../../_resources/deltaRCM_Output')
_shp = delta.eta.shape


# determine the index to plot at
pidx = 22


# run an interation from the checkpoint
delta.init_water_iteration()
delta.run_water_iteration()

for pidx in range(200):
    # here, we recreate some steps of each iteration, in order to set up the
    #   arrays needed to display the action of _check_for_loops
    #
    #   1. extract the water weights
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

    # copy inputs and then run the function to get new outputs
    new_inds0 = np.copy(new_inds)
    new_inds, looped = water_tools._check_for_loops(
        delta.free_surf_walk_inds[:, :pidx], new_inds0, pidx + 1, delta.L0,
        delta.CTR, delta.stage - delta.H_SL)

    looped = looped.astype(np.bool)
    neq = new_inds != new_inds0
    ds0 = np.copy(new_inds)
    whr_neq = np.where(neq)[0]

    # declare the idxs to use:
    # idxs = np.random.randint(low=0, high=delta._Np_water, size=10)

    # if np.any(looped):
    if whr_neq.size > 0:
        breakpoint()


# make a function to plot each point of interest as two points and an arrow
def _plot_a_point(i):
    x0, y0 = shared_tools.custom_unravel(new_inds0[whr_neq[i]])
    x, y = shared_tools.custom_unravel(new_inds[whr_neq[i]])
    delta.show_ind(x0, y0, '.', c=cm(i), ax=ax)
    delta.show_ind(x, y, '.', c=cm(i), ax=ax)
    ax.arrow(x0, y0, (x-x0), (y-y0), c=cm(i))


# make the figure
fig, ax = plt.subplots()
delta.show_attribute('eta', ax=ax, grid=False)
for npt in range(n):
    breakpoint()
    _plot_a_point(npt)
plt.show()
