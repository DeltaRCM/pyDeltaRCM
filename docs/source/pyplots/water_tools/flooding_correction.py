import warnings

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import pyDeltaRCM

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


# manually call only the necessary paths
delta.init_water_iteration()
delta.run_water_iteration()

# lines cobbled together from compute_free_surface and finalize_free_surface
delta.sfc_visit, delta.sfc_sum = pyDeltaRCM.water_tools._accumulate_free_surface_walks(
    delta.free_surf_walk_inds, delta.free_surf_flag, delta.cell_type,
    delta.uw, delta.ux, delta.uy, delta.depth,
    delta._dx, delta._u0, delta.h0, delta._H_SL, delta._S0)
Hnew = delta.eta + delta.depth
Hnew[delta.sfc_visit > 0] = (delta.sfc_sum[delta.sfc_visit > 0] /
                             delta.sfc_visit[delta.sfc_visit > 0])
Hnew[Hnew < delta._H_SL] = delta._H_SL
Hnew[Hnew < delta.eta] = delta.eta[Hnew < delta.eta]

# now run the smoothing function to get grids
Hsmth = pyDeltaRCM.water_tools._smooth_free_surface(
    Hnew, delta.cell_type, delta._Nsmooth, delta._Csmooth)

delta.stage = (((1 - delta._omega_sfc) * delta.stage) +
               (delta._omega_sfc * Hsmth))

before_flood = np.copy(delta.stage)
delta.flooding_correction()
after_flood = np.copy(delta.stage)

# set up axis
fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(12, 4))

# fill in axis0
pyDeltaRCM.debug_tools.plot_domain(
    before_flood, ax=ax[0], grid=False, cmap='viridis')
ax[0].set_title('stage before flooding correction')

# fill in axis1
pyDeltaRCM.debug_tools.plot_domain(
    after_flood, ax=ax[1], grid=False, cmap='viridis')
ax[1].set_title('stage after flooding correction')


# fill in axis2
_diff = after_flood - before_flood
pyDeltaRCM.debug_tools.plot_domain(
    _diff, ax=ax[2], grid=False, cmap='RdBu',
    vmin=-0.1, vmax=0.1)
ax[2].set_title('difference (after - before)')


plt.tight_layout()
plt.show()
