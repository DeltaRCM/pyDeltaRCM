import matplotlib.pyplot as plt
import numpy as np
import pyDeltaRCM
import matplotlib


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
delta.compute_free_surface()


pidx = 60

sfc_visit, sfc_sum = pyDeltaRCM.water_tools._accumulate_free_surface_walks(
    delta.free_surf_walk_inds, delta.free_surf_flag, delta.cell_type,
    delta.uw, delta.ux, delta.uy, delta.depth,
    delta._dx, delta._u0, delta.h0, delta._H_SL, delta._S0)


Hnew = np.full_like(sfc_visit, np.nan)
Hnew[sfc_visit > 0] = (sfc_sum[sfc_visit > 0] /
                       sfc_visit[sfc_visit > 0])


stage_new = delta.eta + delta.depth
stage_new[sfc_visit > 0] = (sfc_sum[sfc_visit > 0] /
                            sfc_visit[sfc_visit > 0])


fig, ax = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(9, 4))

pyDeltaRCM.debug_tools.plot_domain(
    delta.sfc_visit, ax=ax[0, 0], grid=False, cmap='Greys',
    label='sfc_visit (-)')
pyDeltaRCM.debug_tools.plot_domain(
    delta.sfc_sum, ax=ax[0, 1], grid=False, cmap='Blues',
    label='sfc_sum (m)')

pyDeltaRCM.debug_tools.plot_domain(
    Hnew, ax=ax[1, 0], grid=False,
    label='H_new (m)')
pyDeltaRCM.debug_tools.plot_domain(
    stage_new, ax=ax[1, 1], grid=False,
    label='stage (m)')

fig.show()
