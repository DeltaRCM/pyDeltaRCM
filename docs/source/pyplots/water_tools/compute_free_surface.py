import matplotlib.pyplot as plt
import numpy as np
import pyDeltaRCM
import matplotlib


n = 1
cm = matplotlib.cm.get_cmap('tab10')


# init delta model
delta = pyDeltaRCM.DeltaModel(
    '../../_resources/checkpoint.yaml',
    resume_checkpoint='../../_resources/deltaRCM_Output')
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

# set up axis
fig, ax = plt.subplots(3, 2, sharex=True, sharey=True, figsize=(9, 7))
# fig = plt.figure(constrained_layout=True)
# gs = fig.add_gridspec(2, 2)


# fill in axis
pyDeltaRCM.debug_tools.plot_domain(
    delta.eta, ax=ax[0, 0], grid=False, cmap='cividis',
    label='bed elevation (m)')
pyDeltaRCM.debug_tools.plot_domain(
    delta.eta, ax=ax[0, 1], grid=False, cmap='cividis',
    label='bed elevation (m)')
delta.show_line(delta.free_surf_walk_inds[::10, :], 'r-',
                ax=ax[0, 1], alpha=0.1, nozeros=True)


pyDeltaRCM.debug_tools.plot_domain(
    delta.sfc_visit, ax=ax[1, 0], grid=False, cmap='Reds',
    label='sfc_visit (-)')
pyDeltaRCM.debug_tools.plot_domain(
    delta.sfc_sum, ax=ax[1, 1], grid=False, cmap='Blues',
    label='sfc_sum (m)')


pyDeltaRCM.debug_tools.plot_domain(
    Hnew, ax=ax[2, 0], grid=False,
    label='H_new (m)')
pyDeltaRCM.debug_tools.plot_domain(
    stage_new, ax=ax[2, 1], grid=False,
    label='stage (m)')


plt.tight_layout()

if __name__ == '__main__':
    plt.savefig('compute_free_surface.png', transparent=True, dpi=300)
else:
    plt.show()
