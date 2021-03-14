import warnings

import matplotlib
import matplotlib.pyplot as plt

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


fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(9, 3))

# fill in axis
pyDeltaRCM.debug_tools.plot_domain(
    delta.eta, ax=ax[0], grid=False, cmap='cividis',
    label='bed elevation (m)')
pyDeltaRCM.debug_tools.plot_domain(
    delta.eta, ax=ax[1], grid=False, cmap='cividis',
    label='bed elevation (m)')
delta.show_line(delta.free_surf_walk_inds[::10, :].T, 'k-',
                ax=ax[1], alpha=0.1,
                multiline=True, nozeros=True)

fig.show()
