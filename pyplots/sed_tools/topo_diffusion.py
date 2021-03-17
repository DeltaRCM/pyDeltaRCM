import warnings

import numpy as np
import matplotlib.pyplot as plt

import pyDeltaRCM

# filter out the warning raised about no netcdf being found
warnings.filterwarnings("ignore", category=UserWarning)


# init delta model
with pyDeltaRCM.shared_tools._docs_temp_directory() as output_dir:
    delta = pyDeltaRCM.DeltaModel(
        out_dir=output_dir,
        resume_checkpoint='../../_resources/checkpoint')

_shp = delta.eta.shape

delta.route_all_sand_parcels()

_eta_before = np.copy(delta.eta)
delta.topo_diffusion()
_eta_after = np.copy(delta.eta)

# set up axis
fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(12, 4))

# fill in axis0
pyDeltaRCM.debug_tools.plot_domain(
    _eta_before, ax=ax[0], grid=False, cmap='cividis', vmin=-6)
ax[0].set_title('bed elevation before \n topographic diffusion')

# fill in axis1
pyDeltaRCM.debug_tools.plot_domain(
    _eta_after, ax=ax[1], grid=False, cmap='cividis', vmin=-6)
ax[1].set_title('bed elevation before \n topographic diffusion')


# fill in axis2
_diff = _eta_after - _eta_before
pyDeltaRCM.debug_tools.plot_domain(
    _diff, ax=ax[2], grid=False, cmap='RdBu',
    vmin=-0.1, vmax=0.1)
ax[2].set_title('difference (after - before)')


plt.tight_layout()
plt.show()
