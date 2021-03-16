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

_eta_before = np.copy(delta.eta)

delta.route_all_sand_parcels()

_eta_after = np.copy(delta.eta)


# set up axis
fig, ax = plt.subplots()

# fill in axis2
_diff = _eta_after - _eta_before
pyDeltaRCM.debug_tools.plot_domain(
    _diff, ax=ax, grid=False, cmap='RdBu',
    vmin=-0.1, vmax=0.1)
ax.set_title('bed elevation change (m) \n due to sand parcels')


plt.tight_layout()
plt.show()
