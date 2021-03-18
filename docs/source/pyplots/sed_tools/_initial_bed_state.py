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

# set up axis
fig, ax = plt.subplots()

# fill in axis2
pyDeltaRCM.debug_tools.plot_domain(
    delta.eta, ax=ax, grid=False, cmap='cividis')
ax.set_title('intial bed elevation (m)')


plt.tight_layout()
plt.show()
