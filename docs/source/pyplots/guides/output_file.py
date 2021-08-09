import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import pyDeltaRCM


# filter out the warning raised about no netcdf being found
warnings.filterwarnings("ignore", category=UserWarning)


n = 20
cm = matplotlib.cm.get_cmap('tab10')


# init delta model
with pyDeltaRCM.shared_tools._docs_temp_directory() as output_dir:

    delta = pyDeltaRCM.DeltaModel(
        out_dir=output_dir,
        resume_checkpoint='../../_resources/checkpoint')


_shp = delta.eta.shape

# set up axis
fig, ax = plt.subplots(1, 2, figsize=(6, 3))
vmin, vmax = delta.eta.min(), delta.eta.max()

# fill in axis
ax[0].imshow(
    delta.eta, vmin=vmin, vmax=vmax, cmap='cividis')
ax[0].set_xticks([])
ax[0].set_yticks([])
ax[0].set_title('bed elevation')

ax[1].imshow(
    delta.uw, cmap='plasma')
ax[1].set_xticks([])
ax[1].set_yticks([])
ax[0].set_title('velocity')

plt.tight_layout()
plt.show()
