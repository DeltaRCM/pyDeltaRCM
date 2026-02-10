import warnings

import matplotlib
import matplotlib.pyplot as plt

import pyDeltaRCM


# filter out the warning raised about no netcdf being found
warnings.filterwarnings("ignore", category=UserWarning)


n = 1
cm = matplotlib.colormaps.get_cmap("tab10")

param_dict = {"timesteps": 0}
_matrix = {"h0": [2, 5], "hb": [2, 5]}
param_dict["matrix"] = _matrix
param_dict["L0_meters"] = 500
param_dict["N0_meters"] = 500

# init delta models with preprocessor
with pyDeltaRCM.shared_tools._docs_temp_directory() as output_dir:
    param_dict.update({"out_dir": output_dir})

    pp = pyDeltaRCM.preprocessor.Preprocessor(param_dict)

    pp.run_jobs()


fig, ax = plt.subplots(
    2, 2, figsize=(6, 4), subplot_kw=dict(aspect="equal"), sharex=True, sharey=True
)

ax = ax.ravel()
for i in range(4):
    ax[i].imshow(
        pp.job_list[i].deltamodel.eta,
        interpolation="none",
        extent=[
            0,
            pp.job_list[i].deltamodel.Width,
            pp.job_list[i].deltamodel.Length,
            0,
        ],
    )
    ax[i].text(
        0.05,
        0.05,
        f"h0: {pp.job_list[i].deltamodel.h0}\nhb: {pp.job_list[i].deltamodel.hb}",
        ha="left",
        va="bottom",
        color="white",
        fontsize=8,
        transform=ax[i].transAxes,
    )

for i, axi in enumerate(ax.ravel()):
    axi.tick_params(labelsize=7)
    if i % 2 == 0:
        axi.set_ylabel("Length", fontsize=8)
    if i > 2:
        axi.set_xlabel("Width", fontsize=8)

plt.tight_layout()
plt.show()
