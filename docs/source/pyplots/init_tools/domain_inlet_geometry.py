import warnings

import matplotlib
import matplotlib.pyplot as plt

import pyDeltaRCM


# filter out the warning raised about no netcdf being found
warnings.filterwarnings("ignore", category=UserWarning)


n = 1
cm = matplotlib.colormaps.get_cmap("tab10")

param_dict = {"timesteps": 0}
_matrix = {"L0_meters": [200, 500, 1000], "N0_meters": [200, 500, 1000]}
param_dict["matrix"] = _matrix

# init delta models with preprocessor
with pyDeltaRCM.shared_tools._docs_temp_directory() as output_dir:
    param_dict.update({"out_dir": output_dir})

    pp = pyDeltaRCM.preprocessor.Preprocessor(param_dict)

    pp.run_jobs()


fig, ax = plt.subplots(
    3, 3, figsize=(9, 5), subplot_kw=dict(aspect="equal"), sharex=True, sharey=True
)

ax = ax.ravel()
for i in range(9):
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
        f"L0_meters: {pp.job_list[i].deltamodel.L0_meters}\nN0_meters: {pp.job_list[i].deltamodel.N0_meters}",
        ha="left",
        va="bottom",
        color="white",
        fontsize=8,
        transform=ax[i].transAxes,
    )

for i, axi in enumerate(ax.ravel()):
    axi.tick_params(labelsize=7)
    if i % 3 == 0:
        axi.set_ylabel("Length", fontsize=8)
    if i > 6:
        axi.set_xlabel("Width", fontsize=8)

plt.tight_layout()
plt.show()
