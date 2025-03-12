import warnings

import matplotlib
import matplotlib.pyplot as plt

import pyDeltaRCM

# this creates a large plot showing a bunch of the basin parameters

# filter out the warning raised about no netcdf being found
warnings.filterwarnings("ignore", category=UserWarning)


n = 1
cm = matplotlib.colormaps.get_cmap("tab10")

param_dict = {"timesteps": 0}
# _matrix = {"Length": [2500, 5000, 10000]}
# param_dict["matrix"] = _matrix
LENGTH = 2000
WIDTH = 4000
param_dict["Length"] = LENGTH
param_dict["Width"] = WIDTH

# init delta model
with pyDeltaRCM.shared_tools._docs_temp_directory() as output_dir:
    param_dict["out_dir"] = output_dir
    delta = pyDeltaRCM.DeltaModel(**param_dict)

fig, ax = plt.subplots(
    figsize=(6, 3.5),
    subplot_kw=dict(aspect="equal"),
)


ax.imshow(delta.eta, interpolation="none", extent=[0, delta.Width, delta.Length, 0])

# computational edge
ax.contour(
    delta.X[1:, 1:],
    delta.Y[1:, 1:],
    delta.cell_type,
    levels=[-1],
    colors="white",
    linewidths=[1],
    linestyles=["-"],
)
ax.annotate(
    "computational\ndomain edge:\n  Length, Width",
    (0.2 * WIDTH, 0.8 * LENGTH),
    (0.25 * WIDTH + delta.dx, 0.75 * LENGTH - delta.dx),
    fontsize=8,
    color="white",
    arrowprops=dict(arrowstyle="-", color="white"),
)

# inlet
inlet_x = (delta.CTR + 1) * delta.dx
inlet_y = (delta.L0 - 2) * delta.dx
ax.annotate(
    "inlet geometry:\n  h0, N0_meters, L0_meters",
    (inlet_x, inlet_y),  # point
    (inlet_x - (10 * delta.dx), inlet_y + (10 * delta.dx)),  # label
    fontsize=8,
    color="white",
    textcoords="data",
    arrowprops=dict(arrowstyle="-", color="white"),
)

# basin depth
ax.annotate(
    "basin depth:\n  hb",
    (0.75 * WIDTH, 0.45 * LENGTH),
    (0.65 * WIDTH, 0.6 * LENGTH),
    fontsize=8,
    color="white",
    arrowprops=dict(arrowstyle="-", color="white"),
)

ax.set_ylabel("Length", fontsize=8)

ax.tick_params(labelsize=7)
ax.set_xlabel("Width", fontsize=8)

plt.tight_layout()
plt.show()
