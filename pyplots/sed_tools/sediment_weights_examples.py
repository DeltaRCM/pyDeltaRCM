import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import pyDeltaRCM


# filter out the warning raised about no netcdf being found
warnings.filterwarnings("ignore", category=UserWarning)


n = 10
cm = matplotlib.cm.get_cmap('tab10')


class SedimentWeightingCalculator(pyDeltaRCM.DeltaModel):

    def __init__(self, input_file=None, **kwargs):

        # inherit from base model
        super().__init__(input_file, **kwargs)

    def get_sediment_weight_array(self):
        sand_weights = np.zeros((self.L, self.W, 9))
        mud_weights = np.zeros((self.L, self.W, 9))

        for i in range(self.L):
            for j in range(self.W):
                stage_nbrs = self.pad_stage[i:i + 3, j:j + 3]
                depth_nbrs = self.pad_depth[i:i + 3, j:j + 3]
                ct_nbrs = self.pad_cell_type[i:i + 3, j:j + 3]

                _, weight_int = pyDeltaRCM.shared_tools.get_weight_sfc_int(
                    self.stage[i, j], stage_nbrs.ravel(),
                    self.qx[i, j], self.qy[i, j], self.ivec_flat,
                    self.jvec_flat, self.distances_flat)

                try:
                    sand_weights[i, j] = pyDeltaRCM.sed_tools._get_weight_at_cell_sediment(
                        (i, j), weight_int, depth_nbrs.ravel(), ct_nbrs.ravel(),
                        self.dry_depth, self.theta_sand, self.distances_flat)

                    mud_weights[i, j] = pyDeltaRCM.sed_tools._get_weight_at_cell_sediment(
                        (i, j), weight_int, depth_nbrs.ravel(), ct_nbrs.ravel(),
                        self.dry_depth, self.theta_mud, self.distances_flat)
                except Exception:
                    sand_weights[i, j] = np.nan
                    mud_weights[i, j] = np.nan

        self.sand_weights = sand_weights
        self.mud_weights = mud_weights


# init delta model
with pyDeltaRCM.shared_tools._docs_temp_directory() as output_dir:
    delta = SedimentWeightingCalculator(
        out_dir=output_dir,
        resume_checkpoint='../../_resources/checkpoint')

delta.init_water_iteration()
delta.run_water_iteration()
delta.compute_free_surface()
delta.finalize_free_surface()

delta.get_sediment_weight_array()

_shp = delta.eta.shape


# manually call only the necessary paths
delta.init_water_iteration()
delta.run_water_iteration()


NPLOT = 5
hdr = NPLOT // 2

# fig, ax = plt.subplots()
fig = plt.figure(constrained_layout=False, figsize=(6, 9))
gs = fig.add_gridspec(
    nrows=NPLOT+hdr, ncols=5,
    left=0.05, right=0.95, top=0.95, bottom=0.05,
    wspace=0.01, hspace=0.2)

hdr_ax = fig.add_subplot(gs[:hdr, :])
pyDeltaRCM.debug_tools.plot_domain(
    delta.eta, ax=hdr_ax, grid=False, cmap='cividis', vmin=-3, vmax=0.5)
hdr_ax.set_xticks([])
hdr_ax.set_yticks([])
hdr_ax.set_xlim(0, delta.W // 2)
hdr_ax.set_ylim(delta.L // 2, 0)


def plot_pxpy_weights(pxpy, i):

    px, py = pxpy

    hdr_ax.text(py, px, str(i-hdr),
                fontsize=9, color='white',
                ha='center', va='center')

    __ax_list = []

    def _make_single_axis(i, j, values, **kwargs):
        __ax = fig.add_subplot(gs[i, j])
        __ax.set_xticks([])
        __ax.set_yticks([])
        __ax.imshow(values, **kwargs)
        for _i in range(3):
            for _j in range(3):
                __ax.text(
                    _j, _i, str(np.round(values[_i, _j], 2)),
                    ha='center', va='center', fontsize=8)
        return __ax

    # grab the data from the fields and put it into axis
    _eta_data = delta.eta[px-1:px+2, py-1:py+2]
    _eta_ax = _make_single_axis(
        i, 0, _eta_data, cmap='cividis', vmin=-3, vmax=0.5)
    __ax_list.append(_eta_ax)

    _dpth_data = delta.depth[px-1:px+2, py-1:py+2]
    _dpth_ax = _make_single_axis(
        i, 1, _dpth_data, cmap='Blues', vmin=0, vmax=5)
    __ax_list.append(_dpth_ax)

    _vel_data = delta.uw[px-1:px+2, py-1:py+2]
    _vel_ax = _make_single_axis(
        i, 2, _vel_data, cmap='plasma', vmin=0, vmax=1.5)
    __ax_list.append(_vel_ax)

    _sandwght_data = delta.sand_weights[px, py, :].reshape((3, 3))
    _sandwght_ax = _make_single_axis(
        i, 3, _sandwght_data, cmap='YlGn', vmin=0, vmax=1)
    __ax_list.append(_sandwght_ax)

    _mudwght_data = delta.mud_weights[px, py, :].reshape((3, 3))
    _mudwght_ax = _make_single_axis(
        i, 4, _mudwght_data, cmap='YlGn', vmin=0, vmax=1)
    __ax_list.append(_mudwght_ax)

    return __ax_list


# define the example location set
_ex_set = [(10, 94), (20, 78), (38, 72), (22, 87), (47, 43)]


# plot the points
ax0 = plot_pxpy_weights(_ex_set[0], hdr+0)
ax1 = plot_pxpy_weights(_ex_set[1], hdr+1)
ax2 = plot_pxpy_weights(_ex_set[2], hdr+2)
ax3 = plot_pxpy_weights(_ex_set[3], hdr+3)
ax4 = plot_pxpy_weights(_ex_set[4], hdr+4)


# labels
for a, axx in enumerate([ax0, ax1, ax2, ax3, ax4]):
    axx[0].set_ylabel(str(a), rotation=0, labelpad=10,
                      ha='center', va='center')

albls = ['bed elevation\n(eta)', 'depth',
         'velocity\n(uw)', 'sand weights', 'mud weights']
for a, (axx, albl) in enumerate(zip(ax4, albls)):
    axx.set_xlabel(albl)

# show
plt.show()
