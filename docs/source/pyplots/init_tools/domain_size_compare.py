import warnings

import matplotlib
import matplotlib.pyplot as plt

import pyDeltaRCM


# filter out the warning raised about no netcdf being found
warnings.filterwarnings("ignore", category=UserWarning)


n = 1
cm = matplotlib.cm.get_cmap('tab10')

param_dict = {'timesteps': 0}
_matrix = {'Length': [2500, 5000, 10000]}
param_dict['matrix'] = _matrix

# init delta models with preprocessor
with pyDeltaRCM.shared_tools._docs_temp_directory() as output_dir:
    param_dict.update({'out_dir': output_dir})
    
    pp = pyDeltaRCM.preprocessor.Preprocessor(
        param_dict)

    pp.run_jobs()


fig, ax = plt.subplots(
    1, 3, figsize=(9, 3.5),
    subplot_kw=dict(aspect='equal'),
    gridspec_kw=dict(width_ratios=[1, 1, 1]))

for i in range(3):
    ax[i].imshow(
        pp.job_list[i].deltamodel.eta,
        interpolation='none',
        extent=[0, pp.job_list[i].deltamodel.Width,
                pp.job_list[i].deltamodel.Length, 0])
    ax[i].contour(
        pp.job_list[i].deltamodel.X[1:, 1:],
        pp.job_list[i].deltamodel.Y[1:, 1:],
        pp.job_list[i].deltamodel.cell_type,
        levels=[-1], colors='white', linewidths=[1], linestyles=['-'])

ax[2].annotate(
    'computational\ndomain edge',
    (2000, 4000), (200, 6400),
    fontsize=8, color='white',
    arrowprops=dict(arrowstyle='-', color='white'))

ax[0].set_ylabel('Length', fontsize=8)

for axi in ax.ravel():
    axi.tick_params(labelsize=7)
    axi.set_xlabel('Width', fontsize=8)

plt.tight_layout()
plt.show()
