from matplotlib.colors import Normalize

fig, ax = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(10, 4))
norm = Normalize(vmin=3, vmax=100)

for i, job in enumerate(pp.job_list):
    # first convert the field to a rate
    subsidence_rate_field = (job.deltamodel.sigma / job.deltamodel.dt)

    # now convert to mm/yr
    subsidence_rate_field = (subsidence_rate_field * 1000 *
        pyDeltaRCM.shared_tools._scale_factor(If=0.019, units='years'))

    # and display
    im = ax.flat[i].imshow(subsidence_rate_field, norm=norm)

fig.colorbar(im, ax=ax.ravel().tolist())
plt.show()