# create a figure
fig, ax = plt.subplots()

# set the "simulation" range
_times = np.linspace(0, 1e9, num=100)
fb_mdl_muddy = np.zeros_like(_times)
fb_mdl_sandy = np.zeros_like(_times)

# loop through time, change the model time and grab f_bedload values
for i, _time in enumerate(_times):
    # change the model time directly
    mdl_muddy._time = _time  # you should never do this
    mdl_sandy._time = _time  # you should never do this

    # run the hooked method
    mdl_muddy.hook_run_one_timestep()
    mdl_sandy.hook_run_one_timestep()

    # grab the state of the `f_bedload` parameter
    fb_mdl_muddy[i] = mdl_muddy.f_bedload  # get the value
    fb_mdl_sandy[i] = mdl_sandy.f_bedload  # get the value

# add it to the plot
ax.plot(_times, fb_mdl_muddy, '-', c='saddlebrown', lw=2, label='muddy')
ax.plot(_times, fb_mdl_sandy, '--', c='goldenrod', lw=2, label='sandy')
ax.legend()

# clean up
ax.set_ylim(0, 1)
ax.set_ylabel('f_bedload')
ax.set_xlabel('model time (s)')

plt.show()