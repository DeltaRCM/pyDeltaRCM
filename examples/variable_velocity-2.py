end_time = 86400 * 100
_time_array, _velocity_array = create_velocity_array(
            end_time)

# make a plot of the boundary condition
fig, ax = plt.subplots()
ax.plot(_time_array, _velocity_array)
ax.set_xlabel('time (seconds)')
ax.set_ylabel('inlet velocity (meters/second)')
plt.show()