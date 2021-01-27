******************
Output File Format
******************

Model outputs are saved using the `netCDF4 <http://unidata.github.io/netcdf4-python>`_ file format.

Gridded Variables
-----------------
In any given run, the saving parameters "save_<var>_grids" control whether or
not that 2-D grid variable (e.g. velocity) is saved to the netCDF4 file.
In the netCDF4 file, a 3-D array with the dimensions
*time* x *length* x *width* is created for each 2-D grid variable that is set
to be saved. The appropriate units for these variables are stored as well,
such as "meters per second" for the *velocity* grid.

Grid Coordinates
----------------
To save the model information associated with the domain itself, variables
associated with the grid are saved as well. These are the meshed 2-D grids
corresponding to the distance of each cell from the boundary in the "Width"
dimension of the domain, *x* in meters. As well as the distance away from the
boundary of each cell in the "Length" dimension, as *y* in meters. Similarly, a
*time* variable is stored which is a 1-D array (vector) holding the model time
values in seconds, associated with each set of saved output data.

Model Metadata
--------------
In addition to the grid coordinates, model metadata is saved as a group of
1-D arrays (vectors) and 0-D arrays (floats and integers). The values that are
saved as metadata are the following:

- Length of the land surface: `L0`
- Width of the inlet channel: `N0`
- Center of the domain: `CTR`
- Length of cell faces: `dx`
- Depth of inlet channel: `h0`
- Sea level: `H_SL`
- Bedload fraction: `f_bedload`
- Sediment concentration: `C0_percent`
- Characteristic Velocity: `u0`

Working with Model Outputs
--------------------------
The resulting netCDF4 output file can be read using any netCDF4-compatible
library. These libraries range from the
`netCDF4 Python package <https://github.com/Unidata/netcdf4-python>`_ itself,
to higher-level libraries such as
`xarray <https://github.com/pydata/xarray>`_. For deltas, and specifically
*pyDeltaRCM*, there is also a package under development called
`DeltaMetrics <https://github.com/DeltaRCM/DeltaMetrics>`_,
that is being designed to help post-process and analyze *pyDeltaRCM* outputs.
