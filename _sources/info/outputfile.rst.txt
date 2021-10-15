=================
Model Output File
=================

If configured to save any output data, model outputs are saved using the `netCDF4 <http://unidata.github.io/netcdf4-python/>`_ file format.


Gridded Variables
=================

In any given run, the saving parameters "save_<var>_grids" control whether or
not that 2-D grid variable (e.g. velocity) is saved to the netCDF4 file. In
the netCDF4 file, a 3-D array with the dimensions `time` :math:`\times`
`x` :math:`\times` `y` is created for each 2-D grid variable that is set to
be saved. Note that `x` is the *downstream* coordinate, rather than the
Cartesian `x` when displaying the grid. The appropriate units for all
variables are stored: for example "meters per second" for the *velocity*
grid.

.. note::
   
   The format of the output netCDF file coordinate changed in `v2.1.0`. The
   old format is documented
   in :attr:`~pyDeltaRCM.model.DeltaModel.legacy_netcdf`, and that input
   parameter `legacy_netcdf` can be used to create on output netcdf file with
   the old coordinate configuration.


Grid Coordinates
================

Grid coordinates are specified in the variables `time`, `x`, and `y` in the output netCDF4 file.
These arrays are 1D arrays, which specify the location of each cell in the domain in *dimensional* coordinates (e.g., meters).
In the downstream direction,  the distance of each cell from the inlet boundary is specified in `x` in meters.
Similarly, the cross-domain distance is specified in `y` in meters.
Lastly, the `time` variable is stored as a 1D array with model `time` in seconds.


Model Metadata
==============

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
- If subsidence is enabled:
  - Subsidence start time: `start_subsidence`
  - Subsidence rate: `sigma`


Working with Model Outputs
==========================

The resulting netCDF4 output file can be read using any netCDF4-compatible
library. These libraries range from the
`netCDF4 Python package <https://github.com/Unidata/netcdf4-python>`_ itself,
to higher-level libraries such as
`xarray <https://github.com/pydata/xarray>`_. For deltas, and specifically
*pyDeltaRCM*, there is also a package under development called
`DeltaMetrics <https://github.com/DeltaRCM/DeltaMetrics>`_,
that is being designed to help post-process and analyze *pyDeltaRCM* outputs.


Here, we show how to read the output NetCDF file with Python package ``netCDF4``.

.. code::

   import netCDF4 as nc

   data = nc.Dataset('pyDeltaRCM_output.nc')  # the output file path!

This `data` object is a `Dataset` object that can be sliced the same was as a `numpy` array.
For example, we can slice the final bed elevation and velocity of a model run:

.. code::

   final_bed_elevation = data['eta'][-1, :, :]
   final_velocity = data['velocity'][-1, :, :]

These slices look like this, if we were to plot them.

.. plot:: guides/output_file.py
