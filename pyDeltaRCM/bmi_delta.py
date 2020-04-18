#! /usr/bin/env python
"""Basic Model Interface implementation for pyDeltaRCM."""

import types

import numpy as np
from basic_modeling_interface import Bmi

from .deltaRCM_driver import pyDeltaRCM


class BmiDelta(Bmi):

    _name = 'pyDeltaRCM'

    _input_var_names = (
        'channel_exit_water_flow__speed',
        'channel_exit_water_x-section__depth',
        'channel_exit_water_x-section__width',
        'channel_exit_water_sediment~bedload__volume_fraction',
        'channel_exit_water_sediment~suspended__mass_concentration',
        'sea_water_surface__rate_change_elevation',
        'sea_water_surface__mean_elevation',
    )

    _output_var_names = (
        'sea_water_surface__elevation',
        'sea_water__depth',
        'sea_bottom_surface__elevation',
    )

    def __init__(self):
        """Create a BmiDelta model that is ready for initialization."""
        self._delta = None
        self._time = 0.
        self._values = {}
        self._var_units = {}
        self._grids = {}
        self._grid_type = {}

    def initialize(self, filename='deltaRCM.yaml'):
        """Initialize the model.

        Parameters
        ----------
        filename : str, optional
            Path to name of input file.
        """

        self._delta = pyDeltaRCM(filename)

        self._values = {
            'channel_exit_water_flow__speed': self._delta.u0,
            'channel_exit_water_x-section__width': self._delta.N0_meters,
            'channel_exit_water_x-section__depth': self._delta.h0,
            'sea_water_surface__mean_elevation': self._delta.H_SL,
            'sea_water_surface__rate_change_elevation': self._delta.SLR,
            'channel_exit_water_sediment~bedload__volume_fraction': self._delta.f_bedload,
            'channel_exit_water_sediment~suspended__mass_concentration': self._delta.C0_percent,
            'sea_water_surface__elevation': self._delta.stage,
            'sea_water__depth': self._delta.depth,
            'sea_bottom_surface__elevation': self._delta.eta,
        }

        self._var_units = {
            'channel_exit_water_flow__speed': 'm s-1',
            'channel_exit_water_x-section__width': 'm',
            'channel_exit_water_x-section__depth': 'm',
            'sea_water_surface__mean_elevation': 'm',
            'sea_water_surface__rate_change_elevation': 'm yr-1',
            'channel_exit_water_sediment~bedload__volume_fraction': 'fraction',
            'channel_exit_water_sediment~suspended__mass_concentration': 'm3 m-3',
            'sea_water_surface__elevation': 'm',
            'sea_water__depth': 'm',
            'sea_bottom_surface__elevation': 'm',
        }

        self._grids = {
            0: ['sea_water_surface__elevation'],
            1: ['sea_water__depth'],
            2: ['sea_bottom_surface__elevation'],
        }
        self._grid_type = {
            0: 'uniform_rectilinear_grid',
            1: 'uniform_rectilinear_grid',
            2: 'uniform_rectilinear_grid',
        }

    def update(self):
        """Advance model by one time step."""

        self._delta.update()
        self._time += self.get_time_step()

    def update_frac(self, time_frac):
        """Update model by a fraction of a time step.

        Parameters
        ----------
        time_frac : float
            Fraction fo a time step.
        """
        time_step = self.get_time_step()

        self._delta.time_step = time_frac * time_step
        self.update()

        self._delta.time_step = time_step

    def update_until(self, then):
        """Update model until a particular time.

        Parameters
        ----------
        then : float
            Time to run model until.
        """

        if self.get_current_time() != int(self.get_current_time()):

            remainder = self.get_current_time() - int(self.get_current_time())
            self.update_frac(remainder)

        n_steps = (then - self.get_current_time()) / self.get_time_step()

        for _ in range(int(n_steps)):
            self.update()

        remainder = n_steps - int(n_steps)
        if remainder > 0:
            self.update_frac(remainder)

    def finalize(self):
        """Finalize model."""
        self._delta.finalize()
        self._delta = None

    def get_var_type(self, var_name):
        """Data type of variable.

        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.

        Returns
        -------
        str
            Data type.
        """
        return str(self.get_value_ref(var_name).dtype)

    def get_var_units(self, var_name):
        """Get units of variable.

        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.

        Returns
        -------
        str
            Variable units.
        """
        return self._var_units[var_name]

    def get_var_nbytes(self, var_name):
        """Get units of variable.

        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.

        Returns
        -------
        int
            Size of data array in bytes.
        """
        return self.get_value_ref(var_name).nbytes

    def get_var_grid(self, var_name):
        """Grid id for a variable.

        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.

        Returns
        -------
        int
            Grid id.
        """
        for grid_id, var_name_list in list(self._grids.items()):
            if var_name in var_name_list:
                return grid_id

    def get_grid_rank(self, grid_id):
        """Rank of grid.

        Parameters
        ----------
        grid_id : int
            Identifier of a grid.

        Returns
        -------
        int
            Rank of grid.
        """
        return len(self.get_grid_shape(grid_id))

    def get_grid_size(self, grid_id):
        """Size of grid.

        Parameters
        ----------
        grid_id : int
            Identifier of a grid.

        Returns
        -------
        int
            Size of grid.
        """
        return np.prod(self.get_grid_shape(grid_id))

    def get_value_ref(self, var_name):
        """Reference to values.

        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.

        Returns
        -------
        array_like
            Value array.
        """
        return self._values[var_name]

    def get_value(self, var_name):
        """Copy of values.

        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.

        Returns
        -------
        array_like
            Copy of values.
        """
        return self.get_value_ref(var_name).copy()

    def get_value_at_indices(self, var_name, indices):
        """Get values at particular indices.

        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.
        indices : array_like
            Array of indices.

        Returns
        -------
        array_like
            Values at indices.
        """
        return self.get_value_ref(var_name).take(indices)

    def set_value(self, var_name, src):
        """Set model values.

        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.
        src : array_like
            Array of new values.
        """
        val = self.get_value_ref(var_name)
        val[:] = src

    def set_value_at_indices(self, var_name, src, indices):
        """Set model values at particular indices.

        Parameters
        ----------
        var_name : str
            Name of variable as CSDMS Standard Name.
        src : array_like
            Array of new values.
        indices : array_like
            Array of indices.
        """
        val = self.get_value_ref(var_name)
        val.flat[indices] = src

    def get_component_name(self):
        """Name of the component."""
        return self._name

    def get_input_var_names(self):
        """Get names of input variables."""
        return self._input_var_names

    def get_output_var_names(self):
        """Get names of output variables."""
        return self._output_var_names

    def get_grid_shape(self, grid_id):
        """Number of rows and columns of uniform rectilinear grid."""
        var_name = self._grids[grid_id][0]
        return self.get_value_ref(var_name).shape

    def get_grid_spacing(self, grid_id):
        """Spacing of rows and columns of uniform rectilinear grid."""
        return (self._delta.dx, self._delta.dx)

    def get_grid_origin(self, grid_id):
        """Origin of uniform rectilinear grid."""
        return (0., 0.)

    def get_grid_type(self, grid_id):
        """Type of grid."""
        return self._grid_type[grid_id]

    def get_start_time(self):
        """Start time of model."""
        return 0.

    def get_end_time(self):
        """End time of model."""
        return np.finfo('d').max

    def get_current_time(self):
        """Current time of model."""
        return self._time

    def get_time_step(self):
        """Time step of model."""
        return self._delta.time_step
