#! /usr/bin/env python
import yaml
import warnings

import numpy as np
import os
from basic_modeling_interface import Bmi

from .model import DeltaModel

"""Basic Model Interface implementation for pyDeltaRCM."""


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

    _input_vars = {
        'model_output__site_prefix': {'name': 'site_prefix',
            'type': 'str', 'default': ''},
        'model_output__case_prefix': {'name': 'case_prefix',
            'type': 'str', 'default': ''},
        'model_output__out_dir': {'name': 'out_dir',
            'type': 'str', 'default': 'deltaRCM_Output/'},
        'model_grid__length': {'name': 'Length',
            'type': 'float', 'default': 5000.},
        'model_grid__width': {'name': 'Width',
            'type': 'float', 'default': 10000.},
        'model_grid__cell_size': {'name': 'dx',
            'type': 'float', 'default': 100.},
        'land_surface__width': {'name': 'L0_meters',
            'type': 'float', 'default': 300.},
        'land_surface__slope': {'name': 'S0',
            'type': 'float', 'default': 0.00015},
        'model__max_iteration': {'name': 'itermax',
            'type': 'int', 'default': 1},
        'water__number_parcels': {'name': 'Np_water',
            'type': 'int', 'default': 1000},
        'channel__flow_velocity': {'name': 'u0',
            'type': 'float', 'default': 1.},
        'channel__width': {'name': 'N0_meters',
            'type': 'float', 'default': 300.},
        'channel__flow_depth': {'name': 'h0',
            'type': 'float', 'default': 5.},
        'sea_water_surface__mean_elevation': {'name': 'H_SL',
            'type': 'float', 'default': 0.},
        'sea_water_surface__rate_change_elevation': {'name': 'SLR',
            'type': 'float', 'default': 0.},
        'sediment__number_parcels': {'name': 'Np_sed',
            'type': 'int', 'default': 1000},
        'sediment__bedload_fraction': {'name': 'f_bedload',
            'type': 'float', 'default': 0.25},
        'sediment__influx_concentration': {'name': 'C0_percent',
            'type': 'float', 'default': 0.1},
        'model_output__opt_eta_figs': {'name': 'save_eta_figs',
            'type': 'bool', 'default': True},
        'model_output__opt_stage_figs': {'name': 'save_stage_figs',
            'type': 'bool', 'default': False},
        'model_output__opt_depth_figs': {'name': 'save_depth_figs',
            'type': 'bool', 'default': False},
        'model_output__opt_discharge_figs': {'name': 'save_discharge_figs',
            'type': 'bool', 'default': False},
        'model_output__opt_velocity_figs': {'name': 'save_velocity_figs',
            'type': 'bool', 'default': False},
        'model_output__opt_eta_grids': {'name': 'save_eta_grids',
            'type': 'bool', 'default': False},
        'model_output__opt_stage_grids': {'name': 'save_stage_grids',
            'type': 'bool', 'default': False},
        'model_output__opt_depth_grids': {'name': 'save_depth_grids',
            'type': 'bool', 'default': False},
        'model_output__opt_discharge_grids': {'name': 'save_discharge_grids',
            'type': 'bool', 'default': False},
        'model_output__opt_velocity_grids': {'name': 'save_velocity_grids',
            'type': 'bool', 'default': False},
        'model_output__opt_time_interval': {'name': 'save_dt',
            'type': 'int', 'default': 50},
        'coeff__surface_smoothing': {'name': 'Csmooth',
            'type': 'float', 'default': 0.9},
        'coeff__under_relaxation__water_surface': {'name': 'omega_sfc',
            'type': 'float', 'default': 0.1},
        'coeff__under_relaxation__water_flow': {'name': 'omega_flow',
            'type': 'float', 'default': 0.9},
        'coeff__iterations_smoothing_algorithm': {'name': 'Nsmooth',
            'type': 'int', 'default': 5},
        'coeff__depth_dependence__water': {'name': 'theta_water',
            'type': 'float', 'default': 1.0},
        'coeff__depth_dependence__sand': {'name': 'coeff_theta_sand',
            'type': 'float', 'default': 2.0},
        'coeff__depth_dependence__mud': {'name': 'coeff_theta_mud',
            'type': 'float', 'default': 1.0},
        'coeff__non_linear_exp_sed_flux_flow_velocity': {'name': 'beta',
            'type': 'int', 'default': 3},
        'coeff__sedimentation_lag': {'name': 'sed_lag',
            'type': 'float', 'default': 1.0},
        'coeff__velocity_deposition_mud': {'name': 'coeff_U_dep_mud',
            'type': 'float', 'default': 0.3},
        'coeff__velocity_erosion_mud': {'name': 'coeff_U_ero_mud',
            'type': 'float', 'default': 1.5},
        'coeff__velocity_erosion_sand': {'name': 'coeff_U_ero_sand',
            'type': 'float', 'default': 1.05},
        'coeff__topographic_diffusion': {'name': 'alpha',
            'type': 'float', 'default': 0.1},
        'basin__opt_subsidence': {'name': 'toggle_subsidence',
            'type': 'bool', 'default': False},
        'basin__maximum_subsidence_rate': {'name': 'sigma_max',
            'type': 'float', 'default': 0.000825},
        'basin__subsidence_start_timestep': {'name': 'start_subsidence',
            'type': 'float', 'default': 0},
        'basin__opt_stratigraphy': {'name': 'save_strata',
            'type': 'bool', 'default': False}
    }

    def __init__(self):
        """Create a BmiDelta model that is ready for initialization."""
        self._delta = None
        self._values = {}
        self._var_units = {}
        self._grids = {}
        self._grid_type = {}

    def initialize(self, filename = 'deltaRCM.yaml'):

        """Initialize the model.

        Parameters
        ----------
        filename : str, optional
            Path to name of input file.
        """

        input_file_vars = dict()

        # Open and access yaml file --> put in dictionaries
        # only access the user input file if provided.

        if filename != 'deltaRCM.yaml' and os.path.exists(filename):
            user_file = open(filename, mode = 'r')
            user_dict = yaml.load(user_file, Loader = yaml.FullLoader)
            user_file.close()
        else:
            warnings.warn(UserWarning('The specified input file could not be found.'
                                      ' Using default values...'))
            user_dict = dict()

        # go through and populate input vars with user and default values from
        # default dictionary self._input_vars checking user values for correct type.

        for oo in self._input_vars.keys():
            model_name = self._input_vars[oo]['name']
            the_type = eval(self._input_vars[oo]['type'])
            if oo in user_dict and isinstance(user_dict[oo], the_type):
                input_file_vars[model_name] = user_dict[oo]
            elif oo in user_dict and not isinstance(user_dict[oo], the_type):
                warnings.warn(UserWarning('Input for ' + oo + ' not of the right type. '
                      + oo + ' needs to be of type ' + str(the_type)))
                input_file_vars[model_name] = self._input_vars[oo]['default']
            else:
                input_file_vars[model_name] = self._input_vars[oo]['default']

        tmpFile = os.path.join(os.getcwd(), '_tmp.yml')
        inbetweenYAML = open(tmpFile, 'w')
        yaml.dump(input_file_vars, inbetweenYAML)
        inbetweenYAML.close()

        self._delta = DeltaModel(input_file = tmpFile)

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

        os.remove(tmpFile)

    def update(self):
        """Advance model by one time step."""

        self._delta.update()

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
        return self._delta._time

    def get_time_step(self):
        """Time step of model."""
        return self._delta.time_step
