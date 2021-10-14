
import os
import logging
import warnings
import platform
import sys

from math import floor
import numpy as np

from scipy.sparse import lil_matrix, csr_matrix

from netCDF4 import Dataset
import time as time_lib
import yaml
import abc

from . import shared_tools
from . import sed_tools

# tools for initiating deltaRCM model domain


class init_tools(abc.ABC):

    def init_output_infrastructure(self):
        """Initialize the output infrastructure (folder and save lists).

        This method is the first called in the initialization of the
        `DeltaModel`, after the configuration variables have been imported.
        """
        # output directory config
        self.prefix = self.out_dir
        self.prefix_abspath = os.path.abspath(self.prefix)

        # create directory if it does not exist
        if not os.path.exists(self.prefix_abspath):
            os.makedirs(self.prefix_abspath)
            assert os.path.isdir(self.prefix_abspath)  # validate dir created

        self._save_fig_list = dict()  # dict of figure variables to save
        self._save_var_list = dict()  # dict of variables to save
        self._save_var_list['meta'] = dict()  # set up meta dict

    def init_logger(self):
        """Initialize a logger.

        The logger is initialized regardless of the value of ``self.verbose``.
        The level of information printed to the log depends on the verbosity
        setting.
        """
        timestamp = time_lib.strftime('%Y%m%d-%H%M%S')
        self.logger = logging.getLogger(self.prefix_abspath + timestamp)
        self.logger.setLevel(logging.INFO)

        # create the logging file handler
        fh = logging.FileHandler(
            self.prefix_abspath + '/pyDeltaRCM_' + timestamp + '.log')
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)

        # add handler to logger object
        self.logger.addHandler(fh)

        _msg = 'Output log file initialized'
        self.log_info(_msg, verbosity=0)

        # log attributes of the model and environment
        self.log_info('pyDeltaRCM version {}'.format(self.__pyDeltaRCM_version__))  # log the pyDeltaRCM version
        self.log_info('Python version {}'.format(sys.version),
                      verbosity=0)  # log the python version
        self.log_info('Platform: {}'.format(platform.platform()),
                      verbosity=0)  # log the os

    def import_files(self, kwargs_dict={}):
        """Import the input files.

        This method handles the parsing and validation of any options supplied
        via the configuration.

        Parameters
        ----------
        kwargs_dict : :obj:`dict`, optional

            A dictionary with keys matching valid model parameter names that
            can be specified in a configuration YAML file. Keys given in this
            dictionary will supercede values specified in the YAML
            configuration.

        Returns
        -------
        """
        # This dictionary serves as a container to hold values for both the
        # user-specified file and the internal defaults.
        input_file_vars = dict()

        # get the special loader from the shared tools
        loader = shared_tools.custom_yaml_loader()

        # Open and access both yaml files --> put in dictionaries
        # parse default yaml and find expected types
        default_file = open(self.default_file, mode='r')
        default_dict = yaml.load(default_file, Loader=loader)
        default_file.close()
        for k, v in default_dict.items():
            if not type(v['type']) is list:
                default_dict[k]['type'] = [eval(v['type'])]
            else:
                default_dict[k]['type'] = [eval(_v) for _v in v['type']]

        # only access the user input file if provided.
        if self.input_file:
            try:
                user_file = open(self.input_file, mode='r')
                user_dict = yaml.load(user_file, Loader=loader)
                user_file.close()
            except ValueError as e:
                raise e
        else:
            user_dict = dict()

        # replace values in the user yaml file with anything specifed in the
        #   **kwargs input
        for kwk, kwv in kwargs_dict.items():
            if kwk in user_dict.keys():
                warnings.warn(UserWarning(
                    'A keyword specification was also found in the '
                    'user specified input YAML file: %s' % kwk))
            user_dict[kwk] = kwv

        # go through and populate input vars with user and default values,
        # checking user values for correct type.
        for k, v in default_dict.items():
            if k in user_dict:
                expected_type = v['type']
                if type(user_dict[k]) in expected_type:
                    input_file_vars[k] = user_dict[k]
                else:
                    raise TypeError(f'Input for {str(k)} not of the '
                                    f'right type in yaml configuration '
                                    f'file {self.input_file}. '
                                    f'Input type was {type(k).__name__}, '
                                    f'but needs to be {expected_type}.')
            else:
                input_file_vars[k] = default_dict[k]['default']

        # add custom subclass yaml parameters (yaml or defaults) to input vars
        for k, v in self.subclass_parameters.items():
            if k in input_file_vars:
                warnings.warn(UserWarning(
                    'Custom subclass parameter name is already a '
                    'default yaml parameter of the model, '
                    'custom parameter value will not be used.'
                ))
            elif k in user_dict:
                # get expected types
                if not type(v['type']) is list:
                    expected_type = [eval(v['type'])]
                else:
                    expected_type = [eval(_v) for _v in v['type']]
                # evaluate against expected types
                if type(user_dict[k]) in expected_type:
                    input_file_vars[k] = user_dict[k]
                else:
                    raise TypeError(f'Input for {str(k)} not of the '
                                    f'right type in yaml configuration '
                                    f'file {self.input_file}. '
                                    f'Input type was {type(k).__name__}, '
                                    f'but needs to be {expected_type}.')
            else:
                # set using default value
                input_file_vars[k] = v['default']

        # save the input file as a hidden attr and grab needed value
        self._input_file_vars = input_file_vars
        self.out_dir = self._input_file_vars['out_dir']
        self.verbose = self._input_file_vars['verbose']
        if self._input_file_vars['legacy_netcdf']:
            self._netcdf_coords = ('total_time', 'length', 'width')
        else:
            self._netcdf_coords = ('time', 'x', 'y')

    def process_input_to_model(self):
        """Process input file to model variables.

        Loop through the items specified in the model configuration and apply
        them to the model (i.e., ``self``). Additionally, write the input
        values specified into the log file.

        .. note::

            If ``self.resume_checkpoint == True``, then the input values are
            *not* written to the log.
        """
        _msg = 'Setting up model configuration'
        self.log_info(_msg, verbosity=0)

        _msg = f'Model type is: {self.__class__.__name__}'
        self.log_info(_msg, verbosity=0)

        # process the input file to attributes of the model
        for k, v in list(self._input_file_vars.items()):
            setattr(self, k, v)

        # write the input file values to the log
        if not self._resume_checkpoint:
            for k, v in list(self._input_file_vars.items()):
                _msg = 'Configuration variable `{var}`: {val}'.format(
                    var=k, val=v)
                self.log_info(_msg, verbosity=0)

    def determine_random_seed(self):
        """Set the random seed if given.

        If a random seed is specified, set the seed to this value.

        Writes the seed to the log for record.
        """
        if self._seed is None:
            # generate a random seed for reproducibility
            self.seed = np.random.randint((2**32) - 1, dtype='u8')

        shared_tools.set_random_seed(self._seed)

        # always write the seed to file for record and reproducability
        _msg = 'Random seed is: %s ' % str(self._seed)
        self.log_info(_msg, verbosity=0)

    def create_other_variables(self):
        """Model implementation variables.

        Creates variables for model implementation, from specified boundary
        condition variables. This method is run during initial model
        instantition. Internally, this method calls :obj:`set_constants` and
        :obj:`create_boundary_conditions`.

        .. note::

            It is usually not necessary to re-run this method if you change
            boundary condition variables, and instead only re-run
            :obj:`create_boundary_conditions`.
        """
        _msg = 'Setting other variables'
        self.log_info(_msg, verbosity=1)

        self.init_Np_water = self._Np_water
        self.init_Np_sed = self._Np_sed

        self.dx = float(self._dx)

        self.theta_sand = self._coeff_theta_sand * self._theta_water
        self.theta_mud = self._coeff_theta_mud * self._theta_water

        self.U_dep_mud = self._coeff_U_dep_mud * self._u0
        self.U_ero_sand = self._coeff_U_ero_sand * self._u0
        self.U_ero_mud = self._coeff_U_ero_mud * self._u0

        self.L = int(round(self._Length / self._dx))        # num cells in x
        self.W = int(round(self._Width / self._dx))         # num cells in y

        # cross-stream center of domain idx
        self.CTR = floor(self.W / 2.) - 1
        if self.CTR <= 1:
            self.CTR = floor(self.W / 2.)

        self.set_constants()

        self.create_boundary_conditions()

        self._save_any_grids = (self._save_eta_grids or
                                self._save_depth_grids or
                                self._save_stage_grids or
                                self._save_discharge_grids or
                                self._save_velocity_grids or
                                self._save_sedflux_grids or
                                self._save_sandfrac_grids or
                                self._save_discharge_components or
                                self._save_velocity_components)
        if self._save_any_grids:  # always save metadata if saving grids
            self._save_metadata = True

        self._is_finalized = False

    def set_constants(self):
        """Set the model constants.

        Configure constants, including coordinates and distances, as well as
        environmental constants (gravity), and kernels for smoothing
        topography.

        Some of the constants defined herein:
            * `self.g`, gravitational acceleration
            * `self.distances`, distance from cell `i,j` to neighbors (and self)  # noqa: E501
            * `self.iwalk`, step distance cross domain to cell in indexed direction
            * `self.jwalk`, step distance down domain to cell in indexed direction
            * `self.ravel_walk`, flattened index distance to cell in indexed direction

        Each of these attributes also has a `self.xxxxx_flat` sibling
        attribute, which is simply the flattened version of that attribute.
        """
        _msg = 'Setting model constants'
        self.log_info(_msg, verbosity=1)

        # simple constants
        self.g = 9.81   # (gravitation const.)
        sqrt2 = np.sqrt(2)
        sqrt05 = np.sqrt(0.5)

        # translations arrays
        self.distances = np.array([[sqrt2,    1,  sqrt2],
                                   [1,        1,      1],
                                   [sqrt2,    1,  sqrt2]], dtype=np.float32)
        self.ivec      = np.array([[-sqrt05,  0,  sqrt05],  # noqa: E221
                                   [-1,       0,       1],
                                   [-sqrt05,  0,  sqrt05]], dtype=np.float32)
        self.jvec      = np.array([[-sqrt05, -1, -sqrt05],  # noqa: E221
                                   [0,        0,       0],
                                   [sqrt05,   1,  sqrt05]], dtype=np.float32)
        self.iwalk     = np.array([[-1,       0,       1],  # noqa: E221
                                   [-1,       0,       1],
                                   [-1,       0,       1]], dtype=np.int64)
        self.jwalk     = np.array([[-1,      -1,      -1],  # noqa: E221
                                   [0,        0,       0],
                                   [1,        1,       1]], dtype=np.int64)

        # derivatives of translations
        self.distances_flat = self.distances.flatten()
        self.ivec_flat = self.ivec.flatten()
        self.jvec_flat = self.jvec.flatten()
        self.iwalk_flat = self.iwalk.flatten()
        self.jwalk_flat = self.jwalk.flatten()
        self.ravel_walk = (self.jwalk * self.W) + self.iwalk  # walk, flattened
        self.ravel_walk_flat = self.ravel_walk.flatten()

        # kernels for topographic smoothing
        self.kernel1 = np.array([[1, 1, 1],
                                 [1, -8, 1],
                                 [1, 1, 1]]).astype(np.int64)

        self.kernel2 = np.array([[1, 1, 1],
                                 [1, 0, 1],
                                 [1, 1, 1]]).astype(np.int64)

    def create_boundary_conditions(self):
        """Create model boundary conditions

        This method is run during model initialization to determine the
        boundary conditions based on the initial conditions specified as model
        input.

        However, the method also should be called when certain boundary
        condition variables are updated (e.g., during some custom model runs).
        For example, if some property of the inlet flow condition is changed,
        you will *likely* want to re-run this method (e.g., `u0`), because
        there are several other parameters that depend on the value of the
        inlet flow velocity (e.g., `Qw0` and `Qs0`); of course, your
        scientific question may choose to leave these parameters as they are
        too (in which case you should *not* re-run this method). See
        :doc:`/examples/updating_boundary_conditions` for more information.

        .. note::

            This method is automatically called for the "named" variables used
            by the BMI wrapper (e.g., `channel_flow_velocity`,
            `channel_width`, `channel_flow_depth`, and
            `influx_sediment_concentration`., so you do not need to call it
            again if you modify the model boundary conditions that way.
        """
        # inlet length and width
        self.L0 = max(
            1, min(int(round(self._L0_meters / self._dx)), self.L // 4))
        self.N0 = max(
            3, min(int(round(self._N0_meters / self._dx)), self.W // 4))

        self.u_max = 2.0 * self._u0  # maximum allowed flow velocity
        self.C0 = self._C0_percent * 1 / 100.  # sediment concentration

        # (m) critial depth to switch to "dry" node
        self.dry_depth = min(0.1, 0.1 * self._h0)

        self.gamma = (self.g * self.S0 * self._dx /
                      (self.u0**2))  # water weighting coeff

        # (m^3) reference volume, volume to fill cell to characteristic depth
        self.V0 = self.h0 * (self._dx**2)
        self.Qw0 = self.u0 * self.h0 * self.N0 * self._dx    # const discharge

        # at inlet
        self.qw0 = self.u0 * self.h0  # water unit input discharge
        self.Qp_water = self.Qw0 / self._Np_water  # volume each water parcel
        self.qs0 = self.qw0 * self.C0  # sed unit discharge
        self.dVs = 0.1 * self.N0**2 * self.V0  # total sed added per timestep
        self.Qs0 = self.Qw0 * self.C0  # sediment total input discharge
        self.Vp_sed = self.dVs / self._Np_sed  # volume of each sediment parcel

        # max number of jumps for parcel
        if self.stepmax is None:
            self.stepmax = 2 * (self.L + self.W)
        else:
            self.stepmax = int(self.stepmax)

        # initial width of self.free_surf_walk_indices
        self.size_indices = int(self.stepmax / 2)

        self._dt = self.dVs / self.Qs0  # time step size

        self.omega_flow_iter = 2. / self._itermax

        # number of times to repeat topo diffusion
        self.N_crossdiff = int(round(self.dVs / self.V0))

        self._lambda = self._sed_lag  # sedimentation lag

        self.diffusion_multiplier = (self._dt / self.N_crossdiff * self._alpha
                                     * 0.5 / self._dx**2)

    def create_domain(self):
        """Create the model domain.

        This method initializes the model domain, including coordinate arrays,
        gridded fields (eta, qw, etc.) and cell type information. The method
        is called during initialization of the `DeltaModel`, and likely does
        not need to be called again.

        .. hint::

            If you need to modify the model domain after it has been created,
            it is probably safe to modify attributes directly, but take care
            to ensure any dependent fields are also approrpriately changed.
        """
        _msg = 'Creating model domain'
        self.log_info(_msg, verbosity=1)

        # resolve any boundary conditions
        self._hb = self.hb or self.h0  # basin depth

        # ---- coordinates ----
        self.xc = np.arange(0, self.L) * self._dx
        self.yc = np.arange(0, self.W) * self._dx
        self.x, self.y = np.meshgrid(np.arange(0, self.W),
                                     np.arange(0, self.L))
        self.X, self.Y = np.meshgrid(np.arange(0, self.W+1)*self._dx,
                                     np.arange(0, self.L+1)*self._dx)

        # ---- empty arrays ----
        self.cell_type = np.zeros((self.L, self.W), dtype=np.int64)
        self.eta = np.zeros((self.L, self.W), dtype=np.float32)
        self.eta0 = np.copy(self.eta)  # establish eta0 copy
        self.stage = np.zeros((self.L, self.W), dtype=np.float32)
        self.depth = np.zeros((self.L, self.W), dtype=np.float32)
        self.qx = np.zeros((self.L, self.W), dtype=np.float32)
        self.qy = np.zeros((self.L, self.W), dtype=np.float32)
        self.qxn = np.zeros((self.L, self.W), dtype=np.float32)
        self.qyn = np.zeros((self.L, self.W), dtype=np.float32)
        self.qwn = np.zeros((self.L, self.W), dtype=np.float32)
        self.ux = np.zeros((self.L, self.W), dtype=np.float32)
        self.uy = np.zeros((self.L, self.W), dtype=np.float32)
        self.uw = np.zeros((self.L, self.W), dtype=np.float32)
        self.qs = np.zeros((self.L, self.W), dtype=np.float32)
        self.sand_frac = np.zeros((self.L, self.W), dtype=np.float32) + \
            self.sand_frac_bc
        self.active_layer = np.zeros((self.L, self.W), dtype=np.float32) + \
            self.sand_frac_bc
        self.Vp_dep_sand = np.zeros((self.L, self.W), dtype=np.float32)
        self.Vp_dep_mud = np.zeros((self.L, self.W), dtype=np.float32)

        # arrays for computing the free surface after water iteration
        self.free_surf_flag = np.zeros((self._Np_water,), dtype=np.int64)
        self.free_surf_walk_inds = np.zeros(
            (self._Np_water, self.size_indices), dtype=np.int64)
        self.sfc_visit = np.zeros_like(self.depth)
        self.sfc_sum = np.zeros_like(self.depth)

        # ---- domain ----
        cell_land = -2
        cell_channel = 1
        cell_ocean = 0
        cell_edge = -1

        self.cell_type[:self.L0, :] = cell_land

        channel_inds = int(self.CTR - round(self.N0 / 2)) + 1
        y_channel_max = channel_inds + self.N0
        self.cell_type[:self.L0, channel_inds:y_channel_max] = cell_channel

        self.stage[:] = (np.maximum(0, self.L0 - self.y - 1) *
                         self._dx * self._S0)
        self.stage[self.cell_type == cell_ocean] = 0.

        self.depth[self.cell_type == cell_ocean] = self.hb
        self.depth[self.cell_type == cell_channel] = self.h0

        self.qx[self.cell_type == cell_channel] = self.qw0
        self.qx[self.cell_type == cell_ocean] = self.qw0 / 5.
        self.qw = (self.qx**2 + self.qy**2)**(0.5)

        self.ux[self.depth > 0] = self.qx[
            self.depth > 0] / self.depth[self.depth > 0]
        self.uy[self.depth > 0] = self.qy[
            self.depth > 0] / self.depth[self.depth > 0]
        self.uw[self.depth > 0] = self.qw[
            self.depth > 0] / self.depth[self.depth > 0]

        # reset the land cell_type to -2
        self.cell_type[self.cell_type == cell_land] = -2
        self.cell_type[-1, :] = cell_edge
        self.cell_type[:, 0] = cell_edge
        self.cell_type[:, -1] = cell_edge

        bounds = [(np.sqrt((i - 3)**2 + (j - self.CTR)**2))
                  for i in range(self.L)
                  for j in range(self.W)]
        bounds = np.reshape(bounds, (self.L, self.W))

        self.cell_type[bounds >= min(self.L - 5, self.W / 2 - 5)] = cell_edge

        self.cell_type[:self.L0, :] = cell_land
        self.cell_type[:self.L0, channel_inds:y_channel_max] = cell_channel

        self.inlet = np.array(np.unique(np.where(self.cell_type == 1)[1]))
        self.eta[:] = self.stage - self.depth

    def init_sediment_routers(self):
        """Initialize the sediment router object here.

        These are preinitialized because the "boxing" for jitted functions is
        expensive, so we avoid boxing up the constants on each iteration.

        .. important::

            If you change any model boundary conditions, you likely should
            reinitialize the sediment routers (i.e., rerun this method)
        """
        _msg = 'Initializing sediment routers'
        self.log_info(_msg, verbosity=1)

        # initialize the MudRouter object
        self._mr = sed_tools.MudRouter(self._dt, self._dx, self.Vp_sed,
                                       self.u_max, self.U_dep_mud,
                                       self.U_ero_mud,
                                       self.ivec_flat, self.jvec_flat,
                                       self.iwalk_flat, self.jwalk_flat,
                                       self.distances_flat, self.dry_depth,
                                       self._lambda, self._beta,  self.stepmax,
                                       self.theta_mud)
        # initialize the SandRouter object
        self._sr = sed_tools.SandRouter(self._dt, self._dx, self.Vp_sed,
                                        self.u_max, self.qs0, self._u0,
                                        self.U_ero_sand,
                                        self._f_bedload,
                                        self.ivec_flat, self.jvec_flat,
                                        self.iwalk_flat, self.jwalk_flat,
                                        self.distances_flat, self.dry_depth,
                                        self._beta, self.stepmax,
                                        self.theta_sand)

    def init_output_file(self):
        """Creates a netCDF file to store output grids.

        Fills with default variables.

        .. warning::

            Overwrites an existing netcdf file with the same name if
            :attr:`~pyDeltaRCM.model.DeltaModel.clobber_netcdf` is `True`.

        """
        _msg = 'Initializing output NetCDF4 file'
        self.log_info(_msg, verbosity=1)

        # set standard/default metadata values in the dict structure
        if self._save_metadata:
            self.init_metadata_list()

        if (self._save_metadata or
                self._save_any_grids):

            directory = self.prefix
            filename = 'pyDeltaRCM_output.nc'

            file_path = os.path.join(directory, filename)
            _msg = 'Target output NetCDF4 file: {file}'.format(
                file=file_path)
            self.log_info(_msg, verbosity=2)

            if (os.path.exists(file_path)) and \
                    (self._clobber_netcdf is False):
                raise FileExistsError(
                    'Existing NetCDF4 output file in target output location.')
            elif (os.path.exists(file_path)) and \
                    (self._clobber_netcdf is True):
                _msg = 'Replacing existing netCDF file'
                self.logger.warning(_msg)
                warnings.warn(UserWarning(_msg))
                os.remove(file_path)

            self.output_netcdf = Dataset(file_path, 'w',
                                         format='NETCDF4')

            self.output_netcdf.description = 'Output from pyDeltaRCM'
            self.output_netcdf.history = ('Created '
                                          + time_lib.ctime(time_lib.time()))
            self.output_netcdf.source = 'pyDeltaRCM v{ver}'.format(
                ver=self.__pyDeltaRCM_version__)

            # create master dimensions (pulls from `self._netcdf_coords`)
            self.output_netcdf.createDimension(self._netcdf_coords[1], self.L)
            self.output_netcdf.createDimension(self._netcdf_coords[2], self.W)
            self.output_netcdf.createDimension(self._netcdf_coords[0], None)

            # create master coordinates (as netCDF variables)
            time = self.output_netcdf.createVariable(
                'time', 'f4', (self._netcdf_coords[0],))
            time.units = 'second'

            if self._legacy_netcdf:
                # old format is 2d array x and y
                x = self.output_netcdf.createVariable(
                    'x', 'f4', self._netcdf_coords[1:])
                y = self.output_netcdf.createVariable(
                    'y', 'f4', self._netcdf_coords[1:])
                x[:] = self.x
                y[:] = self.y

            else:
                # new output format is 1d x and y
                x = self.output_netcdf.createVariable(
                    'x', 'f4', ('x'))
                y = self.output_netcdf.createVariable(
                    'y', 'f4', ('y'))
                x[:] = self.xc
                y[:] = self.yc

            x.units = 'meter'
            y.units = 'meter'

            # set up variables for output data grids
            def _create_grid_variable(varname, varunits,
                                      vartype='f4', vardims=()):
                _v = self.output_netcdf.createVariable(
                    varname, vartype, vardims)
                _v.units = varunits

            _var_list = list(self._save_var_list.keys())
            _var_list.remove('meta')
            for _val in _var_list:
                _create_grid_variable(_val, self._save_var_list[_val][1],
                                      self._save_var_list[_val][2],
                                      self._save_var_list[_val][3])

            # set up metadata group and populate variables
            def _create_meta_variable(varname, varvalue, varunits,
                                      vartype='f4', vardims=()):
                _v = self.output_netcdf.createVariable(
                    'meta/'+varname, vartype, vardims)
                _v.units = varunits
                _v[:] = varvalue

            self.output_netcdf.createGroup('meta')
            for _val in self._save_var_list['meta'].keys():
                # time-varying initialize w/ None value, fixed use attribute
                if (self._save_var_list['meta'][_val][0] is None):
                    _create_meta_variable(
                        _val, self._save_var_list['meta'][_val][0],
                        self._save_var_list['meta'][_val][1],
                        self._save_var_list['meta'][_val][2],
                        self._save_var_list['meta'][_val][3])
                else:
                    _create_meta_variable(
                        _val, getattr(self,
                                      self._save_var_list['meta'][_val][0]),
                        self._save_var_list['meta'][_val][1],
                        self._save_var_list['meta'][_val][2],
                        self._save_var_list['meta'][_val][3])

            _msg = 'Output netCDF file created'
            self.log_info(_msg, verbosity=2)

    def init_subsidence(self):
        """Initialize subsidence pattern.

        Initializes patterns of subsidence if toggle_subsidence is True
        (default False). Default behavior is for the entire basin to subside
        at a constant rate (with the exception of the land boundary cells along
        the inlet boundary).

        To customize the subsiding region, or even vary the subsiding region
        over the course of the model run, we recommend subclassing the
        DeltaModel class.

        """
        _msg = 'Initializing subsidence'
        self.log_info(_msg, verbosity=1)

        if self._toggle_subsidence:

            self.subsidence_mask = np.ones((self.L, self.W), dtype=bool)
            self.subsidence_mask[:self.L0, :] = False

            self.sigma = self.subsidence_mask * self._subsidence_rate * self.dt

    def init_metadata_list(self):
        """Populate the list of metadata information.

        Sets up the dictionary object for the standard metadata.
        """
        # fixed metadata
        self._save_var_list['meta']['L0'] = ['L0', 'cells', 'i8', ()]
        self._save_var_list['meta']['N0'] = ['N0', 'cells', 'i8', ()]
        self._save_var_list['meta']['CTR'] = ['CTR', 'cells', 'i8', ()]
        self._save_var_list['meta']['dx'] = ['dx', 'meters', 'f4', ()]
        self._save_var_list['meta']['h0'] = ['h0', 'meters', 'f4', ()]
        self._save_var_list['meta']['hb'] = ['hb', 'meters', 'f4', ()]
        self._save_var_list['meta']['cell_type'] = ['cell_type',
                                                    'type', 'i8',
                                                    self._netcdf_coords[1:]]
        # subsidence metadata
        if self._toggle_subsidence:
            self._save_var_list['meta']['start_subsidence'] = [
                'start_subsidence', 'seconds', 'i8', ()
            ]
            self._save_var_list['meta']['sigma'] = [
                'sigma', 'meters per timestep', 'f4',
                self._netcdf_coords[1:]
            ]
        # time-varying metadata
        self._save_var_list['meta']['H_SL'] = [None, 'meters', 'f4',
                                               (self._netcdf_coords[0])]
        self._save_var_list['meta']['f_bedload'] = [None, 'fraction',
                                                    'f4',
                                                    (self._netcdf_coords[0])]
        self._save_var_list['meta']['C0_percent'] = [None, 'percent',
                                                     'f4',
                                                     (self._netcdf_coords[0])]
        self._save_var_list['meta']['u0'] = [None, 'meters per second',
                                             'f4',
                                             (self._netcdf_coords[0])]

    def load_checkpoint(self, defer_output=False):
        """Load the checkpoint from the .npz file.

        Uses the file at the path determined by `self.prefix` and a file named
        `checkpoint.npz`. There are a few pathways for loading, which depend
        on 1) the status of additional grid-saving options, 2) the presence of
        a netCDF output file at the expected output location, and 3) the status
        of the :obj:`defer_output` parameter passed to this method as an
        optional argument.

        As a standard user, you should not need to worry about any of these
        pathways or options. However, if you are developing pyDeltaRCM or
        customizing the model in any way that involves loadind from
        checkpoints, you should be aware of these pathways.

        For example, loading from checkpoint will succeed if no netCDF4 file
        is found, where one is expected (e.g., because :obj:`_save_any_grids`
        is `True`). In this case, a new output netcdf file will be created,
        after a `UserWarning` is issued.

        .. important::

            If you are customing the model and intend to use checkpointing and
            the :obj:`Preprocessor` parallel infrastructure, be sure that
            parameter :obj:`defer_output` is `True` until the
            :obj:`load_checkpoint` method can be called from the thread the
            model will execute on. Failure to do so may result in unexpected
            behavior with indexing in the output netCDF4 file.

        Parameters
        ----------
        defer_output : :obj:`bool`, optional
            Whether to defer any netCDF activities at present. Manipulating
            this variable is critical for parallel operations. See note above.
            Default is `False`.
        """
        _msg = 'Loading from checkpoint.'
        self.log_info(_msg, verbosity=0)

        _msg = 'Locating checkpoint file'
        self.log_info(_msg, verbosity=2)
        ckp_file = os.path.join(self._checkpoint_folder, 'checkpoint.npz')
        checkpoint = np.load(ckp_file, allow_pickle=True)

        # write saved variables back to the model
        _msg = 'Loading variables and grids into model'
        self.log_info(_msg, verbosity=2)

        # load time and counter vars
        self._time = float(checkpoint['time'])
        self.H_SL = float(checkpoint['H_SL'])
        self._time_iter = int(checkpoint['time_iter'])
        self._save_iter = int(checkpoint['save_iter'])
        self._save_time_since_data = int(checkpoint['save_time_since_data'])

        # load grids
        self.eta = checkpoint['eta']
        self.depth = checkpoint['depth']
        self.stage = checkpoint['stage']
        self.uw = checkpoint['uw']
        self.ux = checkpoint['ux']
        self.uy = checkpoint['uy']
        self.qw = checkpoint['qw']
        self.qx = checkpoint['qx']
        self.qy = checkpoint['qy']
        self.sand_frac = checkpoint['sand_frac']
        self.active_layer = checkpoint['active_layer']

        # load and set random state to continue as if run hadn't stopped
        _msg = 'Loading random state'
        self.log_info(_msg, verbosity=2)
        rng_state = tuple(checkpoint['rng_state'])
        shared_tools.set_random_state(rng_state)

        # handle the case with a netcdf file
        if ((self._save_any_grids or self._save_metadata)
                and not defer_output):

            # check if the file exists already
            #    if it does, it needs to be flushed of certain fields
            file_path = os.path.join(self.prefix, 'pyDeltaRCM_output.nc')
            if not os.path.isfile(file_path):
                _msg = ('NetCDF4 output file not found, but was expected. '
                        'Creating a new output file.')
                self.logger.warning(_msg)
                warnings.warn(UserWarning(_msg))

                # create a new file
                # reset output file counters
                self._save_iter = int(0)
                self.hook_init_output_file()
                self.init_output_file()
            else:
                # check if file is open in another process, if so throw error
                try:
                    _dataset = Dataset(file_path, 'r+', format='NETCDF')
                    _dataset.close()  # if this worked then close it
                except OSError:
                    raise RuntimeError(
                        'Could not open the NetCDF file for checkpointing. '
                        'This could be because the file is corrupt, or open '
                        'in another interpreter. Be sure to close any '
                        'connections to the file before proceeding.')
                # if not open elsewhere, then proceed
                # rename the old netCDF4 file
                _msg = 'Renaming old NetCDF4 output file'
                self.log_info(_msg, verbosity=2)
                _tmp_name = os.path.join(self.prefix,
                                         'old_pyDeltaRCM_output.nc')
                os.rename(file_path, _tmp_name)

                # write dims / attributes / variables to new netCDF file
                _msg = 'Creating NetCDF4 output file'
                self.log_info(_msg, verbosity=2)

                # populate default metadata list
                if self._save_metadata:
                    self.init_metadata_list()

                # copy data from old netCDF4 into new one
                with Dataset(_tmp_name) as src, Dataset(file_path, 'w',
                                                        format='NETCDF4') as dst:
                    # copy attributes
                    for name in src.ncattrs():
                        dst.setncattr(name, src.getncattr(name))
                    # copy dimensions
                    for name, dimension in src.dimensions.items():
                        if dimension.isunlimited():
                            dst.createDimension(name, None)
                        else:
                            dst.createDimension(name, len(dimension))
                    # copy groups (meta)
                    for name in src.groups.keys():
                        dst.createGroup(name)
                        for vname, variable in src.groups[name].variables.items():
                            _mname = name + '/' + vname
                            dst.createVariable(_mname, variable.datatype,
                                               variable.dimensions)
                            dst.groups[name].variables[vname][:] = \
                                src.groups[name].variables[vname][:]
                    # copy variables except ones to exclude
                    for name, variable in src.variables.items():
                        dst.createVariable(name, variable.datatype,
                                           variable.dimensions)
                        dst.variables[name][:] = src.variables[name][:]

                # set object attribute for model
                self.output_netcdf = Dataset(file_path, 'r+', format='NETCDF4')

                # synch netcdf file
                self.output_netcdf.sync()

                # delete old netCDF4 file
                os.remove(_tmp_name)

        _msg = 'Successfully loaded checkpoint.'
        self.log_info(_msg, verbosity=1)
