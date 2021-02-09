
import os
import logging
import warnings

from math import floor, sqrt, pi
import numpy as np

from scipy.sparse import lil_matrix, csr_matrix
from scipy import ndimage

from netCDF4 import Dataset
import time as time_lib
import yaml
import re
import abc

from . import shared_tools
from . import sed_tools

# tools for initiating deltaRCM model domain


class init_tools(abc.ABC):

    def init_output_infrastructure(self):
        """Initialize the output infrastructure (i.e., folder).

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

        _msg = 'Output log file initialized.'
        self.log_info(_msg, verbosity=0)

    def import_files(self):

        # This dictionary serves as a container to hold values for both the
        # user-specified file and the internal defaults.
        input_file_vars = dict()

        # Define a loader to handle scientific notation.
        #   waiting for upstream fix here:
        #      https://github.com/yaml/pyyaml/pull/174
        loader = yaml.SafeLoader
        loader.add_implicit_resolver(
            u'tag:yaml.org,2002:float',
            re.compile(r'''^(?:[-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+]?[0-9]+)?
                           |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
                           |\.[0-9_]+(?:[eE][-+]?[0-9]+)?
                           |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\.[0-9_]*
                           |[-+]?\.(?:inf|Inf|INF)
                           |\.(?:nan|NaN|NAN))$''', re.X),
            list(u'-+0123456789.'))

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

        # go through and populate input vars with user and default values,
        # checking user values for correct type.
        for k, v in default_dict.items():
            if k in user_dict:
                expected_type = v['type']
                if type(user_dict[k]) in expected_type:
                    input_file_vars[k] = user_dict[k]
                else:
                    raise TypeError('Input for "{_k}" not of the right type '
                                    'in yaml configuration file "{_file}". '
                                    'Input type was "{_wastype}", '
                                    'but needs to be "{_exptype}".'
                                    .format(_k=str(k), _file=self.input_file,
                                            _wastype=type(k).__name__,
                                            _exptype=expected_type))
            else:
                input_file_vars[k] = default_dict[k]['default']

        # save the input file as a hidden attr and grab needed value
        self._input_file_vars = input_file_vars
        self.out_dir = self._input_file_vars['out_dir']
        self.verbose = self._input_file_vars['verbose']

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

        # process the input file to attributes of the model
        for k, v in list(self._input_file_vars.items()):
            setattr(self, k, v)

        # if checkpoint_dt is the default value (None), then set it to save_dt
        if self.checkpoint_dt is None:
            self.checkpoint_dt = self._save_dt

        # handle a not implemented setup
        if self.save_checkpoint and self._toggle_subsidence:
            raise NotImplementedError('Cannot handle checkpointing with subsidence.')

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

    def set_constants(self):

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
        self.ravel_walk = (self.jwalk * self.W) + self.iwalk  # walk in flattened array
        self.ravel_walk_flat = self.ravel_walk.flatten()

        # kernels for topographic smoothing
        self.kernel1 = np.array([[1, 1, 1],
                                 [1, -8, 1],
                                 [1, 1, 1]]).astype(np.int64)

        self.kernel2 = np.array([[1, 1, 1],
                                 [1, 0, 1],
                                 [1, 1, 1]]).astype(np.int64)

    def create_other_variables(self):
        """Model implementation variables.

        Creates variables for model implementation, from specified boundary
        condition variables. This method is run during initial model
        instantition, but it is also run any time an inlet flow condition
        variable is changed, including `channel_flow_velocity`,
        `channel_width`, `channel_flow_depth`, and
        `influx_sediment_concentration`.
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

        # inlet length and width
        self.L0 = max(
            1, min(int(round(self._L0_meters / self._dx)), self.L // 4))
        self.N0 = max(
            3, min(int(round(self._N0_meters / self._dx)), self.W // 4))

        self.set_constants()

        self.u_max = 2.0 * self._u0  # maximum allowed flow velocity
        self.C0 = self._C0_percent * 1 / 100.  # sediment concentration

        # (m) critial depth to switch to "dry" node
        self.dry_depth = min(0.1, 0.1 * self._h0)

        # cross-stream center of domain idx
        self.CTR = floor(self.W / 2.) - 1
        if self.CTR <= 1:
            self.CTR = floor(self.W / 2.)

        self.gamma = (self.g * self._S0 * self._dx /
                      (self._u0**2))  # water weighting coeff

        # (m^3) reference volume, volume to fill cell to characteristic depth
        self.V0 = self.h0 * (self._dx**2)
        self.Qw0 = self._u0 * self.h0 * self.N0 * self._dx    # const discharge

        # at inlet
        self.qw0 = self._u0 * self.h0  # water unit input discharge
        self.Qp_water = self.Qw0 / self._Np_water  # volume each water parcel
        self.qs0 = self.qw0 * self.C0  # sed unit discharge
        self.dVs = 0.1 * self.N0**2 * self.V0  # total sed added per timestep
        self.Qs0 = self.Qw0 * self.C0  # sediment total input discharge
        self.Vp_sed = self.dVs / self._Np_sed   # volume of each sediment parcel

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

        self._save_any_grids = (self._save_eta_grids or
                                self._save_depth_grids or
                                self._save_stage_grids or
                                self._save_discharge_grids or
                                self._save_velocity_grids or
                                self._save_sedflux_grids or
                                self._save_discharge_components or
                                self._save_velocity_components)
        self._save_any_figs = (self._save_eta_figs or
                               self._save_depth_figs or
                               self._save_stage_figs or
                               self._save_discharge_figs or
                               self._save_velocity_figs or
                               self._save_sedflux_figs)
        if self._save_any_grids:  # always save metadata if saving grids
            self._save_metadata = True
        self._is_finalized = False

    def create_domain(self):
        """
        Creates the model domain
        """
        _msg = 'Creating model domain'
        self.log_info(_msg, verbosity=1)

        # ---- empty arrays ----
        self.x, self.y = np.meshgrid(np.arange(0, self.W),
                                     np.arange(0, self.L))
        self.X, self.Y = np.meshgrid(np.arange(0, self.W+1)*self._dx,
                                     np.arange(0, self.L+1)*self._dx)

        self.cell_type = np.zeros((self.L, self.W), dtype=np.int64)
        self.eta = np.zeros((self.L, self.W), dtype=np.float32)
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

        self.Vp_dep_sand = np.zeros((self.L, self.W), dtype=np.float32)
        self.Vp_dep_mud = np.zeros((self.L, self.W), dtype=np.float32)

        # arrays for computing the free surface after water iteration
        self.free_surf_flag = np.zeros((self._Np_water,), dtype=np.int64)
        self.free_surf_walk_inds = np.zeros(
            (self._Np_water, self.size_indices), dtype=np.int64)
        self.sfc_visit = np.zeros_like(self.depth)
        self.sfc_sum = np.zeros_like(self.depth)

        # ---- domain ----
        cell_land = 2
        cell_channel = 1
        cell_ocean = 0
        cell_edge = -1

        self.cell_type[:self.L0, :] = cell_land

        channel_inds = int(self.CTR - round(self.N0 / 2)) + 1
        y_channel_max = channel_inds + self.N0
        self.cell_type[:self.L0, channel_inds:y_channel_max] = cell_channel

        self.stage[:] = np.maximum(0, self.L0 - self.y - 1) * self._dx * self._S0
        self.stage[self.cell_type == cell_ocean] = 0.

        self.depth[self.cell_type == cell_ocean] = self.h0
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

        self.cell_type[:self.L0, :] = -cell_land
        self.cell_type[:self.L0, channel_inds:y_channel_max] = cell_channel

        self.inlet = np.array(np.unique(np.where(self.cell_type == 1)[1]))
        self.eta[:] = self.stage - self.depth

    def init_sediment_routers(self):
        """Initialize the sediment router object here.

        These are preinitialized because the "boxing" for jitted functions is
        expensive, so we avoid boxing up the constants on each iteration.
        """
        _msg = 'Initializing sediment routers'
        self.log_info(_msg, verbosity=1)

        # initialize the MudRouter object
        self._mr = sed_tools.MudRouter(self._dt, self._dx, self.Vp_sed,
                                       self.u_max, self.U_dep_mud, self.U_ero_mud,
                                       self.ivec_flat, self.jvec_flat,
                                       self.iwalk_flat, self.jwalk_flat,
                                       self.distances_flat, self.dry_depth,
                                       self._lambda, self._beta,  self.stepmax,
                                       self.theta_mud)
        # initialize the SandRouter object
        self._sr = sed_tools.SandRouter(self._dt, self._dx, self.Vp_sed,
                                        self.u_max, self.qs0, self._u0, self.U_ero_sand,
                                        self._f_bedload,
                                        self.ivec_flat, self.jvec_flat,
                                        self.iwalk_flat, self.jwalk_flat,
                                        self.distances_flat, self.dry_depth,
                                        self._beta, self.stepmax,
                                        self.theta_sand)

    def init_stratigraphy(self):
        """Creates sparse array to store stratigraphy data."""
        _msg = 'Initializing stratigraphy storage'
        self.log_info(_msg, verbosity=1)
        if self.save_strata:

            self.strata_counter = 0

            self.n_steps = int(max(1, 5 * int(self._save_dt / self.dt)))

            self.strata_sand_frac = lil_matrix((self.L * self.W, self.n_steps),
                                               dtype=np.float32)

            self.init_eta = self.eta.copy()
            self.strata_eta = lil_matrix((self.L * self.W, self.n_steps),
                                         dtype=np.float32)

    def init_output_file(self):
        """Creates a netCDF file to store output grids.

        Fills with default variables.

        .. warning:: Overwrites an existing netcdf file with the same name.

        """
        _msg = 'Initializing output NetCDF4 file'
        self.log_info(_msg, verbosity=1)

        if (self._save_metadata or
                self._save_any_grids or
                self.save_strata):

            directory = self.prefix
            filename = 'pyDeltaRCM_output.nc'

            file_path = os.path.join(directory, filename)
            _msg = 'Target output NetCDF4 file: {file}'.format(
                file=file_path)
            self.log_info(_msg, verbosity=2)

            if os.path.exists(file_path):
                _msg = 'Replacing existing netCDF file'
                self.logger.warning(_msg)
                warnings.warn(UserWarning(_msg))
                os.remove(file_path)

            self.output_netcdf = Dataset(file_path, 'w',
                                         format='NETCDF4')

            self.output_netcdf.description = 'Output from pyDeltaRCM'
            self.output_netcdf.history = ('Created '
                                          + time_lib.ctime(time_lib.time()))
            self.output_netcdf.source = 'pyDeltaRCM'

            # create master dimensions
            length = self.output_netcdf.createDimension('length', self.L)
            width = self.output_netcdf.createDimension('width', self.W)
            total_time = self.output_netcdf.createDimension('total_time', None)

            # create master coordinates (as netCDF variables)
            x = self.output_netcdf.createVariable(
                'x', 'f4', ('length', 'width'))
            y = self.output_netcdf.createVariable(
                'y', 'f4', ('length', 'width'))
            time = self.output_netcdf.createVariable('time', 'f4',
                                                     ('total_time',))
            x.units = 'meters'
            y.units = 'meters'
            time.units = 'second'
            x[:] = self.x
            y[:] = self.y

            # set up variables for output data grids
            if self.save_eta_grids:
                eta = self.output_netcdf.createVariable(
                    'eta', 'f4', ('total_time', 'length', 'width'))
                eta.units = 'meters'
            if self.save_stage_grids:
                stage = self.output_netcdf.createVariable(
                    'stage', 'f4', ('total_time', 'length', 'width'))
                stage.units = 'meters'
            if self.save_depth_grids:
                depth = self.output_netcdf.createVariable(
                    'depth', 'f4', ('total_time', 'length', 'width'))
                depth.units = 'meters'
            if self.save_discharge_grids:
                discharge = self.output_netcdf.createVariable(
                    'discharge', 'f4', ('total_time', 'length', 'width'))
                discharge.units = 'cubic meters per second'
            if self.save_velocity_grids:
                velocity = self.output_netcdf.createVariable(
                    'velocity', 'f4', ('total_time', 'length', 'width'))
                velocity.units = 'meters per second'
            if self.save_sedflux_grids:
                sedflux = self.output_netcdf.createVariable(
                    'sedflux', 'f4', ('total_time', 'length', 'width'))
                sedflux.units = 'cubic meters per second'
            if self.save_discharge_components:
                discharge_x = self.output_netcdf.createVariable(
                    'discharge_x', 'f4', ('total_time', 'length', 'width'))
                discharge_x.units = 'cubic meters per second'
                discharge_y = self.output_netcdf.createVariable(
                    'discharge_y', 'f4', ('total_time', 'length', 'width'))
                discharge_y.units = 'cubic meters per second'
            if self.save_velocity_components:
                velocity_x = self.output_netcdf.createVariable(
                    'velocity_x', 'f4', ('total_time', 'length', 'width'))
                velocity_x.units = 'meters per second'
                velocity_y = self.output_netcdf.createVariable(
                    'velocity_y', 'f4', ('total_time', 'length', 'width'))
                velocity_y.units = 'meters per second'

            # set up metadata group and populate variables
            def _create_meta_variable(varname, varvalue, varunits,
                                      vartype='f4', vardims=()):
                _v = self.output_netcdf.createVariable(
                    'meta/'+varname, vartype, vardims)
                _v.units = varunits
                _v[:] = varvalue

            self.output_netcdf.createGroup('meta')
            # fixed metadata
            _create_meta_variable('L0', self.L0, 'cells', vartype='i8')
            _create_meta_variable('N0', self.N0, 'cells', vartype='i8')
            _create_meta_variable('CTR', self.CTR, 'cells', vartype='i8')
            _create_meta_variable('dx', self.dx, 'meters')
            _create_meta_variable('h0', self.h0, 'meters')
            _create_meta_variable('cell_type', self.cell_type, 'type',
                                  vartype='i8', vardims=('length', 'width'))
            # time-varying metadata
            _create_meta_variable('H_SL', None, 'meters',
                                  vardims=('total_time'))
            _create_meta_variable('f_bedload', None, 'fraction',
                                  vardims=('total_time'))
            _create_meta_variable('C0_percent', None, 'percent',
                                  vardims=('total_time'))
            _create_meta_variable('u0', None, 'meters per second',
                                  vardims=('total_time'))

            _msg = 'Output netCDF file created'
            self.log_info(_msg, verbosity=2)


    def init_subsidence(self):
        """Initialize subsidence pattern.

        Initializes patterns of subsidence if
        toggle_subsidence is True (default False)

        Uses theta1 and theta2 (defined in yaml) to set the angular bounds for
        the subsiding region. To create a custom subsidence region, we
        recommend subclassing the DeltaModel class and defining your own array
        for self.subsidence_mask (a binary array with 1s in the cells that
        are subsiding and 0s elswhere), and self.sigma (the subsidence mask
        multiplied with the vertical rate of subsidence and the timestep size).

        theta1 and theta2 are set in relation to the inlet orientation. The
        inlet channel is at an angle of 0, if theta1 is -pi/3 radians, this
        means that the angle to the left of the inlet that will be included
        in the subsiding region is 30 degrees. theta2 defines the right angular
        bounds for the subsiding region in a similar fashion.
        """
        _msg = 'Initializing subsidence'
        self.log_info(_msg, verbosity=1)

        if self._toggle_subsidence:

            R1 = 0.3 * self.L
            R2 = 1. * self.L  # radial limits (fractions of L)

            Rloc = np.sqrt((self.y - self.L0)**2 + (self.x - self.W / 2.)**2)

            thetaloc = np.zeros((self.L, self.W))
            thetaloc[self.y > self.L0 - 1] = np.arctan(
                (self.x[self.y > self.L0 - 1] - self.W / 2.)
                / (self.y[self.y > self.L0 - 1] - self.L0 + 1))
            self.subsidence_mask = ((R1 <= Rloc) & (Rloc <= R2) &
                                    (self._theta1 <= thetaloc) &
                                    (thetaloc <= self._theta2))
            self.subsidence_mask[:self.L0, :] = False

            self.sigma = self.subsidence_mask * self._sigma_max * self.dt

    def load_checkpoint(self):
        """Load the checkpoint from the .npz file.

        Uses the file at the path determined by `self.prefix` and a file named
        `checkpoint.npz`.
        """
        _msg = 'Loading from checkpoint.'
        self.log_info(_msg, verbosity=0)

        _msg = 'Locating checkpoint file'
        self.log_info(_msg, verbosity=2)
        ckp_file = os.path.join(self.prefix, 'checkpoint.npz')
        checkpoint = np.load(ckp_file, allow_pickle=True)

        # write saved variables back to the model
        _msg = 'Loading variables into model'
        self.log_info(_msg, verbosity=2)

        self._time = float(checkpoint['time'])
        self.H_SL = float(checkpoint['H_SL'])
        self._time_iter = int(checkpoint['time_iter'])
        self._save_iter = int(checkpoint['save_iter'])
        self._save_time_since_last = int(checkpoint['save_time_since_last'])
        self.uw = checkpoint['uw']
        self.ux = checkpoint['ux']
        self.uy = checkpoint['uy']
        self.qw = checkpoint['qw']
        self.qx = checkpoint['qx']
        self.qy = checkpoint['qy']
        self.depth = checkpoint['depth']
        self.stage = checkpoint['stage']
        self.eta = checkpoint['eta']
        self.n_steps = checkpoint['n_steps']
        self.init_eta = checkpoint['init_eta']
        self.strata_counter = checkpoint['strata_counter']

        # load and set random state to continue as if run hadn't stopped
        _msg = 'Loading random state'
        self.log_info(_msg, verbosity=2)
        rng_state = tuple(checkpoint['rng_state'])
        shared_tools.set_random_state(rng_state)

        # reconstruct the strata arrays
        _msg = 'Loading stratigraphy arrays'
        self.log_info(_msg, verbosity=2)
        strata_eta_csr = csr_matrix((checkpoint['eta_data'],
                                    checkpoint['eta_indices'],
                                    checkpoint['eta_indptr']),
                                    shape=checkpoint['eta_shape'])
        self.strata_eta = strata_eta_csr.tolil()
        # get strata_sand_frac
        strata_sand_csr = csr_matrix((checkpoint['sand_data'],
                                     checkpoint['sand_indices'],
                                     checkpoint['sand_indptr']),
                                     shape=checkpoint['sand_shape'])
        self.strata_sand_frac = strata_sand_csr.tolil()

        # rename the old netCDF4 file
        _msg = 'Renaming old NetCDF4 output file'
        self.log_info(_msg, verbosity=2)
        file_path = os.path.join(self.prefix, 'pyDeltaRCM_output.nc')
        _tmp_name = os.path.join(self.prefix, 'old_pyDeltaRCM_output.nc')
        os.rename(file_path, _tmp_name)

        # write dims / attributes / variables to new netCDF file
        # except the things defined by output_strata()
        _msg = 'Creating NetCDF4 output file'
        self.log_info(_msg, verbosity=2)

        # list of things to not copy over
        dimtoignore = ['total_strata_age']
        vartoignore = ['strata_age', 'strata_sand_frac', 'strata_depth']

        # copy data from old netCDF4 into new one
        with Dataset(_tmp_name) as src, Dataset(file_path, 'w',
                                                format='NETCDF4') as dst:
            # copy attributes
            for name in src.ncattrs():
                dst.setncattr(name, src.getncattr(name))
            # copy dimensions
            for name, dimension in src.dimensions.items():
                if name not in dimtoignore:
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
                if name not in vartoignore:
                    dst.createVariable(name, variable.datatype,
                                       variable.dimensions)
                    dst.variables[name][:] = src.variables[name][:]

        _msg = 'Successfully loaded checkpoint and created new NetCDF file.'
        self.log_info(_msg, verbosity=1)

        # set object attribute for model
        self.output_netcdf = Dataset(file_path, 'r+', format='NETCDF4')

        # synch netcdf file
        self.output_netcdf.sync()

        # delete old netCDF4 file
        os.remove(_tmp_name)
