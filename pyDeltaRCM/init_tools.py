
import os
import logging
import warnings

from math import floor, sqrt, pi
import numpy as np

from scipy.sparse import lil_matrix
from scipy import ndimage

from netCDF4 import Dataset
import time as time_lib
import yaml
import re

from . import shared_tools

# tools for initiating deltaRCM model domain


class init_tools(object):

    def init_output_infrastructure(self):

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
        self.logger = logging.getLogger(self.prefix_abspath+timestamp)
        self.logger.setLevel(logging.INFO)

        # create the logging file handler
        fh = logging.FileHandler(
            self.prefix_abspath + '/pyDeltaRCM_' + timestamp + '.log')
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)

        # add handler to logger object
        self.logger.addHandler(fh)

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

        # now make each one an attribute of the model object.
        for k, v in list(input_file_vars.items()):
            setattr(self, k, v)

    def determine_random_seed(self):
        """Set the random seed if given.

        If a random seed is specified, set the seed to this value.

        Writes the seed to the log for record.
        """
        if self.seed is None:
            # generate a random seed for reproducibility
            self.seed = np.random.randint((2**32) - 1, dtype='u8')

        _msg = 'Setting random seed to: %s ' % str(self.seed)
        self.logger.info(_msg)
        if self.verbose >= 2:
            print(_msg)

        shared_tools.set_random_seed(self.seed)

        # always write the seed to file for record and reproducability
        self.logger.info('Random seed is: %s ' % str(self.seed))

    def set_constants(self):

        self.logger.info('Setting model constants')

        self.g = 9.81   # (gravitation const.)

        sqrt2 = np.sqrt(2)
        self.distances = np.array([[sqrt2, 1, sqrt2],
                                   [1, 1, 1],
                                   [sqrt2, 1, sqrt2]])

        sqrt05 = np.sqrt(0.5)
        self.ivec = np.array([[-sqrt05, 0, sqrt05],
                              [-1, 0, 1],
                              [-sqrt05, 0, sqrt05]])

        self.iwalk = shared_tools.get_iwalk()

        self.jvec = np.array([[-sqrt05, -1, -sqrt05],
                              [0, 0, 0],
                              [sqrt05, 1, sqrt05]])

        self.jwalk = shared_tools.get_jwalk()
        self.dxn_iwalk = [1, 1, 0, -1, -1, -1, 0, 1]
        self.dxn_jwalk = [0, 1, 1, 1, 0, -1, -1, -1]
        self.dxn_dist = \
            [sqrt(self.dxn_iwalk[i]**2 + self.dxn_jwalk[i]**2)
             for i in range(8)]

        self.walk_flat = np.array([1, -self.W + 1, -self.W, -self.W - 1,
                                   -1, self.W - 1, self.W, self.W + 1])

        self.kernel1 = np.array([[1, 1, 1],
                                 [1, -8, 1],
                                 [1, 1, 1]])

        self.kernel2 = np.array([[1, 1, 1],
                                 [1, 0, 1],
                                 [1, 1, 1]])

    def create_other_variables(self):
        """Model implementation variables.

        Creates variables for model implementation, from specified boundary
        condition variables. This method is run during initial model
        instantition, but it is also run any time an inlet flow condition
        variable is changed, including `channel_flow_velocity`,
        `channel_width`, `channel_flow_depth`, and
        `influx_sediment_concentration`.
        """
        self.init_Np_water = self.Np_water
        self.init_Np_sed = self.Np_sed

        self.dx = float(self.dx)

        self.theta_sand = self.coeff_theta_sand * self.theta_water
        self.theta_mud = self.coeff_theta_mud * self.theta_water

        self.U_dep_mud = self.coeff_U_dep_mud * self.u0
        self.U_ero_sand = self.coeff_U_ero_sand * self.u0
        self.U_ero_mud = self.coeff_U_ero_mud * self.u0

        self.L = int(round(self.Length / self.dx))        # num cells in x
        self.W = int(round(self.Width / self.dx))         # num cells in y

        # inlet length and width
        self.L0 = max(1, min(int(round(self.L0_meters / self.dx)), self.L // 4))
        self.N0 = max(3, min(int(round(self.N0_meters / self.dx)), self.W // 4))

        self.set_constants()

        self.u_max = 2.0 * self.u0  # maximum allowed flow velocity
        self.C0 = self.C0_percent * 1 / 100.  # sediment concentration

        self.dry_depth = min(0.1, 0.1 * self.h0)  # (m) critial depth to switch to "dry" node
        self.CTR = floor(self.W / 2.) - 1
        if self.CTR <= 1:
            self.CTR = floor(self.W / 2.)

        self.gamma = self.g * self.S0 * self.dx / (self.u0**2)

        self.V0 = self.h0 * (self.dx**2)  # (m^3) reference volume, volume to fill cell to characteristic depth
        self.Qw0 = self.u0 * self.h0 * self.N0 * self.dx    # const discharge

        # at inlet
        self.qw0 = self.u0 * self.h0  # water unit input discharge
        self.Qp_water = self.Qw0 / self.Np_water    # volume each water parcel
        self.qs0 = self.qw0 * self.C0  # sed unit discharge
        self.dVs = 0.1 * self.N0**2 * self.V0  # total amount of sed added to domain per timestep
        self.Qs0 = self.Qw0 * self.C0  # sediment total input discharge
        self.Vp_sed = self.dVs / self.Np_sed   # volume of each sediment parcel

        self.itmax = 2 * (self.L + self.W)  # max number of jumps for parcel
        self.size_indices = int(self.itmax / 2)  # initial width of self.indices

        self.dt = self.dVs / self.Qs0  # time step size

        self.omega_flow_iter = 2. / self.itermax

        self.N_crossdiff = int(round(self.dVs / self.V0))  # number of times to repeat topo diffusion

        self._lambda = self.sed_lag  # sedimentation lag

        self.diffusion_multiplier = (self.dt / self.N_crossdiff * self.alpha
                                     * 0.5 / self.dx**2)

        self._save_any_grids = (self.save_eta_grids or self.save_depth_grids or
                                self.save_stage_grids or self.save_discharge_grids or
                                self.save_velocity_grids)
        self._save_any_figs = (self.save_eta_figs or self.save_depth_figs or
                               self.save_stage_figs or self.save_discharge_figs or
                               self.save_velocity_figs)
        self._save_figs_sequential = self.save_figs_sequential  # copy as private
        self._is_finalized = False

    def create_domain(self):
        """
        Creates the model domain
        """
        self.logger.info('Creating model domain')

        # ---- empty arrays ----
        self.x, self.y = np.meshgrid(
            np.arange(0, self.W), np.arange(0, self.L))

        self.cell_type = np.zeros((self.L, self.W), dtype=np.int)
        self.eta = np.zeros((self.L, self.W)).astype(np.float32)
        self.stage = np.zeros((self.L, self.W)).astype(np.float32)
        self.depth = np.zeros((self.L, self.W)).astype(np.float32)
        self.qx = np.zeros((self.L, self.W))
        self.qy = np.zeros((self.L, self.W))
        self.qxn = np.zeros((self.L, self.W))
        self.qyn = np.zeros((self.L, self.W))
        self.qwn = np.zeros((self.L, self.W))
        self.ux = np.zeros((self.L, self.W))
        self.uy = np.zeros((self.L, self.W))
        self.uw = np.zeros((self.L, self.W))
        self.qs = np.zeros((self.L, self.W))
        self.Vp_dep_sand = np.zeros((self.L, self.W))
        self.Vp_dep_mud = np.zeros((self.L, self.W))
        self.free_surf_flag = np.zeros((self.Np_water,), dtype=np.int)
        self.looped = np.zeros((self.Np_water,))
        self.indices = np.zeros((self.Np_water, self.size_indices),
                                dtype=np.int)
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

        self.stage[:] = np.maximum(0, self.L0 - self.y - 1) * self.dx * self.S0
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

        self.inlet = list(np.unique(np.where(self.cell_type == 1)[1]))
        self.eta[:] = self.stage - self.depth

    def init_stratigraphy(self):
        """Creates sparse array to store stratigraphy data."""
        if self.save_strata:

            self.strata_counter = 0

            self.n_steps = 5 * self.save_dt

            self.strata_sand_frac = lil_matrix((self.L * self.W, self.n_steps),
                                               dtype=np.float32)

            self.init_eta = self.eta.copy()
            self.strata_eta = lil_matrix((self.L * self.W, self.n_steps),
                                         dtype=np.float32)

    def init_output_grids(self):
        """Creates a netCDF file to store output grids.

        Fills with default variables.

        .. warning:: Overwrites an existing netcdf file with the same name.

        """
        if (self.save_eta_grids or
                self.save_depth_grids or
                self.save_stage_grids or
                self.save_discharge_grids or
                self.save_velocity_grids or
                self.save_strata):

            _msg = 'Generating netCDF file for output grids'
            self.logger.info(_msg)
            if self.verbose >= 2:
                print(_msg)

            directory = self.prefix
            filename = 'pyDeltaRCM_output.nc'

            file_path = os.path.join(directory, filename)
            if os.path.exists(file_path):
                _msg = 'Replacing existing netCDF file'
                self.logger.warning(_msg)
                if self.verbose >= 2:
                    warnings.warn(UserWarning(_msg))
                os.remove(file_path)

            self.output_netcdf = Dataset(file_path, 'w',
                                         format='NETCDF4_CLASSIC')

            self.output_netcdf.description = 'Output grids from pyDeltaRCM'
            self.output_netcdf.history = ('Created '
                                          + time_lib.ctime(time_lib.time()))
            self.output_netcdf.source = 'pyDeltaRCM / CSDMS'

            length = self.output_netcdf.createDimension('length', self.L)
            width = self.output_netcdf.createDimension('width', self.W)
            total_time = self.output_netcdf.createDimension('total_time', None)

            x = self.output_netcdf.createVariable(
                'x', 'f4', ('length', 'width'))
            y = self.output_netcdf.createVariable(
                'y', 'f4', ('length', 'width'))
            time = self.output_netcdf.createVariable('time', 'f4',
                                                     ('total_time',))
            x.units = 'meters'
            y.units = 'meters'
            time.units = 'timesteps'

            x[:] = self.x
            y[:] = self.y

            if self.save_eta_grids:
                eta = self.output_netcdf.createVariable('eta',
                                                        'f4',
                                                        ('total_time', 'length', 'width'))
                eta.units = 'meters'

            if self.save_stage_grids:
                stage = self.output_netcdf.createVariable('stage',
                                                          'f4',
                                                          ('total_time', 'length', 'width'))
                stage.units = 'meters'

            if self.save_depth_grids:
                depth = self.output_netcdf.createVariable('depth',
                                                          'f4',
                                                          ('total_time', 'length', 'width'))
                depth.units = 'meters'

            if self.save_discharge_grids:
                discharge = self.output_netcdf.createVariable('discharge',
                                                              'f4',
                                                              ('total_time', 'length', 'width'))
                discharge.units = 'cubic meters per second'

            if self.save_velocity_grids:
                velocity = self.output_netcdf.createVariable('velocity',
                                                             'f4',
                                                             ('total_time', 'length', 'width'))
                velocity.units = 'meters per second'

            _msg = 'Output netCDF file created'
            self.logger.info(_msg)
            if self.verbose >= 2:
                print(_msg)

    def init_subsidence(self):
        """
        Initializes patterns of subsidence if
        toggle_subsidence is True (default False)

        Modify the equations for self.subsidence_mask and self.sigma as desired
        """
        if self.toggle_subsidence:

            R1 = 0.3 * self.L
            R2 = 1. * self.L  # radial limits (fractions of L)
            theta1 = -pi / 3
            theta2 = pi / 3.   # angular limits

            Rloc = np.sqrt((self.y - self.L0)**2 + (self.x - self.W / 2.)**2)

            thetaloc = np.zeros((self.L, self.W))
            thetaloc[self.y > self.L0 - 1] = np.arctan(
                (self.x[self.y > self.L0 - 1] - self.W / 2.)
                / (self.y[self.y > self.L0 - 1] - self.L0 + 1))
            self.subsidence_mask = ((R1 <= Rloc) & (Rloc <= R2) &
                                    (theta1 <= thetaloc) & (thetaloc <= theta2))
            self.subsidence_mask[:self.L0, :] = False

            self.sigma = self.subsidence_mask * self.sigma_max * self.dt
