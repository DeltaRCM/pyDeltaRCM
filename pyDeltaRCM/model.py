#! /usr/bin/env python
import warnings
import logging
import datetime
import os

from .iteration_tools import iteration_tools
from .sed_tools import sed_tools
from .water_tools import water_tools
from .init_tools import init_tools
from .debug_tools import debug_tools


class DeltaModel(iteration_tools, sed_tools, water_tools,
                 init_tools, debug_tools, object):
    """Main model class.

    Instantiating the model class is described in the :meth:`__init__` method
    below in detail, but generally model instantiation occurs via a model run
    YAML configuration file. These YAML configuration files define model
    parameters which are used in the run; read more about creating input YAML
    configuration files in the :doc:`../../guides/userguide`.

    Once the model has been instantiated, the model is updated via the
    :meth:`update`. This method coordinates the hydrology, sediment transport,
    subsidence, and stratigraphic operation that must occur during each
    timestep. The :meth:`update` should be called iteratively, to "run" the
    model.

    Finally, after all ``update`` steps are complete, the model should be
    finalized (:meth:`finalize`), such that files and data are appropriately
    closed and written to disk.
    """

    def __init__(self, input_file=None):
        """Creates an instance of the pyDeltaRCM model.

        This method handles setting up the run, including parsing input files,
        initializing arrays, and initializing output routines.

        Parameters
        ----------
        input_file : `str`, `os.PathLike`, optional
            User model run configuration file.

        Returns
        -------

        Notes
        -----
        For more information regarding input configuration files, see the
        :doc:`../../guides/userguide`.

        """
        self._time = 0.
        self._time_iter = int(0)
        self._save_time_since_last = float("inf")  # force save on t==0
        self._save_iter = int(0)
        self._save_time_since_checkpoint = 0

        self.input_file = input_file
        _src_dir = os.path.realpath(os.path.dirname(__file__))
        self.default_file = os.path.join(_src_dir, 'default.yml')
        self.import_files()

        self.init_output_infrastructure()
        self.init_logger()

        self.process_input_to_model()
        self.determine_random_seed()

        self.create_other_variables()
        self.create_domain()

        self.init_sediment_routers()
        self.init_subsidence()

        # if resume flag set to True, load checkpoint, open netCDF4
        if self.resume_checkpoint:
            _msg = 'Loading data from checkpoint and reopening netCDF4 file.'
            self.logger.info(_msg)
            self.load_checkpoint()

        else:
            self.init_stratigraphy()
            self.init_output_file()

        self.logger.info('Model initialization complete')

    def update(self):
        """Run the model for one full instance

        Calls, in sequence:

            * the routine to run one timestep (i.e., water surface estimation
              and sediment routing, :meth:`run_one_timestep`)
            * the basin subsidence update pattern (:meth:`apply_subsidence`)
            * the timestep finalization routine (:meth:`finalize_timestep`)
            * straigraphy updating routine (:meth:`record_stratigraphy`)

        If you attempt to override the ``update`` routine, you must implement
        these operations at a minimum.

        Parameters
        ----------

        Returns
        -------

        """
        if self._save_time_since_last >= self.save_dt:
            self.record_stratigraphy()
            self.output_data()
            self._save_iter += int(1)
            self._save_time_since_last = 0

        self.run_one_timestep()
        self.apply_subsidence()
        self.finalize_timestep()

        self._time += self.dt
        self._save_time_since_last += self.dt
        self._save_time_since_checkpoint += self.dt
        self._time_iter += int(1)

        if self._save_time_since_checkpoint >= self.checkpoint_dt:
            self.output_checkpoint()
            self._save_time_since_checkpoint = 0

    def finalize(self):
        """Finalize the model run.

        Finalization includes saving output stratigraphy, closing relevant
        files on disk and clearing variables.

        Parameters
        ----------

        Returns
        -------

        """
        self.logger.info('Finalize model run')

        # get the final timestep recorded, if needed.
        if self._save_time_since_last >= self.save_dt:
            self.record_stratigraphy()
            self.output_data()

        if self._save_time_since_checkpoint >= self.checkpoint_dt:
            self.output_checkpoint()

        self.output_strata()

        try:
            self.output_netcdf.close()
            _msg = 'Closed output netcdf file'
            self.logger.info(_msg)
            if self.verbose >= 2:
                print(_msg)
        except Exception:
            pass

        self._is_finalized = True

    @property
    def out_dir(self):
        """
        Description of out_dir ...
        Must be a string.
        """
        return self._out_dir

    @out_dir.setter
    def out_dir(self, out_dir):
        self._out_dir = out_dir

    @property
    def verbose(self):
        """
        Desciption of verbose ...
        """
        return self._verbose

    @verbose.setter
    def verbose(self, verbose):
        self._verbose = verbose

    @property
    def seed(self):
        """
        seed
        """
        return self._seed

    @seed.setter
    def seed(self, seed):
        self._seed = seed

    @property
    def Length(self):
        """
        Length
        """
        return self._Length

    @Length.setter
    def Length(self, Length):
        if Length <= 0:
            raise ValueError('Length must be a positive number.')
        self._Length = Length

    @property
    def Width(self):
        """
        Width
        """
        return self._Width

    @Width.setter
    def Width(self, Width):
        if Width <= 0:
            raise ValueError('Width must be a positive number.')
        self._Width = Width

    @property
    def dx(self):
        """
        dx
        """
        return self._dx

    @dx.setter
    def dx(self, dx):
        if (dx <= 0) or (dx > self._Length) or (dx > self._Width):
            raise ValueError('dx must be positive and smaller than the' +
                             ' Length and Width parameters.')
        self._dx = dx

    @property
    def L0_meters(self):
        """
        L0 meters
        """
        return self._L0_meters

    @L0_meters.setter
    def L0_meters(self, L0_meters):
        if L0_meters <= 0:
            raise ValueError('L0_meters must be a positive number.')
        self._L0_meters = L0_meters

    @property
    def S0(self):
        """
        S0
        """
        return self._S0

    @S0.setter
    def S0(self, S0):
        self._S0 = S0

    @property
    def itermax(self):
        """
        itermax
        """
        return self._itermax

    @itermax.setter
    def itermax(self, itermax):
        if itermax < 0:
            raise ValueError('itermax must be greater than or equal to 0.')
        self._itermax = itermax

    @property
    def Np_water(self):
        """
        np water
        """
        return self._Np_water

    @Np_water.setter
    def Np_water(self, Np_water):
        if Np_water <= 0:
            raise ValueError('Np_water must be a positive number.')
        self._Np_water = Np_water

    @property
    def u0(self):
        """
        u0
        """
        return self._u0

    @u0.setter
    def u0(self, u0):
        self._u0 = u0

    @property
    def N0_meters(self):
        """
        N0_meters
        """
        return self._N0_meters

    @N0_meters.setter
    def N0_meters(self, N0_meters):
        if N0_meters <= 0:
            raise ValueError('N0_meters must be a positive number.')
        self._N0_meters = N0_meters

    @property
    def h0(self):
        """
        h0
        """
        return self._h0

    @h0.setter
    def h0(self, h0):
        self._h0 = h0

    @property
    def H_SL(self):
        """
        H_SL
        """
        return self._H_SL

    @H_SL.setter
    def H_SL(self, H_SL):
        self._H_SL = H_SL

    @property
    def SLR(self):
        """
        SLR
        """
        return self._SLR

    @SLR.setter
    def SLR(self, SLR):
        self._SLR = SLR

    @property
    def Np_sed(self):
        """
        Np_sed
        """
        return self._Np_sed

    @Np_sed.setter
    def Np_sed(self, Np_sed):
        if Np_sed <= 0:
            raise ValueError('Np_sed must be a positive number.')
        self._Np_sed = Np_sed

    @property
    def f_bedload(self):
        """
        f_bedload
        """
        return self._f_bedload

    @f_bedload.setter
    def f_bedload(self, f_bedload):
        if (f_bedload < 0) or (f_bedload > 1):
            raise ValueError('Value for f_bedload must be between 0 and 1,'
                             ' inclusive.')
        self._f_bedload = f_bedload

    @property
    def C0_percent(self):
        """
        C0_percent
        """
        return self._C0_percent

    @C0_percent.setter
    def C0_percent(self, C0_percent):
        if C0_percent < 0:
            raise ValueError('C0_percent must be greater than or equal to 0.')
        self._C0_percent = C0_percent

    @property
    def Csmooth(self):
        """
        Csmooth
        """
        return self._Csmooth

    @Csmooth.setter
    def Csmooth(self, Csmooth):
        if Csmooth < 0:
            raise ValueError('Csmooth must be greater than or equal to 0.')
        self._Csmooth = Csmooth

    @property
    def toggle_subsidence(self):
        """
        toggle_subsidence
        """
        return self._toggle_subsidence

    @toggle_subsidence.setter
    def toggle_subsidence(self, toggle_subsidence):
        self._toggle_subsidence = toggle_subsidence

    @property
    def theta1(self):
        """
        theta1
        """
        return self._theta1

    @theta1.setter
    def theta1(self, theta1):
        self._theta1 = theta1

    @property
    def theta2(self):
        """
        theta2
        """
        return self._theta2

    @theta2.setter
    def theta2(self, theta2):
        self._theta2 = theta2

    @property
    def sigma_max(self):
        """
        sigma_max
        """
        return self._sigma_max

    @sigma_max.setter
    def sigma_max(self, sigma_max):
        self._sigma_max = sigma_max

    @property
    def start_subsidence(self):
        """
        start_subsidence
        """
        return self._start_subsidence

    @start_subsidence.setter
    def start_subsidence(self, start_subsidence):
        self._start_subsidence = start_subsidence

    @property
    def save_eta_figs(self):
        """
        save_eta_figs
        """
        return self._save_eta_figs

    @save_eta_figs.setter
    def save_eta_figs(self, save_eta_figs):
        self._save_eta_figs = save_eta_figs

    @property
    def save_stage_figs(self):
        """
        save_stage_figs
        """
        return self._save_stage_figs

    @save_stage_figs.setter
    def save_stage_figs(self, save_stage_figs):
        self._save_stage_figs = save_stage_figs

    @property
    def save_depth_figs(self):
        """
        save_depth_figs
        """
        return self._save_depth_figs

    @save_depth_figs.setter
    def save_depth_figs(self, save_depth_figs):
        self._save_depth_figs = save_depth_figs

    @property
    def save_discharge_figs(self):
        """
        save_discharge_figs
        """
        return self._save_discharge_figs

    @save_discharge_figs.setter
    def save_discharge_figs(self, save_discharge_figs):
        self._save_discharge_figs = save_discharge_figs

    @property
    def save_velocity_figs(self):
        """
        save_velocity_figs
        """
        return self._save_velocity_figs

    @save_velocity_figs.setter
    def save_velocity_figs(self, save_velocity_figs):
        self._save_velocity_figs = save_velocity_figs

    @property
    def save_sedflux_figs(self):
        """
        save_sedflux_figs
        """
        return self._save_sedflux_figs

    @save_sedflux_figs.setter
    def save_sedflux_figs(self, save_sedflux_figs):
        self._save_sedflux_figs = save_sedflux_figs

    @property
    def save_figs_sequential(self):
        """
        save_figs_sequential
        """
        return self._save_figs_sequential

    @save_figs_sequential.setter
    def save_figs_sequential(self, save_figs_sequential):
        self._save_figs_sequential = save_figs_sequential

    @property
    def save_metadata(self):
        """
        save_metadata
        """
        return self._save_metadata

    @save_metadata.setter
    def save_metadata(self, save_metadata):
        self._save_metadata = save_metadata

    @property
    def save_eta_grids(self):
        """
        save_eta_grids
        """
        return self._save_eta_grids

    @save_eta_grids.setter
    def save_eta_grids(self, save_eta_grids):
        self._save_eta_grids = save_eta_grids

    @property
    def save_stage_grids(self):
        """
        save_stage_grids
        """
        return self._save_stage_grids

    @save_stage_grids.setter
    def save_stage_grids(self, save_stage_grids):
        self._save_stage_grids = save_stage_grids

    @property
    def save_depth_grids(self):
        """
        save_depth_grids
        """
        return self._save_depth_grids

    @save_depth_grids.setter
    def save_depth_grids(self, save_depth_grids):
        self._save_depth_grids = save_depth_grids

    @property
    def save_discharge_grids(self):
        """
        save_discharge_grids
        """
        return self._save_discharge_grids

    @save_discharge_grids.setter
    def save_discharge_grids(self, save_discharge_grids):
        self._save_discharge_grids = save_discharge_grids

    @property
    def save_velocity_grids(self):
        """
        save_velocity_grids
        """
        return self._save_velocity_grids

    @save_velocity_grids.setter
    def save_velocity_grids(self, save_velocity_grids):
        self._save_velocity_grids = save_velocity_grids

    @property
    def save_sedflux_grids(self):
        """
        save_sedflux_grids
        """
        return self._save_sedflux_grids

    @save_sedflux_grids.setter
    def save_sedflux_grids(self, save_sedflux_grids):
        self._save_sedflux_grids = save_sedflux_grids

    @property
    def save_dt(self):
        """
        save_dt
        """
        return self._save_dt

    @save_dt.setter
    def save_dt(self, save_dt):
        self._save_dt = save_dt

    @property
    def checkpoint_dt(self):
        """
        checkpoint_dt
        """
        return self._checkpoint_dt

    @checkpoint_dt.setter
    def checkpoint_dt(self, checkpoint_dt):
        self._checkpoint_dt = checkpoint_dt

    @property
    def save_strata(self):
        """
        save_strata
        """
        return self._save_strata

    @save_strata.setter
    def save_strata(self, save_strata):
        self._save_strata = save_strata

    @property
    def save_checkpoint(self):
        """
        save_checkpoint
        """
        return self._save_checkpoint

    @save_checkpoint.setter
    def save_checkpoint(self, save_checkpoint):
        self._save_checkpoint = save_checkpoint

    @property
    def resume_checkpoint(self):
        """
        resume_checkpoint
        """
        return self._resume_checkpoint

    @resume_checkpoint.setter
    def resume_checkpoint(self, resume_checkpoint):
        self._resume_checkpoint = resume_checkpoint

    @property
    def omega_sfc(self):
        """
        omega_sfc
        """
        return self._omega_sfc

    @omega_sfc.setter
    def omega_sfc(self, omega_sfc):
        self._omega_sfc = omega_sfc

    @property
    def omega_flow(self):
        """
        omega_flow
        """
        return self._omega_flow

    @omega_flow.setter
    def omega_flow(self, omega_flow):
        self._omega_flow = omega_flow

    @property
    def Nsmooth(self):
        """
        Nsmooth
        """
        return self._Nsmooth

    @Nsmooth.setter
    def Nsmooth(self, Nsmooth):
        self._Nsmooth = Nsmooth

    @property
    def theta_water(self):
        """
        theta_water
        """
        return self._theta_water

    @theta_water.setter
    def theta_water(self, theta_water):
        self._theta_water = theta_water

    @property
    def coeff_theta_sand(self):
        """
        coeff_theta_sand
        """
        return self._coeff_theta_sand

    @coeff_theta_sand.setter
    def coeff_theta_sand(self, coeff_theta_sand):
        self._coeff_theta_sand = coeff_theta_sand

    @property
    def coeff_theta_mud(self):
        """
        coeff_theta_mud
        """
        return self._coeff_theta_mud

    @coeff_theta_mud.setter
    def coeff_theta_mud(self, coeff_theta_mud):
        self._coeff_theta_mud = coeff_theta_mud

    @property
    def beta(self):
        """
        beta
        """
        return self._beta

    @beta.setter
    def beta(self, beta):
        self._beta = beta

    @property
    def sed_lag(self):
        """
        sed_lag
        """
        return self._sed_lag

    @sed_lag.setter
    def sed_lag(self, sed_lag):
        self._sed_lag = sed_lag

    @property
    def coeff_U_dep_mud(self):
        """
        coeff_U_dep_mud
        """
        return self._coeff_U_dep_mud

    @coeff_U_dep_mud.setter
    def coeff_U_dep_mud(self, coeff_U_dep_mud):
        self._coeff_U_dep_mud = coeff_U_dep_mud

    @property
    def coeff_U_ero_mud(self):
        """
        coeff_U_ero_mud
        """
        return self._coeff_U_ero_mud

    @coeff_U_ero_mud.setter
    def coeff_U_ero_mud(self, coeff_U_ero_mud):
        self._coeff_U_ero_mud = coeff_U_ero_mud

    @property
    def coeff_U_ero_sand(self):
        """
        coeff_U_ero_sand
        """
        return self._coeff_U_ero_sand

    @coeff_U_ero_sand.setter
    def coeff_U_ero_sand(self, coeff_U_ero_sand):
        self._coeff_U_ero_sand = coeff_U_ero_sand

    @property
    def alpha(self):
        """
        alpha
        """
        return self._alpha

    @alpha.setter
    def alpha(self, alpha):
        self._alpha = alpha

    @property
    def time(self):
        """Elapsed model time in seconds.
        """
        return self._time

    @property
    def dt(self):
        """The time step.

        Raises
        ------
        UserWarning
            If a very small timestep is configured.
        """
        return self._dt

    @property
    def time_step(self):
        """Alias for `dt`.
        """
        return self._dt

    @time_step.setter
    def time_step(self, new_time_step):
        if new_time_step * self.init_Np_sed < 100:
            warnings.warn(UserWarning('Using a very small time step, '
                                      'Delta might evolve very slowly.'))

        if self.toggle_subsidence:
            self.sigma = self.subsidence_mask * self.sigma_max * new_time_step

        self._dt = new_time_step

    @property
    def time_iter(self):
        """Number of time iterations.

        The number of times the :obj:`update` method has been called.
        """
        return self._time_iter

    @property
    def save_time_since_last(self):
        """Time since data last output.

        The number of times the :obj:`update` method has been called.
        """
        return self._save_time_since_last

    @property
    def save_time_since_checkpoint(self):
        """Time since last data checkpoint was saved."""
        return self._save_time_since_checkpoint

    @property
    def save_iter(self):
        """Number of times data has been saved."""
        return self._save_iter

    @property
    def channel_flow_velocity(self):
        """Get channel flow velocity."""
        return self.u0

    @channel_flow_velocity.setter
    def channel_flow_velocity(self, new_u0):
        self.u0 = new_u0
        self.create_other_variables()
        self.init_sediment_routers()

    @property
    def channel_width(self):
        """Get channel width."""
        return self.N0_meters

    @channel_width.setter
    def channel_width(self, new_N0):
        self.N0_meters = new_N0
        self.create_other_variables()
        self.init_sediment_routers()

    @property
    def channel_flow_depth(self):
        """Get channel flow depth."""
        return self.h0

    @channel_flow_depth.setter
    def channel_flow_depth(self, new_d):
        self.h0 = new_d
        self.create_other_variables()
        self.init_sediment_routers()

    @property
    def sea_surface_mean_elevation(self):
        """Get sea surface mean elevation."""
        return self.H_SL

    @sea_surface_mean_elevation.setter
    def sea_surface_mean_elevation(self, new_se):
        self.H_SL = new_se

    @property
    def sea_surface_elevation_change(self):
        """Get rate of change of sea surface elevation, per timestep."""
        return self.SLR

    @sea_surface_elevation_change.setter
    def sea_surface_elevation_change(self, new_SLR):
        self.SLR = new_SLR

    @property
    def bedload_fraction(self):
        """Get bedload fraction."""
        return self.f_bedload

    @bedload_fraction.setter
    def bedload_fraction(self, new_u0):
        self.f_bedload = new_u0

    @property
    def influx_sediment_concentration(self):
        """Get influx sediment concentration."""
        return self.C0_percent

    @influx_sediment_concentration.setter
    def influx_sediment_concentration(self, new_u0):
        self.C0_percent = new_u0
        self.create_other_variables()
        self.init_sediment_routers()

    @property
    def sea_surface_elevation(self):
        """Get stage."""
        return self.stage

    @property
    def water_depth(self):
        """Get depth."""
        return self.depth

    @property
    def bed_elevation(self):
        """Get bed elevation."""
        return self.eta
