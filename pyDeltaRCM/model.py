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

    def __init__(self, input_file=None, defer_output=False):
        """Creates an instance of the pyDeltaRCM model.

        This method handles setting up the run, including parsing input files,
        initializing arrays, and initializing output routines.

        Parameters
        ----------
        input_file : `str`, `os.PathLike`, optional
            User model run configuration file.

        defer_output : `bool`, optional
            Whether to create the output netCDF file during initialization. In
            most cases, this can be ignored and left to default value `False`.
            However, for parallel simulations it may be necessary to defer the
            NetCDF file creation unitl the simualtion is assigned to the core
            it will compute on, so that the DeltaModel object remains
            pickle-able. Note, you will need to manually trigger the
            :obj:`init_output_file` method if `defer_output` is `True`.

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
            _msg = 'Loading from checkpoint.'
            self.log_info(_msg)
            self.load_checkpoint()

        else:
            self.init_stratigraphy()

        # always re-init output file, will clobber when checkpointing
        if not defer_output:
            self.init_output_file()

        _msg = 'Model initialization complete'
        self.log_info(_msg)
        self.log_model_time()

    def update(self):
        """Run the model for one full instance.

        This method handles the input/output from the model, orchestrating the
        various morphodynamic and basin-scale processes, and incrementing the
        model time-tracking attributes. This method calls, in sequence:

            * the routine to run one timestep (i.e., water surface estimation
              and sediment routing, :meth:`run_one_timestep`)
            * the basin subsidence update pattern (:meth:`apply_subsidence`)
            * the timestep finalization routine (:meth:`finalize_timestep`)
            * straigraphy updating routine (:meth:`record_stratigraphy`)

        If you attempt to override the ``update`` routine, you must implement
        these operations at a minimum. More likely, you can implement what you
        need to by just overriding one of the methods called by `update`.

        Parameters
        ----------

        Returns
        -------

        """
        # record the state of the model
        if self._save_time_since_last >= self.save_dt:
            self.record_stratigraphy()
            self.output_data()
            self._save_iter += int(1)
            self._save_time_since_last = 0

        # update the model, i.e., the actual model morphodynamics
        self.run_one_timestep()
        self.apply_subsidence()
        self.finalize_timestep()

        # update time-tracking fields
        self._time += self.dt
        self._save_time_since_last += self.dt
        self._save_time_since_checkpoint += self.dt
        self._time_iter += int(1)
        self.log_model_time()

        # save a checkpoint if needed
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
        _msg = 'Finalize the model run'
        self.log_info(_msg)

        # get the final timestep recorded, if needed.
        if self._save_time_since_last >= self.save_dt:
            self.record_stratigraphy()
            self.output_data()

        if self._save_time_since_checkpoint >= self.checkpoint_dt:
            self.output_checkpoint()

        self.output_strata()

        try:
            self.output_netcdf.close()
            _msg = 'Closed output NetCDF4 file'
            self.log_info(_msg, verbosity=1)
        except Exception as e:
            self.logger.error('Failed to close output NetCDF4 file')
            self.logger.exception(e)

        self._is_finalized = True

    @property
    def out_dir(self):
        """
        out_dir sets the output directory for the simulation results.

        out_dir is a *string* type parameter, specifying the name of the output
        directory in which the model outputs should be saved.
        """
        return self._out_dir

    @out_dir.setter
    def out_dir(self, out_dir):
        self._out_dir = out_dir

    @property
    def verbose(self):
        """
        verbose controls the degree of information printed to the log.

        verbose is an *integer* type parameter, which controls the verbosity
        of the information saved in the log file. A value of 0, the default,
        is the least informative. A value of 1 saves and prints a bit of
        information, and a value of 2 increases the verbosity further.
        """
        return self._verbose

    @verbose.setter
    def verbose(self, verbose):
        self._verbose = verbose

    @property
    def seed(self):
        """
        seed defines the random number seed used for the simulation.

        seed is an *integer* type parameter specifying the random seed value to
        be used for this model run. If unspecified, a random seed is generated
        and used.
        """
        return self._seed

    @seed.setter
    def seed(self, seed):
        self._seed = seed

    @property
    def Length(self):
        """
        Length sets the total length of the domain.

        Length is either an *integer* or a *float*.
        This is the length of the domain (dimension parallel to the inlet
        channel), in **meters**.
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
        Width sets the total width of the domain.

        Width is either an *integer* or a *float*.
        This is the width of the domain (dimension perpendicular to the inlet
        channel), in **meters**.
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
        dx is the length of the individual cell faces.

        dx is either an *integer* or a *float*.
        This parameter specifies the length of the cell faces in the grid in
        **meters**.
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
        L0_meters defines the thickness of the land along the coast.

        L0 meters is either an *integer* or a *float*.
        Thickness of the land adjacent to the inlet in **meters**.
        This can also be thought of as the length of the inlet channel.
        """
        return self._L0_meters

    @L0_meters.setter
    def L0_meters(self, L0_meters):
        if L0_meters < 0:
            raise ValueError('L0_meters must be a greater than or equal to 0.')
        self._L0_meters = L0_meters

    @property
    def S0(self):
        """
        S0 is the characteristic slope for the delta.

        S0 is either an *integer* or a *float*.
        This sets the characteristic slope for the delta topset.
        This parameter is dimensionless.
        """
        return self._S0

    @S0.setter
    def S0(self, S0):
        self._S0 = S0

    @property
    def itermax(self):
        """
        itermax sets the number of flow routing/free surface iterations.
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
        Np_water is the number of water parcels simulated.

        Np_water represents the number of "parcels" to split the input water
        discharge into for the reduced-complexity flow routing.
        This parameter must be an *integer*
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
        u0 is a reference velocity value.

        u0 is the characteristic or reference velocity value in units of m/s.
        u0 influences the values of other model parameters such as the maximum
        flow velocity, gamma, and the velocities at which sediments deposit and
        erode. This parameter must be an *integer* or a *float*.
        """
        return self._u0

    @u0.setter
    def u0(self, u0):
        self._u0 = u0

    @property
    def N0_meters(self):
        """
        N0_meters defines the width of the inlet channel in meters.

        N0_meters defines the width of the inlet channel in meters. Therefore,
        this parameter must be a positive *integer* or *float*.
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
        h0 is the reference or characteristic water depth in meters.

        h0 is the reference or characteristic water depth in meters. This
        parameter must be an *integer* or *float*. h0 defines
        the depth of water in the inlet channel. One-tenth of the value of h0
        defines the "dry-depth" or the depth at which cells are considered to
        be non-wet (dry).
        """
        return self._h0

    @h0.setter
    def h0(self, h0):
        self._h0 = h0

    @property
    def H_SL(self):
        """
        H_SL sets the sea level elevation in meters.
        """
        return self._H_SL

    @H_SL.setter
    def H_SL(self, H_SL):
        self._H_SL = H_SL

    @property
    def SLR(self):
        """
        SLR is the sea level rise rate.

        SLR is a parameter for defining the sea level rise rate. SLR is
        specified in units of m/s. When prescribing a SLR rate, it is important
        to remember that pyDeltaRCM simulates bankfull discharge conditions.
        Depending on the flood intermittancy intervals you assume, the
        conversion from "model time" into "real time simulated" may vary.
        """
        return self._SLR

    @SLR.setter
    def SLR(self, SLR):
        self._SLR = SLR

    @property
    def If(self):
        """
        Intermittency factor converting model time to real-world time.

        .. important::

            This parameter has no effect on model operations. It is used in
            determining model run duration.
        """
        return self._If

    @If.setter
    def If(self, If):
        if (If <= 0) or (If > 1):
            raise ValueError('If must be in interval (0, 1].')
        self._If = If

    @property
    def Np_sed(self):
        """
        Np_sed is the number of sediment parcels simulated.

        Np_sed represents the number of "parcels" to split the input sediment
        discharge into for the reduced-complexity sediment routing.
        This parameter must be an *integer*
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
        f_bedload is the bedload fraction of the input sediment.

        f_bedload is the fraction of input sediment that is bedload material.
        In pyDeltaRCM, bedload material is coarse grained "sand", and
        suspended load material is fine grained "mud". This parameter must be
        a value between 0 and 1, inclusive.
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
        C0_percent is the sediment concentration in the input water supply.

        C0_percent is the prescribed sediment concentration in the input water
        as a percentage (must be equal to or greater than 0).
        """
        return self._C0_percent

    @C0_percent.setter
    def C0_percent(self, C0_percent):
        if C0_percent < 0:
            raise ValueError('C0_percent must be greater than 0.')
        self._C0_percent = C0_percent

    @property
    def Csmooth(self):
        """
        Csmooth is a free surface smoothing parameter.
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
        toggle_subsidence controls whether subsidence is turned on or off.

        If toggle_subsidence is set to `True` then subsidence is turned on.
        Otherwise if toggle_subsidence is set to `False` (the default) then no
        subsidence will occur.
        """
        return self._toggle_subsidence

    @toggle_subsidence.setter
    def toggle_subsidence(self, toggle_subsidence):
        self._toggle_subsidence = toggle_subsidence

    @property
    def theta1(self):
        """
        theta1 defines the left radial bound for the subsiding region.

        For more information on *theta1* and defining the subsidence pattern,
        refer to :meth:`init_subsidence`
        """
        return self._theta1

    @theta1.setter
    def theta1(self, theta1):
        self._theta1 = theta1

    @property
    def theta2(self):
        """
        theta2 defines the right radial bound for the subsiding region.

        For more information on *theta2* and defining the subsidence pattern,
        refer to :meth:`init_subsidence`
        """
        return self._theta2

    @theta2.setter
    def theta2(self, theta2):
        self._theta2 = theta2

    @property
    def sigma_max(self):
        """
        sigma_max defines the maximum rate of subsidence.
        """
        return self._sigma_max

    @sigma_max.setter
    def sigma_max(self, sigma_max):
        self._sigma_max = sigma_max

    @property
    def start_subsidence(self):
        """
        start_subsidence defines the start time at which subsidence begins.
        """
        return self._start_subsidence

    @start_subsidence.setter
    def start_subsidence(self, start_subsidence):
        self._start_subsidence = start_subsidence

    @property
    def save_eta_figs(self):
        """
        save_eta_figs controls whether or not figures of topography are saved.
        """
        return self._save_eta_figs

    @save_eta_figs.setter
    def save_eta_figs(self, save_eta_figs):
        self._save_eta_figs = save_eta_figs

    @property
    def save_stage_figs(self):
        """
        save_stage_figs controls whether or not stage figures are saved.
        """
        return self._save_stage_figs

    @save_stage_figs.setter
    def save_stage_figs(self, save_stage_figs):
        self._save_stage_figs = save_stage_figs

    @property
    def save_depth_figs(self):
        """
        save_depth_figs controls saving of water depth figures.
        """
        return self._save_depth_figs

    @save_depth_figs.setter
    def save_depth_figs(self, save_depth_figs):
        self._save_depth_figs = save_depth_figs

    @property
    def save_discharge_figs(self):
        """
        save_discharge_figs controls saving of water discharge figures.
        """
        return self._save_discharge_figs

    @save_discharge_figs.setter
    def save_discharge_figs(self, save_discharge_figs):
        self._save_discharge_figs = save_discharge_figs

    @property
    def save_velocity_figs(self):
        """
        save_velocity_figs controls saving of water velocity figures.
        """
        return self._save_velocity_figs

    @save_velocity_figs.setter
    def save_velocity_figs(self, save_velocity_figs):
        self._save_velocity_figs = save_velocity_figs

    @property
    def save_sedflux_figs(self):
        """
        save_sedflux_figs controls saving of sediment flux figures.
        """
        return self._save_sedflux_figs

    @save_sedflux_figs.setter
    def save_sedflux_figs(self, save_sedflux_figs):
        self._save_sedflux_figs = save_sedflux_figs

    @property
    def save_figs_sequential(self):
        """
        save_figs_sequential sets how figures are to be saved.

        save_figs_sequential is a *boolean* parameter that can be set to True
        or False. If True, then for each figure saving parameter set to True
        (e.g. :attr:`save_velocity_figs`) a new figure will be saved at an
        inteval of :attr:`save_dt`. The file names of the figures will
        correspond to the model timestep at which they were saved. If instead,
        the save_figs_sequential parameter is set to False, then only the
        latest figure will be kept, and at each *save_dt* time, the figure file
        will be overwritten using the current model status.
        """
        return self._save_figs_sequential

    @save_figs_sequential.setter
    def save_figs_sequential(self, save_figs_sequential):
        self._save_figs_sequential = save_figs_sequential

    @property
    def save_metadata(self):
        """
        save_metadata explicit control on whether or not metadata is saved.

        save_metadata is a boolean that can be manually togged on (True) to
        ensure metadata is saved to disk even if no other output information is
        being saved. If any grids or strata are being saved, then metadata
        saving will be turned on automatically, even if this parameter is set
        to False. Metadata associated with pyDeltaRCM are 1-D arrays
        (vectors) and 0-D arrays (floats and integers) primarily containing
        information about the domain and the inlet conditions for a given model
        run.
        """
        return self._save_metadata

    @save_metadata.setter
    def save_metadata(self, save_metadata):
        self._save_metadata = save_metadata

    @property
    def save_eta_grids(self):
        """
        save_eta_grids controls whether or not topography information is saved.
        """
        return self._save_eta_grids

    @save_eta_grids.setter
    def save_eta_grids(self, save_eta_grids):
        self._save_eta_grids = save_eta_grids

    @property
    def save_stage_grids(self):
        """
        save_stage_grids controls whether or not stage information is saved.
        """
        return self._save_stage_grids

    @save_stage_grids.setter
    def save_stage_grids(self, save_stage_grids):
        self._save_stage_grids = save_stage_grids

    @property
    def save_depth_grids(self):
        """
        save_depth_grids controls whether or not depth information is saved.
        """
        return self._save_depth_grids

    @save_depth_grids.setter
    def save_depth_grids(self, save_depth_grids):
        self._save_depth_grids = save_depth_grids

    @property
    def save_discharge_grids(self):
        """
        save_discharge_grids controls saving of water discharge information.
        """
        return self._save_discharge_grids

    @save_discharge_grids.setter
    def save_discharge_grids(self, save_discharge_grids):
        self._save_discharge_grids = save_discharge_grids

    @property
    def save_velocity_grids(self):
        """
        save_velocity_grids controls saving of water velocity information.
        """
        return self._save_velocity_grids

    @save_velocity_grids.setter
    def save_velocity_grids(self, save_velocity_grids):
        self._save_velocity_grids = save_velocity_grids

    @property
    def save_sedflux_grids(self):
        """
        save_sedflux_grids controls saving of sediment discharge information.
        """
        return self._save_sedflux_grids

    @save_sedflux_grids.setter
    def save_sedflux_grids(self, save_sedflux_grids):
        self._save_sedflux_grids = save_sedflux_grids

    @property
    def save_discharge_components(self):
        """
        save_discharge_components controls saving of x-y discharge components.
        """
        return self._save_discharge_components

    @save_discharge_components.setter
    def save_discharge_components(self, save_discharge_components):
        self._save_discharge_components = save_discharge_components

    @property
    def save_velocity_components(self):
        """
        save_velocity_components controls saving of x-y velocity components.
        """
        return self._save_velocity_components

    @save_velocity_components.setter
    def save_velocity_components(self, save_velocity_components):
        self._save_velocity_components = save_velocity_components

    @property
    def save_dt(self):
        """
        save_dt defines the saving interval in seconds.
        """
        return self._save_dt

    @save_dt.setter
    def save_dt(self, save_dt):
        self._save_dt = save_dt

    @property
    def checkpoint_dt(self):
        """
        checkpoint_dt defines the interval to create checkpoint information.
        """
        return self._checkpoint_dt

    @checkpoint_dt.setter
    def checkpoint_dt(self, checkpoint_dt):
        self._checkpoint_dt = checkpoint_dt

    @property
    def save_strata(self):
        """
        save_strata controls whether or not stratigraphy information is saved.
        """
        return self._save_strata

    @save_strata.setter
    def save_strata(self, save_strata):
        self._save_strata = save_strata

    @property
    def save_checkpoint(self):
        """
        save_checkpoint controls saving of model checkpoint information.
        """
        return self._save_checkpoint

    @save_checkpoint.setter
    def save_checkpoint(self, save_checkpoint):
        self._save_checkpoint = save_checkpoint

    @property
    def resume_checkpoint(self):
        """
        resume_checkpoint controls loading of a checkpoint if run is resuming.
        """
        return self._resume_checkpoint

    @resume_checkpoint.setter
    def resume_checkpoint(self, resume_checkpoint):
        self._resume_checkpoint = resume_checkpoint

    @property
    def omega_sfc(self):
        """
        omega_sfc is a water surface underrelaxation parameter.
        """
        return self._omega_sfc

    @omega_sfc.setter
    def omega_sfc(self, omega_sfc):
        self._omega_sfc = omega_sfc

    @property
    def omega_flow(self):
        """
        omega_flow is a flow velocity underrelaxation parameter.
        """
        return self._omega_flow

    @omega_flow.setter
    def omega_flow(self, omega_flow):
        self._omega_flow = omega_flow

    @property
    def Nsmooth(self):
        """
        Nsmooth defines the number of times the water surface is smoothed.
        """
        return self._Nsmooth

    @Nsmooth.setter
    def Nsmooth(self, Nsmooth):
        self._Nsmooth = Nsmooth

    @property
    def theta_water(self):
        """
        theta_water is the exponent of depth dependence for weighted routing.

        For the routing of the water parcels, the dependence of the random walk
        on local water depth can be modulated by varying this parameter value.
        As theta_water gets larger, the importance of the water depth to the
        weighting scheme grows.

        .. note::
           The value of *theta_water* also influences the values of
           :attr:`coeff_theta_sand` and :attr:`coeff_theta_mud` which are
           coefficients that are multiplied by *theta_water* to set the theta
           values for the sand and mud routing respectively.
        """
        return self._theta_water

    @theta_water.setter
    def theta_water(self, theta_water):
        self._theta_water = theta_water

    @property
    def coeff_theta_sand(self):
        """
        coeff_theta_sand is the coefficient applied to theta for sand routing.

        coeff_theta_sand is a coefficient applied to the :attr:`theta_water`
        attribute to define the value of theta for sand routing. Theta is
        the exponent applied to the local water depth and is used to weight
        the random walk. For mud and sand these weighting rules vary, which is
        why the :attr:`theta_water` term is multiplied by these coefficients:
        :attr:`coeff_theta_mud` and :attr:`coeff_theta_sand`.
        """
        return self._coeff_theta_sand

    @coeff_theta_sand.setter
    def coeff_theta_sand(self, coeff_theta_sand):
        self._coeff_theta_sand = coeff_theta_sand

    @property
    def coeff_theta_mud(self):
        """
        coeff_theta_mud is the coefficient applied to theta for mud routing.

        coeff_theta_mud is a coefficient applied to the :attr:`theta_water`
        attribute to define the value of theta for mud routing. Theta is
        the exponent applied to the local water depth and is used to weight
        the random walk. For mud and sand these weighting rules vary, which is
        why the :attr:`theta_water` term is multiplied by these coefficients:
        :attr:`coeff_theta_mud` and :attr:`coeff_theta_sand`.
        """
        return self._coeff_theta_mud

    @coeff_theta_mud.setter
    def coeff_theta_mud(self, coeff_theta_mud):
        self._coeff_theta_mud = coeff_theta_mud

    @property
    def beta(self):
        """
        beta is the bedload transport capacity exponent.

        beta is an exponent on the bedload transport terms that is applied to
        the local velocity and threshold velocity terms.
        """
        return self._beta

    @beta.setter
    def beta(self, beta):
        self._beta = beta

    @property
    def sed_lag(self):
        """
        sed_lag is a "sedimentation lag" parameter.

        sed_lag influences the properties of mud deposition by controlling how
        much mud is deposited when the flow velocity is below the threshold
        for mud deposition (:attr:`coeff_U_dep_mud`).
        """
        return self._sed_lag

    @sed_lag.setter
    def sed_lag(self, sed_lag):
        self._sed_lag = sed_lag

    @property
    def coeff_U_dep_mud(self):
        """
        coeff_U_dep_mud is the threshold velocity for mud sediment deposition.

        coeff_U_dep_mud sets the threshold velocity for mud to be depoisted.
        The smaller this parameter is, the longer a mud parcel can travel
        before losing all of its mud volume.
        """
        return self._coeff_U_dep_mud

    @coeff_U_dep_mud.setter
    def coeff_U_dep_mud(self, coeff_U_dep_mud):
        self._coeff_U_dep_mud = coeff_U_dep_mud

    @property
    def coeff_U_ero_mud(self):
        """
        coeff_U_ero_mud is the mud erosion velocity threshold coefficient.

        coeff_U_ero_mud sets the threshold velocity for erosion of the mud
        (suspended load) sediment. The higher this value is, the more difficult
        it is to erode mud deposits.
        """
        return self._coeff_U_ero_mud

    @coeff_U_ero_mud.setter
    def coeff_U_ero_mud(self, coeff_U_ero_mud):
        self._coeff_U_ero_mud = coeff_U_ero_mud

    @property
    def coeff_U_ero_sand(self):
        """
        coeff_U_ero_sand is the sand erosion velocity threshold coefficient.

        coeff_U_ero_sand sets the threshold velocity for sediment erosion of
        the sand (bedload) sediment. The higher this value is, the more
        difficult it is to erode the bed.
        """
        return self._coeff_U_ero_sand

    @coeff_U_ero_sand.setter
    def coeff_U_ero_sand(self, coeff_U_ero_sand):
        self._coeff_U_ero_sand = coeff_U_ero_sand

    @property
    def alpha(self):
        """
        alpha is the topographic diffusion coefficient.

        alpha is the coefficient used for topographic diffusion. It controls
        both the cross-slope sediment flux as well as bank erodability.
        """
        return self._alpha

    @alpha.setter
    def alpha(self, alpha):
        self._alpha = alpha

    @property
    def stepmax(self):
        """
        stepmax is the maximum number of jumps a parcel can make.

        stepmax is the maximum number of jumps a parcel (water or sediment)
        can make. If the parcel reaches the oceanic boundary before stepmax is
        reached, then this condition is not invoked. However if the parcel
        does not reach the oceaninc bounary but does reach "stepmax" jumps,
        then the parcel will just stop moving and disappear.

        If stepmax is not specified in the yaml, the default value assigned
        is 2 times the perimeter of the domain (2 * (self.L + self.W)).

        .. note::
           If stepmax is set too low and many parcels reach the stepmax
           condition during routing, then there may be a considerable
           amount of sediment 'missing' from the system as any sediment in
           a parcel that hits the stepmax threshold disappears from the
           simulation.
        """
        return self._stepmax

    @stepmax.setter
    def stepmax(self, stepmax):
        self._stepmax = stepmax

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
