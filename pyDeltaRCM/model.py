#! /usr/bin/env python
import warnings
import logging
import datetime
import os

from .iteration_tools import iteration_tools
from .sed_tools import sed_tools
from .water_tools import water_tools
from .init_tools import init_tools
from .hook_tools import hook_tools
from .debug_tools import debug_tools
from .shared_tools import _get_version


class DeltaModel(iteration_tools, sed_tools, water_tools,
                 init_tools, hook_tools, debug_tools, object):
    """Main model class.

    Instantiating the model class is described in the :meth:`__init__` method
    below in detail, but generally model instantiation occurs via a model run
    YAML configuration file. These YAML configuration files define model
    parameters which are used in the run; read more about creating input YAML
    configuration files in the :doc:`/guides/user_guide`.

    Once the model has been instantiated, the model is updated via the
    :meth:`update`. This method coordinates the hydrology, sediment transport,
    subsidence, and stratigraphic operation that must occur during each
    timestep. The :meth:`update` should be called iteratively, to "run" the
    model.

    Finally, after all ``update`` steps are complete, the model should be
    finalized (:meth:`finalize`), such that files and data are appropriately
    closed and written to disk.
    """

    def __init__(self, input_file=None, defer_output=False, **kwargs):
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
            :obj:`init_output_file`, :obj:`output_data`, and
            :obj:`output_checkpoint` method if `defer_output` is `True`.

        **kwargs
            Optionally, any parameter typically specified in the YAML file can
            be passed as a keyword argument to the instantiation of the
            DeltaModel. Keywords will override the specification of any value
            in the YAML file. This functionality is intended for advanced use
            cases, and should not be preffered to specifying all inputs in a
            YAML configuration file.

        Returns
        -------

        Notes
        -----
        For more information regarding input configuration files, see the
        :doc:`/guides/user_guide`.

        """
        self.__pyDeltaRCM_version__ = _get_version()

        self._time = 0.
        self._time_iter = int(0)
        self._save_time_since_data = float("inf")  # force save on t==0
        self._save_iter = int(0)
        self._save_time_since_checkpoint = float("inf")  # force save on t==0

        self.input_file = input_file
        _src_dir = os.path.realpath(os.path.dirname(__file__))
        self.default_file = os.path.join(_src_dir, 'default.yml')

        # infrastructure for custom yaml parameters if needed
        if hasattr(self, 'subclass_parameters') is False:
            self.subclass_parameters = dict()  # init dict for custom params

        # check for any deprecated hooks
        self._check_deprecated_hooks()

        # import the input file
        self.hook_import_files()  # user hook
        self.import_files(kwargs)

        # initialize output folders and logger
        self.init_output_infrastructure()
        self.init_logger()

        # apply the configuration
        self.hook_process_input_to_model()
        self.process_input_to_model()

        # determine and set the model seed
        self.determine_random_seed()

        # create model variables based on configuration
        self.hook_create_other_variables()
        self.create_other_variables()

        # create the model domain based on configuration
        self.hook_create_domain()
        self.create_domain()

        # initialize the sediment router classes
        self.init_sediment_routers()

        # set up the subsidence fields
        self.init_subsidence()

        # if resume flag set to True, load checkpoint, open netCDF4
        if self.resume_checkpoint:
            # load values from the checkpoint and don't init final features
            self.hook_load_checkpoint()
            self.load_checkpoint(defer_output)

        else:
            # initialize the output file
            if not defer_output:
                self.hook_init_output_file()
                self.init_output_file()

                # record initial conditions
                self.hook_output_data()
                self.output_data()

                self.hook_output_checkpoint()
                self.output_checkpoint()

        _msg = 'Model initialization complete'
        self.log_info(_msg)
        self.log_model_time()

    def update(self):
        """Run the model for one full timestep.

        This method handles the input/output from the model, orchestrating the
        various morphodynamic and basin-scale processes (from sub-methods),
        and incrementing the model time-tracking attributes. This method
        calls, in sequence:

            * the routine to run one timestep (i.e., water surface estimation
              and sediment routing;
              :meth:`~pyDeltaRCM.iteration_tools.iteration_tools.solve_water_and_sediment_timestep`)
            * the basin subsidence update;
              (:meth:`~pyDeltaRCM.iteration_tools.iteration_tools.apply_subsidence`)
            * the timestep finalization routine (applying boundary conditions);
              (:meth:`~pyDeltaRCM.iteration_tools.iteration_tools.finalize_timestep`)
            * the internal time-tracking attributes of the model are updated
            * the routine to output data to the NetCDF file;
              (:meth:`~pyDeltaRCM.iteration_tools.iteration_tools.output_data`)
            * the routine to output a checkpoint file;
              (:meth:`~pyDeltaRCM.iteration_tools.iteration_tools.output_checkpoint`)

        If you attempt to override the ``update`` routine, you must implement
        these operations at a minimum. More likely, you can implement what you
        need to by utilizing a model `hook`, or overriding one of the methods
        called by `update`. For more information on customizing the model, see
        the :ref:`complete guide <customize_the_model>`.

        Parameters
        ----------

        Returns
        -------

        Raises
        ------
        RuntimeError
            If model has already been finalized via :meth:`finalize`.

        """
        if self._is_finalized:
            raise RuntimeError('Cannot update model, model already finalized!')

        # update the model, i.e., the actual model morphodynamics
        self.hook_solve_water_and_sediment_timestep()
        self.solve_water_and_sediment_timestep()

        self.hook_apply_subsidence()
        self.apply_subsidence()

        self.hook_finalize_timestep()
        self.finalize_timestep()

        # update time-tracking fields
        self._time += self.dt
        self._save_time_since_data += self.dt
        self._save_time_since_checkpoint += self.dt
        self._time_iter += int(1)
        self.log_model_time()

        # record the state of the model if needed
        self.hook_output_data()
        self.output_data()

        # save a checkpoint if needed
        self.hook_output_checkpoint()
        self.output_checkpoint()

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

        if self._is_finalized:
            raise RuntimeError('Cannot finalize model, '
                               'model already finalized!')

        try:
            self.output_netcdf.close()
            _msg = 'Closed output NetCDF4 file'
            self.log_info(_msg, verbosity=1)
        except AttributeError:
            self.log_info('No output NetCDF4 file to close.')
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
    def hb(self):
        """
        hb is the basin depth in meters.

        hb is the basin depth in meters. This parameter must be an *integer*
        or *float*, or *None*. If no value is supplied for this parameter,
        then the value of h0 is used to determine the basin depth (i.e., they
        are the same).
        """
        return self._hb

    @hb.setter
    def hb(self, hb):
        self._hb = hb

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
    def active_layer_thickness(self):
        """
        Active layer thickness for sand fraction.

        The active layer thickness is the depth to which the sand fraction
        values are maintained in the event of erosion from one timestep to the
        next. When erosion exceeds the depth of the active layer, the boundary
        condition is to fill that cell with a value of 0 (maybe this should be
        -1 so that it is clear which cells are not truly mud but unknown
        sediment content based on the active layer thickness?)
        """
        return self._active_layer_thickness

    @active_layer_thickness.setter
    def active_layer_thickness(self, active_layer_thickness):
        if active_layer_thickness is None:
            active_layer_thickness = self._h0 / 2
        elif active_layer_thickness < 0:
            raise ValueError('active_layer thickness must be greater than'
                             ' or equal to 0, cannot be negative.')
        self._active_layer_thickness = active_layer_thickness

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
    def subsidence_rate(self):
        """
        subsidence_rate defines the maximum rate of subsidence.
        """
        return self._subsidence_rate

    @subsidence_rate.setter
    def subsidence_rate(self, subsidence_rate):
        self._subsidence_rate = subsidence_rate

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
        if (save_eta_figs is True) and \
          ('eta' not in self._save_fig_list.keys()):
            self._save_fig_list['eta'] = ['eta']
        elif ((save_eta_figs is False) and
              ('eta' in self._save_fig_list.keys())):
            del self._save_fig_list['eta']
        self._save_eta_figs = save_eta_figs

    @property
    def save_stage_figs(self):
        """
        save_stage_figs controls whether or not stage figures are saved.
        """
        return self._save_stage_figs

    @save_stage_figs.setter
    def save_stage_figs(self, save_stage_figs):
        if (save_stage_figs is True) and \
          ('stage' not in self._save_fig_list.keys()):
            self._save_fig_list['stage'] = ['stage']
        elif ((save_stage_figs is False) and
              ('stage' in self._save_fig_list.keys())):
            del self._save_fig_list['stage']
        self._save_stage_figs = save_stage_figs

    @property
    def save_depth_figs(self):
        """
        save_depth_figs controls saving of water depth figures.
        """
        return self._save_depth_figs

    @save_depth_figs.setter
    def save_depth_figs(self, save_depth_figs):
        if (save_depth_figs is True) and \
          ('depth' not in self._save_fig_list.keys()):
            self._save_fig_list['depth'] = ['depth']
        elif ((save_depth_figs is False) and
              ('depth' in self._save_fig_list.keys())):
            del self._save_fig_list['depth']
        self._save_depth_figs = save_depth_figs

    @property
    def save_discharge_figs(self):
        """
        save_discharge_figs controls saving of water discharge figures.
        """
        return self._save_discharge_figs

    @save_discharge_figs.setter
    def save_discharge_figs(self, save_discharge_figs):
        if (save_discharge_figs is True) and \
          ('discharge' not in self._save_fig_list.keys()):
            self._save_fig_list['discharge'] = ['qw']
        elif ((save_discharge_figs is False) and
              ('discharge' in self._save_fig_list)):
            del self._save_fig_list['discharge']
        self._save_discharge_figs = save_discharge_figs

    @property
    def save_velocity_figs(self):
        """
        save_velocity_figs controls saving of water velocity figures.
        """
        return self._save_velocity_figs

    @save_velocity_figs.setter
    def save_velocity_figs(self, save_velocity_figs):
        if (save_velocity_figs is True) and \
          ('velocity' not in self._save_fig_list.keys()):
            self._save_fig_list['velocity'] = ['uw']
        elif ((save_velocity_figs is False) and
              ('velocity' in self._save_fig_list.keys())):
            del self._save_fig_list['sedflux']
        self._save_velocity_figs = save_velocity_figs

    @property
    def save_sedflux_figs(self):
        """
        save_sedflux_figs controls saving of sediment flux figures.
        """
        return self._save_sedflux_figs

    @save_sedflux_figs.setter
    def save_sedflux_figs(self, save_sedflux_figs):
        if (save_sedflux_figs is True) and \
          ('sedflux' not in self._save_fig_list.keys()):
            self._save_fig_list['sedflux'] = ['qs']
        elif ((save_sedflux_figs is False) and
              ('sedflux' in self._save_fig_list)):
            del self._save_fig_list['sedflux']
        self._save_sedflux_figs = save_sedflux_figs

    @property
    def save_sandfrac_figs(self):
        """
        save_sandfrac_figs controls whether or not figures of the sediment
        surface (i.e., the bed) sand fraction are saved.
        """
        return self._save_sandfrac_figs

    @save_sandfrac_figs.setter
    def save_sandfrac_figs(self, save_sandfrac_figs):
        if (save_sandfrac_figs is True) and \
          ('sandfrac' not in self._save_fig_list.keys()):
            self._save_fig_list['sandfrac'] = ['sand_frac']
        elif ((save_sandfrac_figs is False) and
              ('sandfrac' in self._save_fig_list)):
            del self._save_fig_list['sandfrac']
        self._save_sandfrac_figs = save_sandfrac_figs

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
        if (save_eta_grids is True) and \
          ('eta' not in self._save_var_list.keys()):
            self._save_var_list['eta'] = ['eta', 'meters', 'f4',
                                          self._netcdf_coords]
        elif ((save_eta_grids is False) and
              ('eta' in self._save_var_list.keys())):
            del self._save_var_list['eta']
        self._save_eta_grids = save_eta_grids

    @property
    def save_stage_grids(self):
        """
        save_stage_grids controls whether or not stage information is saved.
        """
        return self._save_stage_grids

    @save_stage_grids.setter
    def save_stage_grids(self, save_stage_grids):
        if (save_stage_grids is True) and \
          ('stage' not in self._save_var_list.keys()):
            self._save_var_list['stage'] = ['stage', 'meters', 'f4',
                                            self._netcdf_coords]
        elif ((save_stage_grids is False) and
              ('stage' in self._save_var_list.keys())):
            del self._save_var_list['stage']
        self._save_stage_grids = save_stage_grids

    @property
    def save_depth_grids(self):
        """
        save_depth_grids controls whether or not depth information is saved.
        """
        return self._save_depth_grids

    @save_depth_grids.setter
    def save_depth_grids(self, save_depth_grids):
        if (save_depth_grids is True) and \
          ('depth' not in self._save_var_list.keys()):
            self._save_var_list['depth'] = ['depth', 'meters', 'f4',
                                            self._netcdf_coords]
        elif ((save_depth_grids is False) and
              ('depth' in self._save_var_list.keys())):
            del self._save_var_list['depth']
        self._save_depth_grids = save_depth_grids

    @property
    def save_discharge_grids(self):
        """
        save_discharge_grids controls saving of water discharge information.
        """
        return self._save_discharge_grids

    @save_discharge_grids.setter
    def save_discharge_grids(self, save_discharge_grids):
        if (save_discharge_grids is True) and \
          ('discharge' not in self._save_var_list.keys()):
            self._save_var_list['discharge'] = ['qw',
                                                'cubic meters per second',
                                                'f4',
                                                self._netcdf_coords]
        elif ((save_discharge_grids is False) and
              ('discharge' in self._save_var_list.keys())):
            del self._save_var_list['discharge']
        self._save_discharge_grids = save_discharge_grids

    @property
    def save_velocity_grids(self):
        """
        save_velocity_grids controls saving of water velocity information.
        """
        return self._save_velocity_grids

    @save_velocity_grids.setter
    def save_velocity_grids(self, save_velocity_grids):
        if (save_velocity_grids is True) and \
          ('velocity' not in self._save_var_list.keys()):
            self._save_var_list['velocity'] = ['uw', 'meters per second', 'f4',
                                               self._netcdf_coords]
        elif ((save_velocity_grids is False) and
              ('velocity' in self._save_var_list.keys())):
            del self._save_var_list['velocity']
        self._save_velocity_grids = save_velocity_grids

    @property
    def save_sedflux_grids(self):
        """
        save_sedflux_grids controls saving of sediment discharge information.
        """
        return self._save_sedflux_grids

    @save_sedflux_grids.setter
    def save_sedflux_grids(self, save_sedflux_grids):
        if (save_sedflux_grids is True) and \
          ('sedflux' not in self._save_var_list.keys()):
            self._save_var_list['sedflux'] = ['qs', 'cubic meters per second',
                                              'f4',
                                              self._netcdf_coords]
        elif ((save_sedflux_grids is False) and
              ('sedflux' in self._save_var_list.keys())):
            del self._save_var_list['sedflux']
        self._save_sedflux_grids = save_sedflux_grids

    @property
    def save_sandfrac_grids(self):
        """
        save_sandfrac_grids controls whether or not the sediment
        surface (i.e., the bed) sand fraction is saved.
        """
        return self._save_sandfrac_grids

    @save_sandfrac_grids.setter
    def save_sandfrac_grids(self, save_sandfrac_grids):
        if (save_sandfrac_grids is True) and \
          ('sandfrac' not in self._save_var_list.keys()):
            self._save_var_list['sandfrac'] = ['sand_frac', 'fraction', 'f4',
                                               self._netcdf_coords]
        elif ((save_sandfrac_grids is False) and
              ('sandfrac' in self._save_var_list.keys())):
            del self._save_var_list['sandfrac']
        self._save_sandfrac_grids = save_sandfrac_grids

    @property
    def save_discharge_components(self):
        """
        save_discharge_components controls saving of x-y discharge components.
        """
        return self._save_discharge_components

    @save_discharge_components.setter
    def save_discharge_components(self, save_discharge_components):
        if (save_discharge_components is True):
            if ('discharge_x' not in self._save_var_list.keys()):
                self._save_var_list['discharge_x'] = [
                    'qx', 'cubic meters per second', 'f4',
                    self._netcdf_coords]
            if ('discharge_y' not in self._save_var_list.keys()):
                self._save_var_list['discharge_y'] = [
                    'qy', 'cubic meters per second', 'f4',
                    self._netcdf_coords]
        elif (save_discharge_components is False):
            if ('discharge_x' in self._save_var_list.keys()):
                del self._save_var_list['discharge_x']
            if ('discharge_y' in self._save_var_list.keys()):
                del self._save_var_list['discharge_y']
        self._save_discharge_components = save_discharge_components

    @property
    def save_velocity_components(self):
        """
        save_velocity_components controls saving of x-y velocity components.
        """
        return self._save_velocity_components

    @save_velocity_components.setter
    def save_velocity_components(self, save_velocity_components):
        if (save_velocity_components is True):
            if ('velocity_x' not in self._save_var_list.keys()):
                self._save_var_list['velocity_x'] = [
                    'ux', 'meters per second', 'f4',
                    self._netcdf_coords]
            if ('velocity_y' not in self._save_var_list.keys()):
                self._save_var_list['velocity_y'] = [
                    'uy', 'meters per second', 'f4',
                    self._netcdf_coords]
        elif (save_velocity_components is False):
            if ('velocity_x' in self._save_var_list.keys()):
                del self._save_var_list['velocity_x']
            if ('velocity_y' in self._save_var_list.keys()):
                del self._save_var_list['velocity_y']
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
        if checkpoint_dt is None:
            checkpoint_dt = self._save_dt
        self._checkpoint_dt = checkpoint_dt

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

        When setting this option in the YAML or command line, you can specify
        a `bool` (e.g., `resume_checkpoint: True`) which will search in the
        :obj:`out_dir` directory for a file named ``checkpoint.npz``.
        Alternatively, you can specify an alternative folder to search for the
        checkpoint *as a string* (e.g., `resume_checkpoint:
        '/some/other/path'`).
        """
        return self._resume_checkpoint

    @resume_checkpoint.setter
    def resume_checkpoint(self, resume_checkpoint):
        if isinstance(resume_checkpoint, str):
            self._checkpoint_folder = resume_checkpoint
            self._resume_checkpoint = True
        else:
            self._checkpoint_folder = self.prefix
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
        `stepmax` is the maximum number of jumps a parcel can make.

        The maximum number of jumps a parcel (water or sediment) can take is
        determined by `stepmax`. If the parcel reaches the oceanic boundary
        before `stepmax` is reached, then this condition is not invoked.
        However if the parcel does not reach the oceanic bounary but does
        reach `stepmax` jumps, then the parcel will just stop moving and
        disappear.

        If `stepmax` is not specified in the yaml, the default value assigned
        is 2 times the perimeter of the domain (2 * (self.L + self.W)).

        .. note::
           If `stepmax` is set too low and many parcels reach the `stepmax`
           condition during routing, then there may be a considerable
           amount of sediment 'missing' from the system as any sediment in
           a parcel that hits the `stepmax` threshold disappears from the
           simulation.
        """
        return self._stepmax

    @stepmax.setter
    def stepmax(self, stepmax):
        self._stepmax = stepmax

    @property
    def clobber_netcdf(self):
        """
        Allows overwriting (clobbering) of an existing netCDF output file.

        Default behavior, clobber_netcdf: False, is for the model to raise an
        error during initialization if a netCDF output file is found in the
        location specified by :attr:`out_dir`. If the clobber_netcdf parameter
        is instead set to True, then if there is an existing netCDF output file
        in the target folder, it will be "clobbered" or overwritten.
        """
        return self._clobber_netcdf

    @clobber_netcdf.setter
    def clobber_netcdf(self, clobber_netcdf):
        self._clobber_netcdf = clobber_netcdf

    @property
    def legacy_netcdf(self):
        """Enable output in legacy netCDF format.

        .. note:: new in `v2.1.0`.

        Default behavior, legacy_netcdf: False, is for the model to use the
        new `v2.1.0` output netCDF format. The updated format is configured
        to match the input expected by `xarray`, which eases interaction with
        model outputs. The change in format is from inconsistently named
        dimensions and *coordinate variables*, to homogeneous definitions.
        Also, the legacy format specified the variables `x` and `y` as 2d
        grids, whereas the updated format uses 1d coordinate arrays.
        
        .. important::

            There are no changes to the dimensionality of data such as bed
            elevation or velocity, only the metadata specifying the location of
            the data are changed.

        +-------------+-------------------+---------------------------------+
        |             | default           | legacy                          |
        +=============+===================+=================================+
        | dimensions  | `time`, `x`, `y`  | `total_time`, `length`, `width` |
        +-------------+-------------------+---------------------------------+
        | variables   | `time`, `x`, `y`  | `time`, `y`, `x`; x, y as 2D    |
        +-------------+-------------------+---------------------------------+
        | data        | `t-x-y` array     | `t-y-x` array                   |
        +-------------+-------------------+---------------------------------+

        .. hint::

            If you are beginning a new project, use `legacy_netcdf == False`,
            and update scripts accordingly.
        """
        return self._legacy_netcdf

    @legacy_netcdf.setter
    def legacy_netcdf(self, legacy_netcdf):
        self._legacy_netcdf = legacy_netcdf

    @property
    def time(self):
        """Elapsed model time in seconds.
        """
        return self._time

    @property
    def dt(self):
        """The time step.

        The value of the timestep (:math:`\\Delta t`) is a balance between
        computation efficiency and model stability [1]_. The
        :ref:`reference-volume`, which characterizes the
        volume of an inlet-channel cell, and the sediment discharge to the
        model domain scale the timestep as:

        .. math::

            \\Delta t = \\dfrac{0.1 {N_0}^2 V_0}{Q_{s0}}

        where :math:`Q_{s0}` is the sediment discharge into the model domain
        (m:sup:`3`/s), :math:`V_0` is the reference volume, and :math:`N_0` is
        the number of cells across the channel (determined by :obj:`N0_meters`
        and :obj:`dx`).

        .. [1] A reduced-complexity model for river delta formation – Part 1:
               Modeling deltas with channel dynamics, M. Liang, V. R. Voller,
               and C. Paola, Earth Surf. Dynam., 3, 67–86, 2015.
               https://doi.org/10.5194/esurf-3-67-2015

        Raises
        ------
        UserWarning
            If a very small timestep is configured.
        """
        return self._dt

    @property
    def time_step(self):
        """Alias for :obj:`dt`.
        """
        return self._dt

    @time_step.setter
    def time_step(self, new_time_step):
        if new_time_step * self.init_Np_sed < 100:
            warnings.warn(UserWarning('Using a very small time step, '
                                      'Delta might evolve very slowly.'))

        if self.toggle_subsidence:
            self.sigma = (self.sigma / self._dt) * new_time_step

        self._dt = new_time_step

    @property
    def time_iter(self):
        """Number of time iterations.

        The number of times the :obj:`update` method has been called.
        """
        return self._time_iter

    @property
    def save_time_since_data(self):
        """Time since data last output.

        The elapsed time (seconds) since data was output with
        :obj:`output_data`.
        """
        return self._save_time_since_data

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
        self.create_boundary_conditions()
        self.init_sediment_routers()

    @property
    def channel_width(self):
        """Get channel width."""
        return self.N0 * self._dx

    @channel_width.setter
    def channel_width(self, new_N0_meters):
        self.N0_meters = new_N0_meters
        self.create_boundary_conditions()
        self.init_sediment_routers()
        if self.channel_width != new_N0_meters:
            warnings.warn(UserWarning(
                'Channel width was updated to {0} m, rather than input {1},'
                'due to grid resolution or imposed domain '
                'restrictions.'.format(self.channel_width, new_N0_meters)))

    @property
    def channel_flow_depth(self):
        """Get channel flow depth."""
        return self.h0

    @channel_flow_depth.setter
    def channel_flow_depth(self, new_d):
        self.h0 = new_d
        self.create_boundary_conditions()
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
    def influx_sediment_concentration(self, new_C0):
        self.C0_percent = new_C0 * 100
        self.create_boundary_conditions()
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

    @property
    def sand_frac_bc(self):
        """Sand fraction boundary condition."""
        return self._sand_frac_bc

    @sand_frac_bc.setter
    def sand_frac_bc(self, sand_frac_bc):
        self._sand_frac_bc = sand_frac_bc
