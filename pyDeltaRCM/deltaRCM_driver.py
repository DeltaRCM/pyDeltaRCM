#! /usr/bin/env python
import warnings
import logging
from .deltaRCM_tools import Tools
import datetime


class pyDeltaRCM(Tools):

    _input_vars = {
        'model_output__site_prefix': {'name':'site_prefix',
            'type': 'string', 'default': ''},
        'model_output__case_prefix': {'name':'case_prefix',
            'type': 'string', 'default': ''},
        'model_output__out_dir': {'name':'out_dir',
            'type': 'string', 'default': 'deltaRCM_Output/'},
        'model_grid__length': {'name':'Length',
            'type': 'float', 'default': 5000.},
        'model_grid__width': {'name':'Width',
            'type': 'float', 'default': 10000.},
        'model_grid__cell_size': {'name':'dx',
            'type': 'float', 'default': 100.},
        'land_surface__width': {'name':'L0_meters',
            'type': 'float', 'default': 300.}, 
        'land_surface__slope': {'name':'S0',
            'type': 'float', 'default': 0.00015},
        'model__max_iteration': {'name':'itermax',
            'type': 'long', 'default': 1},
        'water__number_parcels': {'name':'Np_water',
            'type': 'long', 'default': 1000},
        'channel__flow_velocity': {'name':'u0',
            'type': 'float', 'default': 1.},
        'channel__width': {'name':'N0_meters',
            'type': 'float', 'default': 300.},
        'channel__flow_depth': {'name':'h0',
            'type': 'float', 'default': 5.},
        'sea_water_surface__mean_elevation': {'name':'H_SL',
            'type': 'float', 'default': 0.},
        'sea_water_surface__rate_change_elevation': {'name':'SLR',
            'type': 'float', 'default': 0.},
        'sediment__number_parcels': {'name':'Np_sed',
            'type': 'long', 'default': 1000},
        'sediment__bedload_fraction': {'name':'f_bedload',
            'type': 'float', 'default': 0.25}, 
        'sediment__influx_concentration': {'name':'C0_percent',
            'type': 'float', 'default': 0.1},                   
        'model_output__opt_eta_figs': {'name':'save_eta_figs',
            'type': 'choice', 'default': True},
        'model_output__opt_stage_figs': {'name':'save_stage_figs',
            'type': 'choice', 'default': False},
        'model_output__opt_depth_figs': {'name':'save_depth_figs',
            'type': 'choice', 'default': False},
        'model_output__opt_eta_grids': {'name':'save_eta_grids',
            'type': 'choice', 'default': False},
        'model_output__opt_stage_grids': {'name':'save_stage_grids',
            'type': 'choice', 'default': False},
        'model_output__opt_depth_grids': {'name':'save_depth_grids',
            'type': 'choice', 'default': False},
        'model_output__opt_time_interval': {'name':'save_dt',
            'type': 'long', 'default': 50},
        'coeff__surface_smoothing': {'name': 'Csmooth',
            'type': 'float', 'default': 0.9},
        'coeff__under_relaxation__water_surface': {'name': 'omega_sfc',
            'type': 'float', 'default': 0.1},
        'coeff__under_relaxation__water_flow': {'name': 'omega_flow',
            'type': 'float', 'default': 0.9},
        'coeff__iterations_smoothing_algorithm': {'name': 'Nsmooth',
            'type': 'long', 'default': 5},
        'coeff__depth_dependence__water': {'name': 'theta_water',
            'type': 'float', 'default': 1.0},
        'coeff__depth_dependence__sand': {'name': 'coeff_theta_sand',
            'type': 'float', 'default': 2.0},
        'coeff__depth_dependence__mud': {'name': 'coeff_theta_mud',
            'type': 'float', 'default': 1.0},
        'coeff__non_linear_exp_sed_flux_flow_velocity': {'name': 'beta',
            'type': 'long', 'default': 3},
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
        'basin__opt_subsidence': {'name':'toggle_subsidence',
            'type': 'choice', 'default': False},
        'basin__maximum_subsidence_rate': {'name': 'sigma_max',
            'type': 'float', 'default': 0.000825},
        'basin__subsidence_start_timestep': {'name': 'start_subsidence',
            'type': 'float', 'default': 0},
        'basin__opt_stratigraphy': {'name': 'save_strata',
            'type': 'choice', 'default': False}
        }
        
        
    #############################################
    ################## __init__ #################
    #############################################

    def __init__(self, input_file):
        '''
        Creates an instance of the model

        Sets the most commonly changed variables here
        Calls functions to set the rest and create the domain (for cleanliness)
        '''
        
        self._time = 0.
        self._time_step = 1.
        
        self.verbose = True
        self.input_file = input_file
        
        self.create_dicts()
        self.set_defaults()
        self.import_file()
        
        self.create_other_variables()
        
        self.init_logger()
        
        self.create_domain()
        
        self.init_subsidence()
        self.init_stratigraphy()
        self.init_output_grids()



    #############################################
    ################### update ##################
    #############################################

    def update(self):
        '''
        Run the model for one full instance
        '''
        
        self.run_one_timestep()
        
        self.apply_subsidence()
        self.record_stratigraphy()
        
        self.finalize_timestep()
        self.output_data()
        
        self._time += self.time_step



    #############################################
    ################## finalize #################
    #############################################        
        
    def finalize(self):
    
        self.output_strata()
        
        try:
            self.output_netcdf.close()
            if self.verbose:
                print 'Closed output netcdf file.'
        except:
            pass
            
            


    @property
    def time_step(self):
        """The time step."""
        return self._time_step

    @time_step.setter
    def time_step(self, new_dt):
        if new_dt * self.init_Np_sed < 100:
            warnings.warn('Using a very small timestep.')
            warnings.warn('Delta might evolve very slowly.')
            
        self.Np_sed = int(new_dt * self.init_Np_sed)
        self.Np_water = int(new_dt * self.init_Np_water)
        
        if self.toggle_subsidence:
            self.sigma = self.subsidence_mask * self.sigma_max * new_dt
        
        self._time_step = new_dt

    @property
    def channel_flow_velocity(self):
        ''' Get channel flow velocity '''
        return self.u0
        
    @channel_flow_velocity.setter
    def channel_flow_velocity(self, new_u0):
        self.u0 = new_u0
        self.create_other_variables()

    @property
    def channel_width(self):
        ''' Get channel width '''
        return self.N0_meters
        
    @channel_width.setter
    def channel_width(self, new_N0):
        self.N0_meters = new_N0
        self.create_other_variables()
        
    @property
    def channel_flow_depth(self):
        ''' Get channel flow depth '''
        return self.h0
        
    @channel_flow_depth.setter
    def channel_flow_depth(self, new_d):
        self.h0 = new_d
        self.create_other_variables()        

    @property
    def sea_surface_mean_elevation(self):
        ''' Get sea surface mean elevation '''
        return self.H_SL
        
    @sea_surface_mean_elevation.setter
    def sea_surface_mean_elevation(self, new_se):
        self.H_SL = new_se
        
    @property
    def sea_surface_elevation_change(self):
        ''' Get rate of change of sea surface elevation, per timestep'''
        return self.SLR
        
    @sea_surface_elevation_change.setter
    def sea_surface_elevation_change(self, new_SLR):
        ''' Set rate of change of sea surface elevation, per timestep'''
        self.SLR = new_SLR
        
    @property
    def bedload_fraction(self):
        ''' Get bedload fraction '''
        return self.f_bedload
        
    @bedload_fraction.setter
    def bedload_fraction(self, new_u0):
        self.f_bedload = new_u0

    @property
    def influx_sediment_concentration(self):
        ''' Get influx sediment concentration '''
        return self.C0_percent
        
    @influx_sediment_concentration.setter
    def influx_sediment_concentration(self, new_u0):
        self.C0_percent = new_u0
        self.create_other_variables()
        
    @property
    def sea_surface_elevation(self):
        ''' Get stage '''
        return self.stage
        
    @property
    def water_depth(self):
        ''' Get depth '''
        return self.depth
        
    @property
    def bed_elevation(self):
        ''' Get bed elevation '''
        return self.eta




            
            
        




