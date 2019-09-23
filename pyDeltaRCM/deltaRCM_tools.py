#! /usr/bin/env python
from math import floor, sqrt, pi
import numpy as np
from random import shuffle
import matplotlib
from matplotlib import pyplot as plt
from scipy import ndimage
import sys, os, re, string
from netCDF4 import Dataset
import time as time_lib
from scipy.sparse import lil_matrix, csc_matrix, hstack
import logging
import time
import sed_tools, water_tools, init_tools

np.random.seed(0)

class Tools(sed_tools, water_tools, init_tools, object):

    #############################################
    ############# run_one_timestep ##############
    #############################################

    def run_one_timestep(self):
        '''
        Run the time loop once
        '''

        timestep = self._time

        if self.verbose:
            print('-'*20)
            print('Time = ' + str(self._time))

        for iteration in range(self.itermax):

            self.count = 0

            self.init_water_iteration()
            self.run_water_iteration()

            self.free_surf(iteration)
            self.finalize_water_iteration(timestep,iteration)

        self.sed_route()

    def finalize_timestep(self):
        '''
        Clean up after sediment routing
        Update sea level if baselevel changes
        '''

        self.flooding_correction()
        self.stage[:] = np.maximum(self.stage, self.H_SL)
        self.depth[:] = np.maximum(self.stage - self.eta, 0)

        self.eta[0,self.inlet] = self.stage[0, self.inlet] - self.h0
        self.depth[0,self.inlet] = self.h0

        self.H_SL = self.H_SL + self.SLR * self.dt

    #################################

    def step_update(self, ind, new_ind, new_cell):

        istep = self.iwalk.flat[new_cell]
        jstep = self.jwalk.flat[new_cell]
        dist = np.sqrt(istep**2 + jstep**2)

        if dist > 0:

            self.qxn.flat[ind] += jstep / dist
            self.qyn.flat[ind] += istep / dist
            self.qwn.flat[ind] += self.Qp_water / self.dx / 2.

            self.qxn.flat[new_ind] += jstep / dist
            self.qyn.flat[new_ind] += istep / dist
            self.qwn.flat[new_ind] += self.Qp_water / self.dx / 2.

        return dist

    def calculate_new_ind(self, ind, new_cell):

        new_ind = (ind[0] + self.jwalk.flat[new_cell], ind[1] +
                   self.iwalk.flat[new_cell])

        new_ind_flat = np.ravel_multi_index(new_ind, self.depth.shape,mode='wrap') # added wrap mode to fct to resolve ValueError due to negative numbers

        return new_ind_flat

    def get_weight(self, ind):

        stage_ind = self.pad_stage[ind[0]-1+1:ind[0]+2+1, ind[1]-1+1:ind[1]+2+1]

        weight_sfc = np.maximum(0,
                     (self.stage[ind] - stage_ind) / self.distances)

        weight_int = np.maximum(0, (self.qx[ind] * self.jvec +
                                    self.qy[ind] * self.ivec) / self.distances)

        if ind[0] == 0:
            weight_sfc[0,:] = 0
            weight_int[0,:] = 0

        depth_ind = self.pad_depth[ind[0]-1+1:ind[0]+2+1, ind[1]-1+1:ind[1]+2+1]
        ct_ind = self.pad_cell_type[ind[0]-1+1:ind[0]+2+1, ind[1]-1+1:ind[1]+2+1]

        weight_sfc[(depth_ind <= self.dry_depth) | (ct_ind == -2)] = 0
        weight_int[(depth_ind <= self.dry_depth) | (ct_ind == -2)] = 0

        if np.nansum(weight_sfc) > 0:
            weight_sfc = weight_sfc / np.nansum(weight_sfc)

        if np.nansum(weight_int) > 0:
            weight_int = weight_int / np.nansum(weight_int)

        self.weight = self.gamma * weight_sfc + (1 - self.gamma) * weight_int
        self.weight = depth_ind ** self.theta_water * self.weight
        self.weight[depth_ind <= self.dry_depth] = np.nan

        new_cell = self.random_pick(self.weight)

        return new_cell

    #############################################
    ############### randomization ###############
    #############################################

    def random_pick(self, probs):
        '''
        Randomly pick a number weighted by array probs (len 8)
        Return the index of the selected weight in array probs
        '''

        num_nans = sum(np.isnan(probs))

        if np.nansum(probs) == 0:
            probs[~np.isnan(probs)] = 1
            probs[1,1] = 0

        probs[np.isnan(probs)] = 0
        cutoffs = np.cumsum(probs)
        idx = cutoffs.searchsorted(np.random.uniform(0, cutoffs[-1]))

        return idx

    def random_pick_inlet(self, choices, probs = None):
        '''
        Randomly pick a number from array choices weighted by array probs
        Values in choices are column indices

        Return a tuple of the randomly picked index for row 0
        '''

        if not probs:
            probs = np.array([1 for i in range(len(choices))])

        cutoffs = np.cumsum(probs)
        idx = cutoffs.searchsorted(np.random.uniform(0, cutoffs[-1]))

        return choices[idx]

    #############################################
    ############### weight arrays ###############
    #############################################

    def build_weight_array(self, array, fix_edges = False, normalize = False):
        '''
        Create np.array((8,L,W)) of quantity a
        in each of the neighbors to a cell
        '''

        a_shape = array.shape

        wgt_array = np.zeros((8, a_shape[0], a_shape[1]))
        nums = list(range(8))

        wgt_array[nums[0],:,:-1] = array[:,1:] # E
        wgt_array[nums[1],1:,:-1] = array[:-1,1:] # NE
        wgt_array[nums[2],1:,:] = array[:-1,:] # N
        wgt_array[nums[3],1:,1:] = array[:-1,:-1] # NW
        wgt_array[nums[4],:,1:] = array[:,:-1] # W
        wgt_array[nums[5],:-1,1:] = array[1:,:-1] # SW
        wgt_array[nums[6],:-1,:] = array[1:,:] # S
        wgt_array[nums[7],:-1,:-1] = array[1:,1:] # SE

        if fix_edges:
            wgt_array[nums[0],:,-1] = wgt_array[nums[0],:,-2]
            wgt_array[nums[1],:,-1] = wgt_array[nums[1],:,-2]
            wgt_array[nums[7],:,-1] = wgt_array[nums[7],:,-2]
            wgt_array[nums[1],0,:] = wgt_array[nums[1],1,:]
            wgt_array[nums[2],0,:] = wgt_array[nums[2],1,:]
            wgt_array[nums[3],0,:] = wgt_array[nums[3],1,:]
            wgt_array[nums[3],:,0] = wgt_array[nums[3],:,1]
            wgt_array[nums[4],:,0] = wgt_array[nums[4],:,1]
            wgt_array[nums[5],:,0] = wgt_array[nums[5],:,1]
            wgt_array[nums[5],-1,:] = wgt_array[nums[5],-2,:]
            wgt_array[nums[6],-1,:] = wgt_array[nums[6],-2,:]
            wgt_array[nums[7],-1,:] = wgt_array[nums[7],-2,:]

        if normalize:
            a_sum = np.sum(wgt_array, axis=0)
            wgt_array[:,a_sum!=0] = wgt_array[:,a_sum!=0] / a_sum[a_sum!=0]

        return wgt_array

    def get_wet_mask_nh(self):
        '''
        Returns np.array((8,L,W)), for each neighbor around a cell
        with 1 if the neighbor is wet and 0 if dry
        '''

        wet_mask = (self.depth > self.dry_depth) * 1
        wet_mask_nh = self.build_weight_array(wet_mask, fix_edges = True)

        return wet_mask_nh

    #############################################
    ################# updaters ##################
    #############################################

    #############################################
    ############## initialization ###############
    #############################################

    def get_var_name(self, long_var_name):
        return self._var_name_map[ long_var_name ]

    def import_file(self):

        self.input_file_vars = dict()
        numvars = 0

        o = open(self.input_file, mode = 'r')

        for line in o:
            line = re.sub('\s$','',line)
            line = re.sub('\A[: :]*','',line)
            ln = re.split('\s*[\:\=]\s*', line)

            if len(ln)>1:

                ln[0] = str.lower(ln[0])

                if ln[0] in self._input_var_names:

                    numvars += 1

                    var_type = self._var_type_map[ln[0]]

                    ln[1] = re.sub('[: :]+$','',ln[1])

                    if var_type == 'string':
                        self.input_file_vars[str(ln[0])] = str(ln[1])
                    if var_type == 'float':
                        self.input_file_vars[str(ln[0])] = float(ln[1])
                    if var_type == 'long':
                        self.input_file_vars[str(ln[0])] = int(ln[1])
                    if var_type == 'choice':

                        ln[1] = str.lower(ln[1])

                        if ln[1] == 'yes' or ln[1] == 'true':
                            self.input_file_vars[str(ln[0])] = True
                        elif ln[1] == 'no' or ln[1] == 'false':
                            self.input_file_vars[str(ln[0])] = False
                        else:
                            print("Alert! Options for 'choice' type variables "\
                                  "are only Yes/No or True/False.\n")

                else:
                    print("Alert! The input file contains an unknown entry.")

        o.close()

        for k,v in list(self.input_file_vars.items()):
            setattr(self, self.get_var_name(k), v)


    def expand_stratigraphy(self):
        '''
        Expand the size of arrays that store stratigraphy data
        '''

        if self.verbose: self.logger.info('Expanding stratigraphy arrays')

        lil_blank = lil_matrix((self.L * self.W, self.n_steps),
                                dtype=np.float32)

        self.strata_eta = hstack([self.strata_eta, lil_blank], format='lil')
        self.strata_sand_frac = hstack([self.strata_sand_frac, lil_blank],
                                        format='lil')


    def record_stratigraphy(self):
        '''
        Saves the sand fraction of deposited sediment
        into a sparse array created by init_stratigraphy().

        Only runs if save_strata is True
        '''

        timestep = self._time

        if self.save_strata and (timestep % self.save_dt == 0):

            timestep = int(timestep)

            if self.strata_eta.shape[1] <= timestep:
                self.expand_stratigraphy()

            if self.verbose:
                self.logger.info('Storing stratigraphy data')

            ################### sand frac ###################
            # -1 for cells with deposition volumes < vol_limit
            # vol_limit for any mud (to diff from no deposition in sparse array)
            # (overwritten if any sand deposited)

            sand_frac = -1 * np.ones((self.L, self.W))

            vol_limit = 0.000001 # threshold deposition volume
            sand_frac[self.Vp_dep_mud > vol_limit] = vol_limit

            sand_loc = self.Vp_dep_sand > 0
            sand_frac[sand_loc] = (self.Vp_dep_sand[sand_loc] /
                                  (self.Vp_dep_mud[sand_loc] +
                                  self.Vp_dep_sand[sand_loc]))
            # store indices and sand_frac into a sparse array
            row_s = np.where(sand_frac.flatten() >= 0)[0]
            col_s = np.zeros((len(row_s),))
            data_s = sand_frac[sand_frac >= 0]

            sand_sparse = csc_matrix((data_s, (row_s, col_s)),
                                      shape=(self.L * self.W, 1))
            # store sand_sparse into strata_sand_frac
            self.strata_sand_frac[:,self.strata_counter] = sand_sparse

            ################### eta ###################

            diff_eta = self.eta - self.init_eta

            row_s = np.where(diff_eta.flatten() != 0)[0]
            col_s = np.zeros((len(row_s),))
            data_s = self.eta[diff_eta != 0]

            eta_sparse = csc_matrix((data_s, (row_s, col_s)),
                                    shape=(self.L * self.W, 1))
            self.strata_eta[:,self.strata_counter] = eta_sparse

            if self.toggle_subsidence and self.start_subsidence <= timestep:

                sigma_change = (self.strata_eta[:,:self.strata_counter] -
                                self.sigma.flatten()[:,np.newaxis])
                self.strata_eta[:,:self.strata_counter] = lil_matrix(sigma_change)

            self.strata_counter += 1

    def apply_subsidence(self):
        '''
        Apply subsidence to domain if
        toggle_subsidence is True and
        start_subsidence is <= timestep
        '''

        if self.toggle_subsidence:

            timestep = self._time

            if self.start_subsidence <= timestep:
                if self.verbose:
                    self.logger.info('Applying subsidence')
                self.eta[:] = self.eta - self.sigma

    def output_data(self):
        '''
        Plots and saves figures of eta, depth, and stage
        '''

        timestep = self._time

        if timestep % self.save_dt == 0:

            if (self.save_eta_grids or
            self.save_depth_grids or
            self.save_stage_grids or
            self.save_discharge_grids or
            self.save_velocity_grids or
            self.save_strata):

                timestep = self._time
                shape = self.output_netcdf.variables['time'].shape
                self.output_netcdf.variables['time'][shape[0]] = timestep

            ############ FIGURES #############
            if self.save_eta_figs:

                plt.pcolor(self.eta)
                plt.clim(self.clim_eta[0], self.clim_eta[1])
                plt.colorbar()
                plt.axis('equal')
                self.save_figure(self.prefix + "eta_" + str(timestep))

            if self.save_stage_figs:

                plt.pcolor(self.stage)
                plt.colorbar()
                plt.axis('equal')
                self.save_figure(self.prefix + "stage_" + str(timestep))

            if self.save_depth_figs:

                plt.pcolor(self.depth)
                plt.colorbar()
                plt.axis('equal')
                self.save_figure(self.prefix + "depth_" + str(timestep))

            if self.save_discharge_figs:

                plt.pcolor(self.qw)
                plt.colorbar()
                plt.axis('equal')
                self.save_figure(self.prefix + "discharge_" + str(timestep))

            if self.save_velocity_figs:
                plt.pcolor(self.uw)
                plt.colorbar()
                plt.axis('equal')
                self.save_figure(self.prefix + "velocity_" + str(timestep))

            ############ GRIDS #############
            if self.save_eta_grids:
                if self.verbose: self.logger.info('Saving grid: eta')
                self.save_grids('eta', self.eta, shape[0])

            if self.save_depth_grids:
                if self.verbose: self.logger.info('Saving grid: depth')
                self.save_grids('depth', self.depth, shape[0])

            if self.save_stage_grids:
                if self.verbose: self.logger.info('Saving grid: stage')
                self.save_grids('stage', self.stage, shape[0])

            if self.save_discharge_grids:
                if self.verbose: self.logger.info('Saving grid: discharge')
                self.save_grids('discharge', self.qw, shape[0])

            if self.save_velocity_grids:
                if self.verbose: self.logger.info('Saving grid: velocity')
                self.save_grids('velocity', self.uw, shape[0])

    def output_strata(self):
        '''
        Saves the stratigraphy sparse matrices into output netcdf file
        '''

        if self.save_strata:

            if self.verbose:
                self.logger.info('\nSaving final stratigraphy to netCDF file')

            self.strata_eta = self.strata_eta[:, :self.strata_counter]

            shape = self.strata_eta.shape

            total_strata_age = self.output_netcdf.createDimension(
                                                            'total_strata_age',
                                                             shape[1])

            strata_age = self.output_netcdf.createVariable('strata_age',
                                                        np.int32,
                                                        ('total_strata_age'))
            strata_age.units = 'timesteps'
            self.output_netcdf.variables['strata_age'][:] = list(range(shape[1]-1,-1, -1))

            sand_frac = self.output_netcdf.createVariable('strata_sand_frac',
                                         np.float32,
                                        ('total_strata_age','length','width'))
            sand_frac.units = 'fraction'

            strata_elev = self.output_netcdf.createVariable('strata_depth',
                                           np.float32,
                                          ('total_strata_age','length','width'))
            strata_elev.units = 'meters'

            for i in range(shape[1]):

                sf = self.strata_sand_frac[:,i].toarray()
                sf = sf.reshape(self.eta.shape)
                sf[sf == 0] = -1

                self.output_netcdf.variables['strata_sand_frac'][i,:,:] = sf

                sz = self.strata_eta[:,i].toarray().reshape(self.eta.shape)
                sz[sz == 0] = self.init_eta[sz == 0]

                self.output_netcdf.variables['strata_depth'][i,:,:] = sz


            if self.verbose:
                self.logger.info('Stratigraphy data saved.')

    #############################################
    ################## output ###################
    #############################################

    def save_figure(self, path, ext='png', close=True):
        '''
        Save a figure.

        path : string
            The path (and filename without extension) to save the figure to.
        ext : string (default='png')
            The file extension. This must be supported by the active
            matplotlib backend (see matplotlib.backends module).  Most
            backends support 'png', 'pdf', 'ps', 'eps', and 'svg'.
        '''

        directory = os.path.split(path)[0]
        filename = "%s.%s" % (os.path.split(path)[1], ext)
        if directory == '': directory = '.'

        if not os.path.exists(directory):
            if self.verbose:
                self.logger.info('Creating output directory')
            os.makedirs(directory)

        savepath = os.path.join(directory, filename)
        plt.savefig(savepath)

        if close: plt.close()

    def save_grids(self, var_name, var, ts):
        '''
        Save a grid into an existing netCDF file.
        File should already be open (by init_output_grid) as self.output_netcdf

        var_name : string
                The name of the variable to be saved
        var : object
                The numpy array to be saved
        timestep : int
                The current timestep (+1, so human readable)
        '''

        try:

            self.output_netcdf.variables[var_name][ts,:,:] = var

        except:
            self.logger.info('Error: Cannot save grid to netCDF file.')

    def check_size_of_indices_matrix(self, it):
        if it >= self.indices.shape[1]:
            '''
            Initial size of self.indices is half of self.itmax
            because the number of iterations doesn't go beyond
            that for many timesteps.

            Once it reaches it > self.itmax/2 once, make the size
            self.iter for all further timesteps
            '''

            if self.verbose:
                self.logger.info('Increasing size of self.indices')

            indices_blank = np.zeros((np.int(self.Np_water), np.int(self.itmax/4)), dtype=np.int)

            self.indices = np.hstack((self.indices, indices_blank))
