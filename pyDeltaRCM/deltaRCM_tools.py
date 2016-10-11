#! /usr/bin/env python
from math import floor, sqrt, pi
import numpy as np
from random import shuffle
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
from scipy import ndimage
import sys, os, re, string
from netCDF4 import Dataset
import time as time_lib
from scipy.sparse import lil_matrix, csc_matrix, hstack

class Tools(object):


    #############################################
    ############### randomization ###############
    #############################################

    def random_pick(self, probs):
        '''
        Randomly pick a number weighted by array probs (len 8)
        Return the index of the selected weight in array probs
        '''

        if np.max(probs) == 0:
            probs = np.array([1./8 for i in range(8)])

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
            if self.verbose: print 'Creating output directory'
            os.makedirs(directory)

        savepath = os.path.join(directory, filename)
        plt.savefig(savepath)

        if close: plt.close()
            
            
    def save_grids(self, var_name, var):
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
        
        timestep = self._time
        
        try:
            shape = self.output_netcdf[var_name].shape
            self.output_netcdf[var_name][shape[0],:,:] = var
            
            self.output_netcdf['time'][shape[0]] = timestep
            
        except:
            print 'Error: Cannot save grid to netCDF file.'
        
      
      
                  

    #############################################
    ############### weight arrays ###############
    #############################################

    def build_weight_array(self, array, fix_edges = False, normalize = False):
        '''
        Create np.array((8,L,W)) of quantity a in each of the neighbors to a cell
        '''

        a_shape = array.shape

        wgt_array = np.zeros((8, a_shape[0], a_shape[1]))
        nums = range(8)

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



    def get_wgt_sfc(self, wet_mask_nh):
        '''
        Get np.array((8,L,W)) (H - H_neighbor)/dist
        for each neighbor around a cell

        Takes an narray of the same size with 1 if wet and 0 if not
        '''

        wgt_sfc = self.build_weight_array(self.stage, fix_edges = True)

        wgt_sfc = (self.stage - wgt_sfc) / np.array(self.dxn_dist)[:, np.newaxis, np.newaxis]

        wgt_sfc = wgt_sfc * wet_mask_nh
        wgt_sfc[wgt_sfc<0] = 0

        wgt_sfc_sum = np.sum(wgt_sfc,axis=0)
        wgt_sfc[:,wgt_sfc_sum>0] = wgt_sfc[:,wgt_sfc_sum>0] / wgt_sfc_sum[wgt_sfc_sum>0]

        return wgt_sfc



    def get_wgt_int(self, wet_mask_nh):
        '''
        Returns np.array((8,L,W)) (qx*dxn_ivec + qy*dxn_jvec)/dist
        for each neighbor around a cell

        Takes an array of the same size with 1 if wet and 0 if dry
        '''

        wgt_int = (self.qx * np.array(self.dxn_ivec)[:,np.newaxis,np.newaxis] + \
            self.qy * np.array(self.dxn_jvec)[:,np.newaxis,np.newaxis]) / \
            np.array(self.dxn_dist)[:,np.newaxis,np.newaxis]

        wgt_int[1:4,0,:] = 0

        wgt_int = wgt_int * wet_mask_nh
        wgt_int[wgt_int<0] = 0
        wgt_int_sum = np.sum(wgt_int, axis=0)

        wgt_int[:,wgt_int_sum>0] = wgt_int[:,wgt_int_sum>0]/wgt_int_sum[wgt_int_sum>0]

        return wgt_int



    def get_wgt(self):
        '''
        Returns np.array((8,L,W)) of the probabilities of flow
        between a cell and each of its neighbors

        If the probabilities are zero in all directions, they will
        be split equally among all wet neighbors
        '''

        wet_mask_nh = self.get_wet_mask_nh()
        wgt_sfc = self.get_wgt_sfc(wet_mask_nh)
        wgt_int = self.get_wgt_int(wet_mask_nh)


        weight = self.gamma * wgt_sfc + (1-self.gamma) * wgt_int

        wgt = self.build_weight_array(self.depth, fix_edges = True)
        wgt = wgt**self.theta_water * weight

        wet_mask = 1*(self.depth > self.dry_depth)
        wgt = wgt * wet_mask
        wgt[wgt<0] = 0
        wgt_sum = np.sum(wgt,axis=0)
        wgt[:,wgt_sum>0] = wgt[:,wgt_sum>0] / wgt_sum[wgt_sum>0]

        # give wet cells with zero wgt to all wet neighbors equal probs for each of them
        # wet cells with zero probabilities to all neighbors
        wet_mask = 1*(self.depth > self.dry_depth)

        wet_cells = np.where((wgt_sum + (wet_mask-1)) == 0)

        wet = [(wet_cells[0][i],wet_cells[1][i]) for i in range(len(wet_cells[0]))]

        # new weights to those cells - partitioned equally among the wet neighbors
        new_vals = [0 if sum(wet_mask_nh[:,i[0],i[1]])==0 else wet_mask_nh[:,i[0],i[1]]/sum(wet_mask_nh[:,i[0],i[1]]) for i in wet]

        for i in range(len(new_vals)):
            wgt[:,wet[i][0],wet[i][1]] = new_vals[i]

        wgt[1:4,0,:] = 0

        return wgt



    def get_sed_weight(self):
        '''
        Get np.array((8,L,W)) of probability field of routing to neighbors
        for sediment parcels
        '''

        wet_mask_nh = self.get_wet_mask_nh()

        weight = self.get_wgt_int(wet_mask_nh) * \
            self.depth**self.theta_sand * wet_mask_nh

        weight[weight<0] = 0.
        weight_sum = np.sum(weight,axis=0)
        weight[:,weight_sum>0] = weight[:,weight_sum>0] / weight_sum[weight_sum>0]

        weight_f = np.zeros((self.L*self.W, 8))

        for i in range(8):
            weight_f[:,i] = weight[i,:,:].flatten()

        return weight_f



    #############################################
    ################# smoothing #################
    #############################################

    def smoothing_filter(self, stageTemp):
        '''
        Smooth water surface

        If any of the cells in a 9-cell window are wet, apply this filter

        stageTemp : water surface
        stageT : smoothed water surface
        '''

        stageT = stageTemp.copy()
        wet_mask = self.depth > self.dry_depth

        for t in range(self.Nsmooth):

            local_mean = ndimage.uniform_filter(stageT)

            stageT[wet_mask] = self.Csmooth * stageT[wet_mask] + \
                (1-self.Csmooth) * local_mean[wet_mask]

        returnval = (1-self.omega_sfc) * self.stage + self.omega_sfc * stageT

        return returnval



    def flooding_correction(self):
        '''
        Flood dry cells along the shore if necessary

        Check the neighbors of all dry cells. If any dry cells have wet neighbors
        Check that their stage is not higher than the bed elevation of the center cell
        If it is, flood the dry cell
        '''

        wet_mask = self.depth > self.dry_depth
        wet_mask_nh = self.get_wet_mask_nh()
        wet_mask_nh_sum = np.sum(wet_mask_nh, axis=0)

        # makes wet cells look like they have only dry neighbors
        wet_mask_nh_sum[wet_mask] = 0

        # indices of dry cells with wet neighbors
        shore_ind = np.where(wet_mask_nh_sum > 0)

        stage_nhs = self.build_weight_array(self.stage)
        eta_shore = self.eta[shore_ind]

        for i in range(len(shore_ind[0])):

            # pretends dry neighbor cells have stage zero so they cannot be > eta_shore[i]
            stage_nh = wet_mask_nh[:,shore_ind[0][i],shore_ind[1][i]] * \
                stage_nhs[:,shore_ind[0][i],shore_ind[1][i]]

            if (stage_nh > eta_shore[i]).any():
                self.stage[shore_ind[0][i],shore_ind[1][i]] = max(stage_nh)



    def topo_diffusion(self):
        '''
        Diffuse topography after routing all coarse sediment parcels
        '''

        wgt_cell_type = self.build_weight_array(self.cell_type > -2)
        wgt_qs = self.build_weight_array(self.qs) + self.qs
        wet_mask_nh = self.get_wet_mask_nh()

        multiplier = self.dt/self.N_crossdiff * self.alpha * 0.5 / self.dx**2

        for n in range(self.N_crossdiff):

            wgt_eta = self.build_weight_array(self.eta) - self.eta

            crossflux_nb = multiplier * wgt_qs * wgt_eta * wet_mask_nh
            
            crossflux = np.sum(crossflux_nb, axis=0)
            
            self.eta[:] = self.eta + crossflux



    #############################################
    ################# updaters ##################
    #############################################
                
                
    

    def update_flow_field(self, iteration):
        '''
        Update water discharge after one water iteration
        '''
        
        timestep = self._time

        dloc = (self.qxn**2 + self.qyn**2)**(0.5)
        
        qwn_div = np.ones((self.L,self.W))
        qwn_div[dloc>0] = self.qwn[dloc>0] / dloc[dloc>0]
        
        self.qxn *= qwn_div
        self.qyn *= qwn_div

        if timestep > 0:

            omega = self.omega_flow_iter
            if iteration == 0: omega = self.omega_flow

            self.qx = self.qxn*omega + self.qx*(1-omega)
            self.qy = self.qyn*omega + self.qy*(1-omega)

        else:

            self.qx = self.qxn.copy(); self.qy = self.qyn.copy()

        self.qw = (self.qx**2 + self.qy**2)**(0.5)
        self.qx[0,self.inlet] = self.qw0
        self.qy[0,self.inlet] = 0
        self.qw[0,self.inlet] = self.qw0



    def update_velocity_field(self):
        '''
        Update the flow velocity field after one water iteration
        '''

        mask = (self.depth > self.dry_depth) * (self.qw > 0)
        self.uw[mask] = np.minimum(self.u_max, self.qw[mask] / self.depth[mask])
        self.uw[~mask] = 0
        self.ux[mask]= self.uw[mask] * self.qx[mask] / self.qw[mask]
        self.ux[~mask] = 0
        self.uy[mask]= self.uw[mask] * self.qy[mask] / self.qw[mask]
        self.uy[~mask] = 0



    #############################################
    ################# water flow ################
    #############################################

    def init_water_iteration(self):

        wgt = self.get_wgt()

        for i in range(8):
            self.wgt_flat[:,i] = wgt[i,:,:].flatten()

        self.qxn[:] = 0; self.qyn[:] = 0; self.qwn[:] = 0

        self.indices = np.zeros((self.Np_water, self.size_indices), dtype = np.int)
        self.path_number = np.arange(self.Np_water)
        self.save_paths = []



    def run_water_iteration(self):
        '''
        Route all parcels of water in one iteration
        '''

        these_indices = map(lambda x: self.random_pick_inlet(self.inlet), range(self.Np_water))
        
        self.indices[:,0] = these_indices
        self.qxn.flat[these_indices] += 1

        water_continue = True
        it = 0

        while water_continue:

            ngh = map(self.random_pick, self.wgt_flat[these_indices])
            new_indices = these_indices + self.walk_flat[ngh]
            new_ind_type = self.cell_type.flat[new_indices]

            keepers = new_ind_type >= 0
            
            # save the path numbers of the ones that reached the edge or gone on land
            if self.path_number[~keepers].any():
                self.save_paths.append( list(self.path_number[~keepers]) )
                

            walk_vals = self.walk[ngh]
            self.qxn.flat[these_indices] += walk_vals[:,0]
            self.qyn.flat[these_indices] += walk_vals[:,1]
            
            ngh = np.array(ngh)[keepers]
            walk_vals = self.walk[ngh]
            
            these_indices = new_indices[keepers]
            self.path_number = self.path_number[keepers]
            
            self.qxn.flat[these_indices] += walk_vals[:,0]
            self.qyn.flat[these_indices] += walk_vals[:,1]

            it += 1

            # check for looping
            if len(self.path_number)>0:
                keeper = np.ones((len(these_indices),), dtype=np.int)
                for i in range(len(these_indices)):

                    if np.in1d(self.indices[self.path_number[i],:it], these_indices[i]).any():
                        keeper[i] = 0

                    if these_indices[i]<0:
                        keeper[i] = 0

                if np.min(keeper)==0:

                    these_indices = these_indices[keeper == 1]
                    self.path_number = self.path_number[keeper == 1]
                    
                    
                    
            if it >= self.indices.shape[1]:
                '''
                Initial size of self.indices is half of self.itmax because the number of
                iterations doesn't go beyond that for many timesteps.
                
                Once it reaches it > self.itmax/2 once, make the size self.iter for all
                further timesteps
                '''
            
                if self.verbose: print 'Increasing size of self.indices'
            
                indices_blank = np.zeros((self.Np_water, self.itmax/2), dtype = np.int)
                self.indices = np.hstack((self.indices, indices_blank))
                
                self.size_indices = self.itmax
            
                
                    
            self.indices[self.path_number,it] = these_indices

            if it == self.itmax-1 or len(these_indices)==0:
                water_continue = False

        # update qwn by counting the indices
        all_indices = self.indices.flatten()
        all_indices.sort()
        loc = np.where(all_indices>0)[0][0]
        ind_index_all = all_indices[loc:]

        ind_count = np.bincount(ind_index_all)
        ind_index = range(max(ind_index_all)+1)

        qwn_sum = ind_count[ind_index] * self.Qp_water/self.dx

        self.qwn.flat[ind_index] += qwn_sum



    def finalize_water_iteration(self, iteration):
        '''
        Finish updating flow fields
        Clean up at end of water iteration
        '''
        
        self.stage[:] = np.maximum(self.stage, self.H_SL)
        self.depth[:] = np.maximum(self.stage - self.eta, 0)

        self.update_flow_field(iteration)
        self.update_velocity_field()



    def get_profiles(self):
        '''
        Calculate the water surface profiles after routing flow parcels
        Update water surface array
        '''

        paths_for_profile = [i for j in self.save_paths for i in j]

        assert len(paths_for_profile) == len(set(paths_for_profile)), "save_paths has repeats!"

        # get all the unique indices in good paths
        unique_cells = list(set([j for i in paths_for_profile for j in list(set(self.indices[i]))]))
        try:
            unique_cells.remove(0)
        except:
            pass

        unique_cells.sort()

        # extract the values needed for the paths -- no need to do this for the entire space
        uw_unique = self.uw.flat[unique_cells]
        depth_unique = self.depth.flat[unique_cells]
        ux_unique = self.ux.flat[unique_cells]
        uy_unique = self.uy.flat[unique_cells]

        profile_mask = np.add(uw_unique > 0.5*self.u0, depth_unique < 0.1*self.h0)

        all_unique = zip(profile_mask,uw_unique,ux_unique,uy_unique)

        sfc_array = np.zeros((len(unique_cells),2))

        # make dictionaries to use as lookup tables
        lookup = {}
        self.sfc_change = {}

        for i in range(len(unique_cells)):
            lookup[unique_cells[i]] = all_unique[i]
            self.sfc_change[unique_cells[i]] = sfc_array[i]

        # process each profile
        for i in paths_for_profile:

            path = self.indices[i]
            path = path[np.where(path>0)]

            prf = [lookup[i][0] for i in path]

            # find the last True
            try:
                last_True = (len(prf) - 1) - prf[::-1].index(True)
                sub_path = path[:last_True]

                sub_path_unravel = np.unravel_index(sub_path, self.eta.shape)

                path_diff = np.diff(sub_path_unravel)
                ux_ = [lookup[i][2] for i in sub_path[:-1]]
                uy_ = [lookup[i][3] for i in sub_path[:-1]]
                uw_ = [lookup[i][1] for i in sub_path[:-1]]

                dH = self.S0 * (ux_ * path_diff[0] + uy_ * path_diff[1]) * self.dx
                dH = [dH[i] / uw_[i] if uw_[i]>0 else 0 for i in range(len(dH))]
                dH.append(0)

                newH = np.zeros(len(sub_path))
                for i in range(-2,-len(sub_path)-1,-1):
                    newH[i] = newH[i+1] + dH[i]

                for i in range(len(sub_path)):
                    self.sfc_change[sub_path[i]] += [newH[i],1]
            except:
                pass

        stageTemp = self.eta + self.depth

        for k, v in self.sfc_change.iteritems():
            if np.max(v) > 0:
                stageTemp.flat[k] = v[0]/v[1]

        self.stage[:] = self.smoothing_filter(stageTemp)



    #############################################
    ################# sed flow ##################
    #############################################

    def init_sed_timestep(self):
        '''
        Set up arrays to start sed routing timestep
        '''

        self.qs[:] = 0
        self.Vp_dep_sand[:] = 0
        self.Vp_dep_mud[:] = 0



    def one_fine_timestep(self):
        '''
        Route all parcels of fine sediment
        '''

        self.num_fine = int(self.Np_sed - self.num_coarse)

        if self.num_fine>0:

            these_indices = map(lambda x: self.random_pick_inlet(self.inlet),range(self.num_fine))
            
            self.indices = np.zeros((self.num_fine, self.itmax), dtype=np.int)
            self.indices[:,0] = these_indices

            path_number = np.arange(self.num_fine)
            self.Vp_res = np.zeros((self.Np_sed,)) + self.Vp_sed
            self.qs.flat[these_indices] += self.Vp_res[path_number]/2./self.dt/self.dx

            sed_continue = True
            it = 0

            while sed_continue:

                weight = self.get_sed_weight()

                ngh = map(self.random_pick, weight[these_indices])
                new_indices = these_indices + self.walk_flat[ngh]
                new_ind_type = self.cell_type.flat[new_indices]

                Vp_res_update = self.Vp_res[path_number]/2./self.dt/self.dx
                self.qs.flat[these_indices] += Vp_res_update
                self.qs.flat[new_indices] += Vp_res_update


                these_indices = new_indices[new_ind_type >= 0]
                path_number = path_number[new_ind_type >= 0]

                if len(path_number)>0:
                    # check for looping
                    keeper = np.ones((len(these_indices),), dtype=np.int)
                    for i in range(len(these_indices)):
                        if np.in1d(self.indices[path_number[i],:], these_indices[i]).any():
                            keeper[i] = 0
                        if these_indices[i]<0:
                            keeper[i] = 0
                    if np.min(keeper)==0:
                        these_indices = these_indices[keeper == 1]
                        path_number = path_number[keeper == 1]

                # save to the master indices
                it += 1
                self.indices[path_number,it] = these_indices

                UW_loc = self.uw.flat[these_indices]
                if (UW_loc < self.U_dep_mud).any():

                    update_ind = these_indices[UW_loc < self.U_dep_mud]
                    update_path = path_number[UW_loc < self.U_dep_mud]

                    Vp_res_ = (self.sed_lag * self.Vp_res[update_path] *
                    (self.U_dep_mud**self.beta - self.uw.flat[update_ind]**self.beta) /
                    (self.U_dep_mud**self.beta))

                    self.Vp_dep = (self.stage.flat[update_ind] - self.eta.flat[update_ind])/4. * self.dx**2
                    self.Vp_dep = np.array([min((Vp_res_[i],self.Vp_dep[i])) for i in range(len(self.Vp_dep))])
                    
                    self.Vp_dep_mud.flat[update_ind] += self.Vp_dep

                    self.Vp_res[update_path] -= self.Vp_dep

                    self.eta.flat[update_ind] += self.Vp_dep / self.dx**2
                    
                    self.depth.flat[update_ind] = self.stage.flat[update_ind] - self.eta.flat[update_ind]
                    
                    update_uw = [min(self.u_max, self.qw.flat[i]/self.depth.flat[i]) for i in update_ind]
                    self.uw.flat[update_ind] = update_uw

                    update_uwqw = [self.uw.flat[i]/self.qw.flat[i] if self.qw.flat[i]>0 else 0 for i in update_ind]
                    
                    self.ux.flat[update_ind] = self.qx.flat[update_ind] * update_uwqw
                    self.uy.flat[update_ind] = self.qy.flat[update_ind] * update_uwqw



                if (UW_loc > self.U_ero_mud).any():

                    update_ind = these_indices[UW_loc > self.U_ero_mud]
                    update_path = path_number[UW_loc > self.U_ero_mud]

                    Vp_res_ = self.Vp_sed * (self.uw.flat[update_ind]**self.beta - self.U_ero_mud**self.beta) / (self.U_ero_mud**self.beta)
                    self.Vp_ero = (self.stage.flat[update_ind] - self.eta.flat[update_ind])/4. * self.dx**2
                    self.Vp_ero = np.array([min((Vp_res_[i],self.Vp_ero[i])) for i in range(len(self.Vp_ero))])

                    self.eta.flat[update_ind] -= self.Vp_ero / self.dx**2

                    self.depth.flat[update_ind] = self.stage.flat[update_ind] - self.eta.flat[update_ind]
                    update_uw = [min(self.u_max, self.qw.flat[i]/self.depth.flat[i]) for i in update_ind]
                    self.uw.flat[update_ind] = update_uw

                    update_uwqw = [self.uw.flat[i]/self.qw.flat[i] if self.qw.flat[i]>0 else 0 for i in update_ind]
                    self.ux.flat[update_ind] = self.qx.flat[update_ind] * update_uwqw
                    self.uy.flat[update_ind] = self.qy.flat[update_ind] * update_uwqw

                    self.Vp_res[update_path] += self.Vp_ero


                if it == self.itmax-1 or len(these_indices)==0:
                    sed_continue = False



    def one_coarse_timestep(self):
        '''
        Route all parcels of coarse sediment
        '''

        self.num_coarse = int(round(self.Np_sed*self.f_bedload))

        if self.num_coarse>0:

            these_indices = map(lambda x: self.random_pick_inlet(self.inlet),range(self.num_coarse))

            self.indices = np.zeros((self.num_coarse,self.itmax), dtype=np.int)
            self.indices[:,0] = these_indices

            path_number = np.arange(self.num_coarse)
            self.Vp_res = np.zeros((self.Np_sed,)) + self.Vp_sed
            self.qs.flat[these_indices] += self.Vp_res[path_number]/2./self.dt/self.dx

            sed_continue = True
            it = 0

            while sed_continue:

                weight = self.get_sed_weight()

                ngh = map(self.random_pick, weight[these_indices])
                new_indices = these_indices + self.walk_flat[ngh]
                new_ind_type = self.cell_type.flat[new_indices]

                self.qs.flat[these_indices] += self.Vp_res[path_number]/2./self.dt/self.dx
                self.qs.flat[new_indices] += self.Vp_res[path_number]/2./self.dt/self.dx

                these_indices = new_indices[new_ind_type >= 0]
                path_number = path_number[new_ind_type >= 0]

                if len(path_number)>0:
                    # check for looping
                    keeper = np.ones((len(these_indices),), dtype=np.int)
                    for i in range(len(these_indices)):
                        if np.in1d(self.indices[path_number[i],:], these_indices[i]).any():
                            keeper[i] = 0
                        if these_indices[i]<0:
                            keeper[i] = 0
                    if np.min(keeper)==0:
                        these_indices = these_indices[keeper == 1]
                        path_number = path_number[keeper == 1]

                it += 1
                self.indices[path_number,it] = these_indices

                qs_cap = self.qs0 * self.f_bedload/self.u0**self.beta * self.uw.flat[these_indices]**self.beta

                qs_loc = self.qs.flat[these_indices]

                if (qs_loc > qs_cap).any():
                

                    update_ind = these_indices[qs_loc > qs_cap]
                    update_path = path_number[qs_loc > qs_cap]
                    Vp_res_ = self.Vp_res[update_path]

                    self.Vp_dep = self.depth.flat[update_ind]/4.* self.dx**2
                    self.Vp_dep = np.minimum(Vp_res_, self.Vp_dep)
             
                    self.Vp_res[update_path] -= self.Vp_dep

                    eta_change = self.Vp_dep / self.dx**2
                    self.eta.flat[update_ind] += eta_change
                    
                    self.depth.flat[update_ind] = self.stage.flat[update_ind] - self.eta.flat[update_ind]

                    update_uw = [min(self.u_max, self.qw.flat[i]/self.depth.flat[i]) for i in update_ind]
                    self.uw.flat[update_ind] = update_uw

                    update_uwqw = [self.uw.flat[i]/self.qw.flat[i] if self.qw.flat[i]>0 else 0 for i in update_ind]
                    self.ux.flat[update_ind] = self.qx.flat[update_ind] * update_uwqw
                    self.uy.flat[update_ind] = self.qy.flat[update_ind] * update_uwqw
                    


                if ((qs_loc < qs_cap) * (self.uw.flat[these_indices] > self.U_ero_sand)).any():
                

                    update_ind = these_indices[(qs_loc < qs_cap) * (self.uw.flat[these_indices] > self.U_ero_sand)]
                    update_path = path_number[(qs_loc < qs_cap) * (self.uw.flat[these_indices] > self.U_ero_sand)]

                    Vp_res_ = self.Vp_sed * (self.uw.flat[update_ind]**self.beta - self.U_ero_sand**self.beta) / (self.U_ero_sand**self.beta)
                    
                    
                    Vp_ero_ = self.depth.flat[update_ind]/4. * self.dx**2
                    self.Vp_ero = np.minimum(Vp_res_,Vp_ero_)
                    

                    eta_change = self.Vp_ero / self.dx**2
                    self.eta.flat[update_ind] -= eta_change
                    self.depth.flat[update_ind] = self.stage.flat[update_ind] - self.eta.flat[update_ind]


                    update_uw = [min(self.u_max, self.qw.flat[i]/self.depth.flat[i]) for i in update_ind]
                    self.uw.flat[update_ind] = update_uw

                    update_uwqw = [self.uw.flat[i]/self.qw.flat[i] if self.qw.flat[i]>0 else 0 for i in update_ind]
                    self.ux.flat[update_ind] = self.qx.flat[update_ind] * update_uwqw
                    self.uy.flat[update_ind] = self.qy.flat[update_ind] * update_uwqw

                    self.Vp_res[update_path] += self.Vp_ero
                    


                if it == self.itmax-1 or len(these_indices)==0:
                    sed_continue = False

        self.topo_diffusion()
        



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

        self.H_SL = self.H_SL + self.SLR * self.time_step
        



    #############################################
    ############## initialization ###############
    #############################################

    def get_var_name(self, long_var_name): 
        return self._var_name_map[ long_var_name ]



    def import_file(self):

        if self.verbose: print 'Reading input file...'

        self.input_file_vars = dict()
        numvars = 0

        o = open(self.input_file, mode = 'r')

        for line in o:
            line = re.sub('\s$','',line)
            line = re.sub('\A[: :]*','',line)
            ln = re.split('\s*[\:\=]\s*', line)

            if len(ln)>1:

                ln[0] = string.lower(ln[0])

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

                        ln[1] = string.lower(ln[1])

                        if ln[1] == 'yes' or ln[1] == 'true':
                            self.input_file_vars[str(ln[0])] = True
                        elif ln[1] == 'no' or ln[1] == 'false':
                            self.input_file_vars[str(ln[0])] = False
                        else:
                            print "Alert! The option for the 'choice' type variable " \
                                  "in the input file '" + str(ln[0]) + "' is unrecognized. " \
                                  "Please use only Yes/No or True/False as values.\n"

                else:
                    print "Alert! The input file contains an unknown entry. The variable '" \
                          + str(ln[0]) + "' is not an input variable for this model. Check " \
                          " the spelling of the variable name and only use the symbols : and = " \
                            "in variable assignments.\n"

        o.close()
        
        for k,v in self.input_file_vars.items():
            setattr(self, self.get_var_name(k), v)
        
        if self.verbose: print 'Finished reading ' + str(numvars) + ' variables from input file.'

 
        
    def set_defaults(self):
    
        for k,v in self._var_default_map.items(): setattr(self, self._var_name_map[k], v)



    def create_dicts(self):
                                   
        self._input_var_names = self._input_vars.keys()

        self._var_type_map = dict()
        self._var_name_map = dict()
        self._var_default_map = dict()

        for k in self._input_vars.keys():
            self._var_type_map[k] = self._input_vars[k]['type']
            self._var_name_map[k] = self._input_vars[k]['name']
            self._var_default_map[k] = self._input_vars[k]['default']



    def set_constants(self):

        self.g = 9.81   # (gravitation const.)
    
    
        self.dxn_iwalk = [1,1,0,-1,-1,-1,0,1]
        self.dxn_jwalk = [0,1,1,1,0,-1,-1,-1]
        self.dxn_dist = \
        [sqrt(self.dxn_iwalk[i]**2 + self.dxn_jwalk[i]**2) for i in range(8)]
    
        SQ05 = sqrt(0.5)
        self.dxn_ivec = [0,-SQ05,-1,-SQ05,0,SQ05,1,SQ05]
        self.dxn_jvec = [1,SQ05,0,-SQ05,-1,-SQ05,0,SQ05]

        self.walk_flat = np.array([1, -self.W+1, -self.W, -self.W-1, -1, self.W-1, self.W, self.W+1])
        self.walk = np.array([[0,1], [-SQ05, SQ05], [-1,0], [-SQ05,-SQ05], 
                              [0,-1], [SQ05,-SQ05], [1,0], [SQ05,SQ05]])
    
    
    def init_subsidence(self):
        '''
        Initializes patterns of subsidence if
        toggle_subsidence is True (default False)
        
        Modify the equations for self.subsidence_mask and self.sigma as desired
        '''
    
        if self.toggle_subsidence:
        
            R1 = 0.3 * self.L; R2 = 1. * self.L     # radial limits (fractions of L)
            theta1 = -pi/3; theta2 =  pi/3.         # angular limits
            
            Rloc = np.sqrt((self.y - self.L0)**2 + (self.x - self.W / 2.)**2)

            thetaloc = np.zeros((self.L, self.W))
            thetaloc[self.y > self.L0 - 1] = np.arctan((self.x[self.y > self.L0 - 1] - self.W / 2.) /
                                             (self.y[self.y > self.L0 - 1] - self.L0 + 1))
            
            self.subsidence_mask = ((R1 <= Rloc) & (Rloc <= R2) &
                                    (theta1 <= thetaloc) & (thetaloc <= theta2))
            
            self.subsidence_mask[:self.L0,:] = False
            
            self.sigma = self.subsidence_mask * self.sigma_max * self.time_step


      
        
    def create_other_variables(self):
    
        self.init_Np_water = self.Np_water
        self.init_Np_sed = self.Np_sed
    
        self.dx = float(self.dx)
    
        self.theta_sand = self.coeff_theta_sand * self.theta_water
        self.theta_mud = self.coeff_theta_mud * self.theta_water
    
        self.U_dep_mud = self.coeff_U_dep_mud * self.u0
        self.U_ero_sand = self.coeff_U_ero_sand * self.u0
        self.U_ero_mud = self.coeff_U_ero_mud * self.u0
    
        self.L0 = max(1, int(round(self.L0_meters / self.dx)))
        self.N0 = max(3, int(round(self.N0_meters / self.dx)))
    
        self.L = int(round(self.Length/self.dx))        # num cells in x
        self.W = int(round(self.Width/self.dx))         # num cells in y
        
        self.set_constants()

        self.u_max = 2.0 * self.u0              # maximum allowed flow velocity
    
        self.C0 = self.C0_percent * 1/100.      # sediment concentration

        # (m) critial depth to switch to "dry" node
        self.dry_depth = min(0.1, 0.1*self.h0)
        self.CTR = floor(self.W / 2.)

        self.gamma = self.g * self.S0 * self.dx / (self.u0**2)

        self.V0 = self.h0 * (self.dx**2)    # (m^3) reference volume (volume to
                                            # fill cell to characteristic depth)

        self.Qw0 = self.u0 * self.h0 * self.N0 * self.dx    # const discharge
                                                            # at inlet        
                                                                                                   
        self.qw0 = self.u0 * self.h0                # water unit input discharge
        self.Qp_water = self.Qw0 / self.Np_water    # volume each water parcel

        self.qs0 = self.qw0 * self.C0               # sed unit discharge

        self.dVs = 0.1 * self.N0**2 * self.V0       # total amount of sed added 
                                                    # to domain per timestep

        self.Qs0 = self.Qw0 * self.C0           # sediment total input discharge
        self.Vp_sed = self.dVs / self.Np_sed    # volume of each sediment parcel
    
        self.itmax = 2 * (self.L + self.W)      # max number of jumps for parcel
        self.size_indices = int(self.itmax/2)   # initial width of self.indices
        
        self.dt = self.dVs / self.Qs0           # time step size

        self.omega_flow = 0.9
        self.omega_flow_iter = 2. / self.itermax
 
        # number of times to repeat topo diffusion
        self.N_crossdiff = int(round(self.dVs / self.V0))
 
    
        # self.prefix
        self.prefix = self.out_dir
        
        if self.out_dir[-1] is not '/':
            self.prefix = self.out_dir + '/'
        
        if self.site_prefix:
            self.prefix += self.site_prefix + '_'
        if self.case_prefix:
            self.prefix += self.case_prefix + '_'



    def create_domain(self):
        '''
        Creates the model domain
        '''

        ##### empty arrays #####

        self.x, self.y = np.meshgrid(np.arange(0,self.W), np.arange(0,self.L))
    
        self.cell_type = np.zeros((self.L,self.W))
    
        self.eta = np.zeros((self.L,self.W)).astype(np.float32)
        self.stage = np.zeros((self.L,self.W))
        self.depth = np.zeros((self.L,self.W))

        self.qx = np.zeros((self.L,self.W))
        self.qy = np.zeros((self.L,self.W))
        self.qxn = np.zeros((self.L,self.W))
        self.qyn = np.zeros((self.L,self.W))
        self.qwn = np.zeros((self.L,self.W))
        self.ux = np.zeros((self.L,self.W))
        self.uy = np.zeros((self.L,self.W))
        self.uw = np.zeros((self.L,self.W))
    
        self.wgt_flat = np.zeros((self.L*self.W, 8))

        self.qs = np.zeros((self.L,self.W))
        self.Vp_dep_sand = np.zeros((self.L,self.W))
        self.Vp_dep_mud = np.zeros((self.L,self.W))


        ##### domain #####
        cell_land = 2
        cell_channel = 1
        cell_ocean = 0
        cell_edge = -1
        
        self.cell_type[:self.L0,:] = cell_land
        
        channel_inds = int(self.CTR - round(self.N0 / 2))
        self.cell_type[:self.L0, channel_inds:channel_inds + self.N0 + 1] = cell_channel

        self.stage[:] = np.maximum(0, self.L0 - self.y - 1) * self.dx * self.S0
        self.stage[self.cell_type == cell_ocean] = 0.
        
        self.depth[self.cell_type == cell_ocean] = self.h0
        self.depth[self.cell_type == cell_channel] = self.h0

        self.qx[self.cell_type == cell_channel] = self.qw0
        self.qx[self.cell_type == cell_ocean] = self.qw0 / 5.
        self.qw = (self.qx**2 + self.qy**2)**(0.5)

        self.ux[self.depth>0] = self.qx[self.depth>0] / self.depth[self.depth>0]
        self.uy[self.depth>0] = self.qy[self.depth>0] / self.depth[self.depth>0]
        self.uw[self.depth>0] = self.qw[self.depth>0] / self.depth[self.depth>0]

        self.cell_type[self.cell_type == cell_land] = -2   # reset the land cell_type to -2
        self.cell_type[-1,:] = cell_edge
        self.cell_type[:,0] = cell_edge
        self.cell_type[:,-1] = cell_edge
    
        self.inlet = list(np.unique(np.where(self.cell_type == 1)[1]))
        self.eta[:] = self.stage - self.depth
        
        self.clim_eta = (-self.h0 - 1, 0.05)
        
    
    
    def init_stratigraphy(self):
        '''
        Creates sparse array to store stratigraphy data
        '''
        
        if self.save_strata:
        
            self.n_steps = 10 * self.save_dt
        
            self.strata_sand_frac = lil_matrix((self.L * self.W, self.n_steps), dtype=np.float32)
            
            self.init_eta = self.eta.copy()
            self.strata_eta = lil_matrix((self.L * self.W, self.n_steps), dtype=np.float32)


    def expand_stratigraphy(self):
        '''
        Expand the size of arrays that store stratigraphy data
        '''
        
        if self.verbose: print 'Expanding stratigraphy arrays'
        
        lil_blank = lil_matrix((self.L * self.W, self.n_steps), dtype=np.float32)
        
        self.strata_eta = hstack([self.strata_eta, lil_blank], format='lil')
        self.strata_sand_frac = hstack([self.strata_sand_frac, lil_blank], format='lil')

            
        
    def init_output_grids(self):
        '''
        Creates a netCDF file to store output grids
        Fills with default variables
        
        Overwrites an existing netcdf file with the same name
        '''
        
        if self.save_eta_grids or self.save_depth_grids or self.save_stage_grids or self.save_strata:
        
            if self.verbose: print 'Generating netCDF file for output grids...'
            
            directory = self.prefix
            filename = 'pyDeltaRCM_output.nc'

            if not os.path.exists(directory):
                if self.verbose: print 'Creating output directory'
                os.makedirs(directory)

            file_path = os.path.join(directory, filename)

            if os.path.exists(file_path):
                if self.verbose: print '*** Replaced existing netCDF file ***'
                os.remove(file_path)

            self.output_netcdf = Dataset(file_path, 'w', format='NETCDF4_CLASSIC')

            self.output_netcdf.description = 'Output grids from pyDeltaRCM'
            self.output_netcdf.history = 'Created ' + time_lib.ctime(time_lib.time())
            self.output_netcdf.source = 'pyDeltaRCM / CSDMS'

            length = self.output_netcdf.createDimension('length', self.L)
            width = self.output_netcdf.createDimension('width', self.W)
            total_time = self.output_netcdf.createDimension('total_time', None)
            
                

            x = self.output_netcdf.createVariable('x', np.float32, ('length','width'))
            y = self.output_netcdf.createVariable('y', np.float32, ('length','width'))
            time = self.output_netcdf.createVariable('time', np.int32, ('total_time',))

            x.units = 'meters'
            y.units = 'meters'
            time.units = 'timesteps'

            x[:] = self.x
            y[:] = self.y
            
                           
            if self.save_eta_grids:
                eta = self.output_netcdf.createVariable('eta',
                                                        np.float32,
                                                        ('total_time','length','width'))
                eta.units = 'meters'
                           
                    
            if self.save_stage_grids:
                stage = self.output_netcdf.createVariable('stage',
                                                          np.float32,
                                                          ('total_time','length','width'))
                stage.units = 'meters'
                           
                    
            if self.save_depth_grids:
                depth = self.output_netcdf.createVariable('depth',
                                                          np.float32,
                                                          ('total_time','length','width'))
                depth.units = 'meters'
                
                
                
            if self.verbose: print 'Output netCDF file created.'



    #############################################
    ############# run_one_timestep ##############
    #############################################

    def run_one_timestep(self):
        '''
        Run the time loop once
        '''
        
        timestep = self._time

        if self.verbose:
            print '-'*20
            print 'Time = ' + str(self._time)


        for iteration in range(self.itermax):

            self.init_water_iteration()
            
            self.run_water_iteration()
            
            ########################################
            # Not calculating water surface profiles
            ########################################
#             if timestep>0:
#                 self.get_profiles()

            self.finalize_water_iteration(iteration)
            

        self.init_sed_timestep()

        self.one_coarse_timestep()
        self.one_fine_timestep()

        

        
        
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
                print 'Storing stratigraphy data'
                
            ################### sand frac ###################
            # -1 for cells with deposition volumes < vol_limit
            # vol_limit for any mud (to diff from no deposition in sparse array)
            # (overwritten if any sand deposited)
            
            sand_frac = -1 * np.ones((self.L, self.W))

            vol_limit = 0.000001 # threshold deposition volume
            sand_frac[self.Vp_dep_mud > vol_limit] = vol_limit

            sand_loc = self.Vp_dep_sand > 0
            sand_frac[sand_loc] = (self.Vp_dep_sand[sand_loc] /
                                  (self.Vp_dep_mud[sand_loc] + self.Vp_dep_sand[sand_loc]))

            # store indices and sand_frac into a sparse array
            row_s = np.where(sand_frac.flatten() >= 0)[0]
            col_s = np.zeros((len(row_s),))
            data_s = sand_frac[sand_frac >= 0]

            sand_sparse = csc_matrix((data_s, (row_s, col_s)), shape=(self.L * self.W, 1))

            # store sand_sparse into strata_sand_frac
            self.strata_sand_frac[:,timestep] = sand_sparse
            
            
            ################### eta ###################
            
            diff_eta = self.eta - self.init_eta
            
            row_s = np.where(diff_eta.flatten() != 0)[0]
            col_s = np.zeros((len(row_s),))
            data_s = self.eta[diff_eta != 0]
           
            eta_sparse = csc_matrix((data_s, (row_s, col_s)), shape=(self.L * self.W, 1))
            
            self.strata_eta[:,timestep] = eta_sparse
            
            if self.toggle_subsidence and self.start_subsidence <= timestep:
            
                sigma_change = self.strata_eta[:,:timestep] - self.sigma.flatten()[:,np.newaxis]
                self.strata_eta[:,:timestep] = lil_matrix(sigma_change)
            
        
        
        
        
        
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
                    print 'Applying subsidence'
            
                self.eta[:] = self.eta - self.sigma
                
        

    def output_data(self):
        '''
        Plots and saves figures of eta, depth, and stage
        '''
        
        timestep = self._time

        if timestep % self.save_dt == 0:
            
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
                
                
            ############ GRIDS #############
            if self.save_eta_grids:
                if self.verbose: print 'Saving grid: eta'
                self.save_grids('eta', self.eta)
            
            if self.save_depth_grids:
                if self.verbose: print 'Saving grid: depth'  
                self.save_grids('depth', self.depth)

            if self.save_stage_grids:
                if self.verbose: print 'Saving grid: stage'
                self.save_grids('stage', self.stage)                
    
    
    
    
    def output_strata(self):
        '''
        Saves the stratigraphy sparse matrices into output netcdf file
        '''
        
        if self.save_strata:
        
            if self.verbose:
                print '\nSaving final stratigraphy data to netCDF files...'
           
               
            shape = self.strata_eta.shape
           
            total_strata_age = self.output_netcdf.createDimension('total_strata_age', shape[1])
            

            strata_age = self.output_netcdf.createVariable('strata_age',
                                                            np.int32,
                                                            ('total_strata_age'))
            strata_age.units = 'timesteps'
            self.output_netcdf['strata_age'][:] = range(shape[1]-1, -1, -1)


            sand_frac = self.output_netcdf.createVariable('strata_sand_frac',
                                                          np.float32,
                                                          ('total_strata_age','length','width'))
            sand_frac.units = 'fraction'


            strata_elev = self.output_netcdf.createVariable('strata_depth',
                                                          np.float32,
                                                          ('total_strata_age','length','width'))
            strata_elev.units = 'meters'



            for i in range(shape[1]):

                sf = self.strata_sand_frac[:,i].toarray().reshape(self.eta.shape)
                sf[sf == 0] = -1

                self.output_netcdf['strata_sand_frac'][i,:,:] = sf

                sz = self.strata_eta[:,i].toarray().reshape(self.eta.shape)
                sz[sz == 0] = self.init_eta[sz == 0]

                self.output_netcdf['strata_depth'][i,:,:] = sz


            if self.verbose:
                print 'Stratigraphy data saved.'
        
        
        
        
        