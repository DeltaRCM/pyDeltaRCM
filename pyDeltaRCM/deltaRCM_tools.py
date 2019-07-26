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

np.random.seed(0)

class Tools(object):



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




    def sed_route(self):
        '''route all sediment'''

        self.pad_depth = np.pad(self.depth, 1, 'edge')

        self.qs[:] = 0
        self.Vp_dep_sand[:] = 0
        self.Vp_dep_mud[:] = 0

        self.sand_route()
        self.mud_route()

        self.topo_diffusion()






    def deposit(self, Vp_dep, px, py):
        '''deposit sand or mud'''

        eta_change_loc = Vp_dep / self.dx**2

        self.eta[px,py] = self.eta[px,py] + eta_change_loc #update bed
        self.depth[px,py] = self.stage[px,py] - self.eta[px,py]



        if self.depth[px,py] < 0:
            self.depth[px,py] = 0

        self.pad_depth[px+1,py+1] = self.depth[px,py]

        if self.depth[px,py] > 0:
            self.uw[px,py] = min(self.u_max, self.qw[px,py] / self.depth[px,py])

        self.update_u(px,py)

        self.Vp_res = self.Vp_res - Vp_dep
        #update amount of sediment left in parcel





    def erode(self, Vp_ero, px, py):
        '''erode sand or mud
        total sediment mass is preserved but individual categories
        of sand and mud are not'''


        eta_change_loc = -Vp_ero / (self.dx**2)

        self.eta[px,py] = self.eta[px,py] + eta_change_loc
        self.depth[px,py] = self.stage[px,py] - self.eta[px,py]


        if self.depth[px,py] < 0:
            self.depth[px,py] = 0

        self.pad_depth[px+1,py+1] = self.depth[px,py]


        if self.depth[px,py] > 0:
            self.uw[px,py] = min(self.u_max, self.qw[px,py] / self.depth[px,py])

        self.update_u(px,py)

        self.Vp_res = self.Vp_res + Vp_ero



    def update_u(self, px, py):
        '''update velocities after erosion or deposition'''

        if self.qw[px,py] > 0:

            self.ux[px,py] = self.uw[px,py] * self.qx[px,py] / self.qw[px,py]
            self.uy[px,py] = self.uw[px,py] * self.qy[px,py] / self.qw[px,py]

        else:
            self.ux[px,py] = 0
            self.uy[px,py] = 0



    def sand_dep_ero(self, px, py):
        '''decide if erode or deposit sand'''

        U_loc = self.uw[px,py]

        qs_cap = (self.qs0 * self.f_bedload / self.u0**self.beta *
                  U_loc**self.beta)

        qs_loc = self.qs[px,py]

        Vp_dep = 0
        Vp_ero = 0



        if qs_loc > qs_cap:
            #if more sed than transport capacity has gone through cell
            #deposit sand

            Vp_dep = min(self.Vp_res,
                        (self.stage[px,py] - self.eta[px,py]) / 4. *
                        (self.dx**2))

            self.deposit(Vp_dep, px, py)


        elif (U_loc > self.U_ero_sand) and (qs_loc < qs_cap):
            #erosion can only occur if haven't reached transport capacity

            Vp_ero = (self.Vp_sed *
                      (U_loc**self.beta - self.U_ero_sand**self.beta) /
                      self.U_ero_sand**self.beta)

            Vp_ero = min(Vp_ero,
                        (self.stage[px,py] - self.eta[px,py]) / 4. *
                        (self.dx**2))

            self.erode(Vp_ero,px,py)


        self.Vp_dep_sand[px,py] = self.Vp_dep_sand[px,py] + Vp_dep





    def mud_dep_ero(self,px,py):
        '''decide if deposit or erode mud'''

        U_loc = self.uw[px,py]

        Vp_dep = 0
        Vp_ero = 0

        if U_loc < self.U_dep_mud:

            Vp_dep = (self._lambda * self.Vp_res *
                     (self.U_dep_mud**self.beta - U_loc**self.beta) /
                     (self.U_dep_mud**self.beta))

            Vp_dep = min(Vp_dep,
                        (self.stage[px,py] - self.eta[px,py]) / 4. *
                        (self.dx**2))
            #change limited to 1/4 local depth

            self.deposit(Vp_dep,px,py)



        if U_loc > self.U_ero_mud:

            Vp_ero = (self.Vp_sed *
                     (U_loc**self.beta - self.U_ero_mud**self.beta) /
                     self.U_ero_mud**self.beta)

            Vp_ero = min(Vp_ero,
                        (self.stage[px,py] - self.eta[px,py]) / 4. *
                        (self.dx**2))
            #change limited to 1/4 local depth

            self.erode(Vp_ero,px,py)


        self.Vp_dep_mud[px,py] = self.Vp_dep_mud[px,py] + Vp_dep





    def sed_parcel(self,theta_sed,sed,px,py):
        '''route one sediment parcel'''

        it = 0
        sed_continue = 1


        while (sed_continue == 1) and (it < self.itmax):
            #choose next with weights

            it += 1
            ind = (px,py)

            depth_ind = self.pad_depth[ind[0]-1+1:ind[0]+2+1, ind[1]-1+1:ind[1]+2+1]
            cell_type_ind = self.pad_cell_type[ind[0]-1+1:ind[0]+2+1, ind[1]-1+1:ind[1]+2+1]

            w1 = np.maximum(0, (self.qx[ind] * self.jvec +
                           self.qy[ind] * self.ivec))
            w2 = depth_ind ** theta_sed
            weight = (w1 * w2 / self.distances)

            weight[depth_ind <= self.dry_depth] = 0.0001
            weight[cell_type_ind == -2] = np.nan


            if ind[0] == 0:
                weight[0,:] = np.nan


            new_cell = self.random_pick(weight)

            jstep = self.iwalk.flat[new_cell]
            istep = self.jwalk.flat[new_cell]
            dist = np.sqrt(istep**2 + jstep**2)


            ########################################################
            # deposition and erosion
            ########################################################
            if sed == 'sand': #sand

                if dist > 0:
                    # deposition in current cell
                    self.qs[px,py] = (self.qs[px,py] +
                                      self.Vp_res / 2 / self.dt / self.dx)

                px = px + istep
                py = py + jstep

                if dist > 0:
                    # deposition in downstream cell
                    self.qs[px,py] = (self.qs[px,py] +
                                      self.Vp_res / 2 / self.dt / self.dx)

                self.sand_dep_ero(px,py)


            if sed == 'mud': #mud

                px = px + istep
                py = py + jstep

                self.mud_dep_ero(px,py)


            if self.cell_type[px,py] == -1:
                sed_continue = 0






    def sand_route(self):
        '''route sand parcels; topo diffusion'''

        theta_sed = self.theta_sand


        num_starts = int(self.Np_sed * self.f_bedload)
        start_indices = [self.random_pick_inlet(self.inlet) for x in range(num_starts)]

        for np_sed in range(num_starts):

            self.Vp_res = self.Vp_sed

            px = 0
            py = start_indices[np_sed]

            self.qs[px,py] = (self.qs[px,py] +
                              self.Vp_res / 2. / self.dt / self.dx)

            self.sed_parcel(theta_sed, 'sand', px, py)

#         self.topo_diffusion()






    def topo_diffusion(self):
        '''
        Diffuse topography after routing all coarse sediment parcels
        '''


        for crossdiff in range(self.N_crossdiff):

            a = ndimage.convolve(self.eta, self.kernel1, mode='constant')
            b = ndimage.convolve(self.qs, self.kernel2, mode='constant')
            c = ndimage.convolve(self.qs * self.eta, self.kernel2,
                                 mode='constant')

            self.cf = (self.diffusion_multiplier *
                     (self.qs * a - self.eta * b + c))


            self.cf[self.cell_type == -2] = 0
            self.cf[0,:] = 0

            self.eta += self.cf








    def mud_route(self):
        '''route mud parcels'''

        theta_sed = self.theta_mud

        num_starts = int(self.Np_sed * (1-self.f_bedload))
        start_indices = [self.random_pick_inlet(self.inlet) for x in range(num_starts)]

        for np_sed in range(num_starts):

            self.Vp_res = self.Vp_sed

            px = 0
            py = start_indices[np_sed]

            self.sed_parcel(theta_sed, 'mud', px, py)







    def free_surf(self, it):
        '''calculate free surface after routing one water parcel'''


        Hnew = np.zeros((self.L,self.W))

        for n,i in enumerate(self.indices):

            inds = np.unravel_index(i[i > 0], self.depth.shape)
            xs, ys = inds

            Hnew[:] = 0

            if ((self.cell_type[xs[-1],ys[-1]] == -1) and
                (self.looped[n] == 0)):

                self.count += 1
                Hnew[xs[-1],ys[-1]] = self.H_SL
                #if cell is in ocean, H = H_SL (downstream boundary condition)

                it0 = 0

                for it in range(len(xs)-2, -1, -1):
                    #counting back from last cell visited

                    i = int(xs[it])
                    ip = int(xs[it+1])
                    j = int(ys[it])
                    jp = int(ys[it+1])
                    dist = np.sqrt((ip-i)**2 + (jp-j)**2)

                    if dist > 0:

                        if it0 == 0:

                            if ((self.uw[i,j] > self.u0 * 0.5) or
                                (self.depth[i,j] < 0.1*self.h0)):
                                #see if it is shoreline

                                it0 = it

                            dH = 0

                        else:

                            if self.uw[i,j] == 0:

                                dH = 0
                                # if no velocity
                                # no change in water surface elevation

                            else:

                                dH = (self.S0 *
                                     (self.ux[i,j] * (ip - i) * self.dx +
                                      self.uy[i,j] * (jp - j) * self.dx) /
                                      self.uw[i,j])
                                      # difference between streamline and
                                      # parcel path

                    Hnew[i,j] = Hnew[ip,jp] + dH
                    #previous cell's surface plus difference in H

                    self.sfc_visit[i,j] = self.sfc_visit[i,j] + 1
                    #add up # of cell visits

                    self.sfc_sum[i,j] = self.sfc_sum[i,j] + Hnew[i,j]
                    # sum of all water surface elevations




    def update_water(self,timestep,itr):
        '''update surface after routing all parcels
        could divide into 3 functions for cleanliness'''

        #####################################################################
        # update free surface
        #####################################################################

        Hnew = self.eta + self.depth
        Hnew[Hnew < self.H_SL] = self.H_SL
        #water surface height not under sea level


        Hnew[self.sfc_visit > 0] = (self.sfc_sum[self.sfc_visit > 0] /
                                    self.sfc_visit[self.sfc_visit > 0])
        #find average water surface elevation for a cell

        Hnew_pad = np.pad(Hnew, 1, 'edge')

        #smooth newly calculated free surface
        Htemp = Hnew

        for itsmooth in range(self.Nsmooth):

            Hsmth = Htemp

            for i in range(self.L):

                for j in range(self.W):

                    if self.cell_type[i,j] > -2:
                        #locate non-boundary cells


                        sumH = 0
                        nbcount = 0

                        ct_ind = self.pad_cell_type[i-1+1:i+2+1, j-1+1:j+2+1]
                        Hnew_ind = Hnew_pad[i-1+1:i+2+1, j-1+1:j+2+1]

                        Hnew_ind[1,1] = 0
                        Hnew_ind[ct_ind == -2] = 0

                        sumH = np.sum(Hnew_ind)
                        nbcount = np.sum(Hnew_ind >0)


                        if nbcount > 0:

                            Htemp[i,j] = (self.Csmooth * Hsmth[i,j] +
                                          (1 - self.Csmooth) * sumH / nbcount)
                            #smooth if are not wall cells

        Hsmth = Htemp

        if timestep > 0:
            self.stage = ((1 - self.omega_sfc) * self.stage +
                          self.omega_sfc * Hsmth)

        self.flooding_correction()




    #############################################
    ################# water flow ################
    #############################################

    def init_water_iteration(self):


        self.qxn[:] = 0; self.qyn[:] = 0; self.qwn[:] = 0

        self.free_surf_flag[:] = 0
        self.indices[:] = 0
        self.sfc_visit[:] = 0
        self.sfc_sum[:] = 0


        self.pad_stage = np.pad(self.stage, 1, 'edge')

        self.pad_depth = np.pad(self.depth, 1, 'edge')

        self.pad_cell_type = np.pad(self.cell_type, 1, 'edge')




    def run_water_iteration(self):

        iter = 0
        start_indices = [self.random_pick_inlet(self.inlet) for x in range(self.Np_water)]

        self.qxn.flat[start_indices] += 1
        self.qwn.flat[start_indices] += self.Qp_water / self.dx / 2.

        self.indices[:,0] = start_indices
        current_inds = list(start_indices)

        self.looped[:] = 0

        while (sum(current_inds) > 0) & (iter < self.itmax):


            iter += 1

            self.check_size_of_indices_matrix(iter)

            inds = np.unravel_index(current_inds, self.depth.shape)
            inds_tuple = [(inds[0][i], inds[1][i]) for i in range(len(inds[0]))]


            new_cells = [self.get_weight(x)
                            if x != (0,0) else 4 for x in inds_tuple]


            new_inds = list(map(lambda x,y: self.calculate_new_ind(x,y)
                            if y != 4 else 0, inds_tuple, new_cells))


            dist = list(map(lambda x,y,z: self.step_update(x,y,z) if x > 0
                       else 0, current_inds, new_inds, new_cells))



            new_inds = np.array(new_inds, dtype = np.int)
            new_inds[np.array(dist) == 0] = 0


            self.indices[:,iter] = new_inds

            current_inds = self.check_for_loops(new_inds, iter)


            current_inds = self.check_for_boundary(current_inds)


            self.indices[:,iter] = current_inds



            current_inds[self.free_surf_flag > 0] = 0




    def check_for_boundary(self, inds):

        self.free_surf_flag[(self.cell_type.flat[inds] == -1) & (self.free_surf_flag == 0)] = 1

        self.free_surf_flag[(self.cell_type.flat[inds] == -1) & (self.free_surf_flag == -1)] = 2

        inds[self.free_surf_flag == 2] = 0

        return inds



    def check_for_loops(self, inds, it):

        looped = [len(i[i>0]) != len(set(i[i>0])) for i in self.indices]

        for n in range(self.Np_water):

            ind = inds[n]
            # revised 'it' to 'np.max(it)' to match python 2 > assessment
            if (looped[n]) and (ind > 0) and (np.max(it) > self.L0):

                self.looped[n] += 1

                it = np.unravel_index(ind, self.depth.shape)

                px, py = it

                Fx = px - 1
                Fy = py - self.CTR

                Fw = np.sqrt(Fx**2 + Fy**2)

                if Fw != 0:
                    px = px + np.round(Fx / Fw * 5.)
                    py = py + np.round(Fy / Fw * 5.)

                px = max(px, self.L0)
                px = int(min(self.L - 2, px))

                py = max(1, py)
                py = int(min(self.W - 2, py))

                nind = np.ravel_multi_index((px,py), self.depth.shape)

                inds[n] = nind

                self.free_surf_flag[n] = -1


        return inds




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





    def finalize_water_iteration(self, timestep, iteration):
        '''
        Finish updating flow fields
        Clean up at end of water iteration
        '''

        self.update_water(timestep, iteration)

        self.stage[:] = np.maximum(self.stage, self.H_SL)
        self.depth[:] = np.maximum(self.stage - self.eta, 0)

        self.update_flow_field(iteration)
        self.update_velocity_field()



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





    def flooding_correction(self):
        '''
        Flood dry cells along the shore if necessary

        Check the neighbors of all dry cells. If any dry cells have wet
        neighbors, check that their stage is not higher than the bed elevation
        of the center cell.
        If it is, flood the dry cell.
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

            # pretends dry neighbor cells have stage zero
            # so they cannot be > eta_shore[i]

            stage_nh = wet_mask_nh[:,shore_ind[0][i],shore_ind[1][i]] * \
                stage_nhs[:,shore_ind[0][i],shore_ind[1][i]]

            if (stage_nh > eta_shore[i]).any():
                self.stage[shore_ind[0][i],shore_ind[1][i]] = max(stage_nh)



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
            self.qx = self.qxn.copy()
            self.qy = self.qyn.copy()

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


    def set_defaults(self):

        for k,v in list(self._var_default_map.items()):
            setattr(self, self._var_name_map[k], v)



    def create_dicts(self):

        self._input_var_names = list(self._input_vars.keys())

        self._var_type_map = dict()
        self._var_name_map = dict()
        self._var_default_map = dict()

        for k in list(self._input_vars.keys()):
            self._var_type_map[k] = self._input_vars[k]['type']
            self._var_name_map[k] = self._input_vars[k]['name']
            self._var_default_map[k] = self._input_vars[k]['default']



    def set_constants(self):

        self.g = 9.81   # (gravitation const.)

        sqrt2 = np.sqrt(2)
        self.distances = np.array([[sqrt2, 1, sqrt2],
                                   [1, 1, 1],
                                   [sqrt2, 1, sqrt2]])

        sqrt05 = np.sqrt(0.5)
        self.ivec = np.array([[-sqrt05, 0, sqrt05],
                              [-1, 0, 1],
                              [-sqrt05, 0, sqrt05]])

        self.iwalk = np.array([[-1, 0, 1],
                               [-1, 0, 1],
                               [-1, 0, 1]])

        self.jvec = np.array([[-sqrt05, -1, -sqrt05],
                              [0, 0, 0],
                              [sqrt05, 1, sqrt05]])

        self.jwalk = np.array([[-1, -1, -1],
                               [0, 0, 0],
                               [1, 1, 1]])


        self.dxn_iwalk = [1,1,0,-1,-1,-1,0,1]
        self.dxn_jwalk = [0,1,1,1,0,-1,-1,-1]
        self.dxn_dist = \
        [sqrt(self.dxn_iwalk[i]**2 + self.dxn_jwalk[i]**2) for i in range(8)]



        self.walk_flat = np.array([1, -self.W+1, -self.W, -self.W-1,
                                    -1, self.W-1, self.W, self.W+1])


        self.kernel1 = np.array([[1, 1, 1],
                                 [1, -8, 1],
                                 [1, 1, 1]])

        self.kernel2 = np.array([[1, 1, 1],
                                 [1, 0, 1],
                                 [1, 1, 1]])





    def create_other_variables(self):

        self.init_Np_water = self.Np_water
        self.init_Np_sed = self.Np_sed

        self.dx = float(self.dx)

        self.theta_sand = self.coeff_theta_sand * self.theta_water
        self.theta_mud = self.coeff_theta_mud * self.theta_water
        self.Nsmooth = 1

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
        self.CTR = floor(self.W / 2.) - 1

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

        self._lambda = 1.                       # sedimentation lag


        self.diffusion_multiplier = (self.dt/self.N_crossdiff * self.alpha *
                                    0.5 / self.dx**2)


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

        self.cell_type = np.zeros((self.L,self.W), dtype=np.int)

        self.eta = np.zeros((self.L,self.W)).astype(np.float32)
        self.stage = np.zeros((self.L,self.W)).astype(np.float32)
        self.depth = np.zeros((self.L,self.W)).astype(np.float32)

        self.qx = np.zeros((self.L,self.W))
        self.qy = np.zeros((self.L,self.W))
        self.qxn = np.zeros((self.L,self.W))
        self.qyn = np.zeros((self.L,self.W))
        self.qwn = np.zeros((self.L,self.W))
        self.ux = np.zeros((self.L,self.W))
        self.uy = np.zeros((self.L,self.W))
        self.uw = np.zeros((self.L,self.W))


        self.qs = np.zeros((self.L,self.W))
        self.Vp_dep_sand = np.zeros((self.L,self.W))
        self.Vp_dep_mud = np.zeros((self.L,self.W))

        self.free_surf_flag = np.zeros((self.Np_water,), dtype=np.int)
        self.looped = np.zeros((self.Np_water,))
        self.indices = np.zeros((self.Np_water, self.size_indices),
                                 dtype = np.int)

        self.sfc_visit = np.zeros_like(self.depth)
        self.sfc_sum = np.zeros_like(self.depth)

        ##### domain #####
        cell_land = 2
        cell_channel = 1
        cell_ocean = 0
        cell_edge = -1

        self.cell_type[:self.L0,:] = cell_land

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

        self.ux[self.depth>0] = self.qx[self.depth>0] / self.depth[self.depth>0]
        self.uy[self.depth>0] = self.qy[self.depth>0] / self.depth[self.depth>0]
        self.uw[self.depth>0] = self.qw[self.depth>0] / self.depth[self.depth>0]

        # reset the land cell_type to -2
        self.cell_type[self.cell_type == cell_land] = -2
        self.cell_type[-1,:] = cell_edge
        self.cell_type[:,0] = cell_edge
        self.cell_type[:,-1] = cell_edge

        bounds = [(np.sqrt((i-3)**2 + (j-self.CTR)**2))
            for i in range(self.L)
            for j in range(self.W)]
        bounds =  np.reshape(bounds,(self.L, self.W))

        self.cell_type[bounds >= min(self.L - 5, self.W/2 - 5)] = cell_edge

        self.cell_type[:self.L0,:] = -cell_land
        self.cell_type[:self.L0, channel_inds:y_channel_max] = cell_channel




        self.inlet = list(np.unique(np.where(self.cell_type == 1)[1]))
        self.eta[:] = self.stage - self.depth

        self.clim_eta = (-self.h0 - 1, 0.05)




    def init_stratigraphy(self):
        '''
        Creates sparse array to store stratigraphy data
        '''


        if self.save_strata:

            self.strata_counter = 0

            self.n_steps = 5 * self.save_dt

            self.strata_sand_frac = lil_matrix((self.L * self.W, self.n_steps),
                                                dtype=np.float32)

            self.init_eta = self.eta.copy()
            self.strata_eta = lil_matrix((self.L * self.W, self.n_steps),
                                          dtype=np.float32)



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






    def init_output_grids(self):
        '''
        Creates a netCDF file to store output grids
        Fills with default variables

        Overwrites an existing netcdf file with the same name
        '''

        if (self.save_eta_grids or
            self.save_depth_grids or
            self.save_stage_grids or
            self.save_discharge_grids or
            self.save_velocity_grids or
            self.save_strata):

            if self.verbose:
                self.logger.info('Generating netCDF file for output grids...')

            directory = self.prefix
            filename = 'pyDeltaRCM_output.nc'

            if not os.path.exists(directory):
                if self.verbose: self.logger.info('Creating output directory')
                os.makedirs(directory)

            file_path = os.path.join(directory, filename)

            if os.path.exists(file_path):
                if self.verbose:
                    self.logger.info('*** Replaced existing netCDF file ***')
                os.remove(file_path)

            self.output_netcdf = Dataset(file_path, 'w',
                                         format='NETCDF4_CLASSIC')

            self.output_netcdf.description = 'Output grids from pyDeltaRCM'
            self.output_netcdf.history = ('Created ' +
                                          time_lib.ctime(time_lib.time()))
            self.output_netcdf.source = 'pyDeltaRCM / CSDMS'

            length = self.output_netcdf.createDimension('length', self.L)
            width = self.output_netcdf.createDimension('width', self.W)
            total_time = self.output_netcdf.createDimension('total_time', None)



            x = self.output_netcdf.createVariable('x', 'f4', ('length','width'))
            y = self.output_netcdf.createVariable('y', 'f4', ('length','width'))
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
                                            ('total_time','length','width'))
                eta.units = 'meters'


            if self.save_stage_grids:
                stage = self.output_netcdf.createVariable('stage',
                                             'f4',
                                            ('total_time','length','width'))
                stage.units = 'meters'


            if self.save_depth_grids:
                depth = self.output_netcdf.createVariable('depth',
                                             'f4',
                                            ('total_time','length','width'))
                depth.units = 'meters'


            if self.save_discharge_grids:
                discharge = self.output_netcdf.createVariable('discharge',
                                             'f4',
                                            ('total_time','length','width'))
                discharge.units = 'cubic meters per second'


            if self.save_velocity_grids:
                velocity = self.output_netcdf.createVariable('velocity',
                                             'f4',
                                            ('total_time','length','width'))
                velocity.units = 'meters per second'



            if self.verbose: self.logger.info('Output netCDF file created.')




    def init_subsidence(self):
        '''
        Initializes patterns of subsidence if
        toggle_subsidence is True (default False)

        Modify the equations for self.subsidence_mask and self.sigma as desired
        '''

        if self.toggle_subsidence:

            R1 = 0.3 * self.L; R2 = 1. * self.L # radial limits (fractions of L)
            theta1 = -pi/3; theta2 =  pi/3.   # angular limits

            Rloc = np.sqrt((self.y - self.L0)**2 + (self.x - self.W / 2.)**2)

            thetaloc = np.zeros((self.L, self.W))
            thetaloc[self.y > self.L0 - 1] = np.arctan(
                            (self.x[self.y > self.L0 - 1] - self.W / 2.) /
                            (self.y[self.y > self.L0 - 1] - self.L0 + 1))

            self.subsidence_mask = ((R1 <= Rloc) & (Rloc <= R2) &
                                    (theta1 <= thetaloc) & (thetaloc <= theta2))

            self.subsidence_mask[:self.L0,:] = False

            self.sigma = self.subsidence_mask * self.sigma_max * self.dt







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
            self.output_netcdf.variables['strata_age'][:] = list(range(shape[1]-1,
                                                                  -1, -1))


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





    def init_logger(self):

        if self.verbose:

            self.logger = logging.getLogger("driver")
            self.logger.setLevel(logging.INFO)

            # create the logging file handler
            st = timestr = time.strftime("%Y%m%d-%H%M%S")
            fh = logging.FileHandler("pyDeltaRCM_" + st + ".log")

            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            fh.setFormatter(formatter)

            # add handler to logger object
            self.logger.addHandler(fh)
