
import sys
import os
import re
import string
import logging
import time

from math import floor, sqrt, pi
import numpy as np
from random import shuffle

import matplotlib
from matplotlib import pyplot as plt

from scipy.sparse import lil_matrix, csc_matrix, hstack
from scipy import ndimage

from netCDF4 import Dataset

from .shared_tools import shared_tools

# tools for water routing algorithms


class water_tools(shared_tools):

    def init_water_iteration(self):

        self.qxn[:] = 0
        self.qyn[:] = 0
        self.qwn[:] = 0

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
        self.qwn.flat[start_indices] += self.Qp_water / self.dx / 2

        self.indices[:, 0] = start_indices
        current_inds = list(start_indices)

        self.looped[:] = 0

        self.get_water_weight_array()

        while (sum(current_inds) > 0) & (iter < self.itmax):

            iter += 1

            self.check_size_of_indices_matrix(iter)

            inds = np.unravel_index(current_inds, self.depth.shape)
            inds_tuple = list(zip(*inds))

            new_cells = [self.get_new_cell(x)
                         if x != (0, 0) else 4
                         for x in inds_tuple]

            new_inds = self.calculate_new_indices(inds_tuple, new_cells)

            dist = self.update_steps(current_inds, new_inds, new_cells)

            new_inds = np.array(new_inds, dtype=np.int)
            new_inds[np.array(dist) == 0] = 0

            self.indices[:, iter] = new_inds

            current_inds = self.check_for_loops(new_inds, iter)

            current_inds = self.check_for_boundary(current_inds)

            self.indices[:, iter] = current_inds

            current_inds[self.free_surf_flag > 0] = 0

    def free_surf(self, it):
        """calculate free surface after routing one water parcel"""

        Hnew = np.zeros((self.L, self.W))

        for n, i in enumerate(self.indices):

            inds = np.unravel_index(i[i > 0], self.depth.shape)
            xs, ys = inds

            Hnew[:] = 0

            if ((self.cell_type[xs[-1], ys[-1]] == -1) and
                    (self.looped[n] == 0)):

                Hnew[xs[-1], ys[-1]] = self.H_SL
                # if cell is in ocean, H = H_SL (downstream boundary condition)

                it0 = 0

                for it in range(len(xs) - 2, -1, -1):
                    # counting back from last cell visited

                    i = int(xs[it])
                    ip = int(xs[it + 1])
                    j = int(ys[it])
                    jp = int(ys[it + 1])
                    dist = np.sqrt((ip - i)**2 + (jp - j)**2)

                    if dist > 0:

                        if it0 == 0:

                            if ((self.uw[i, j] > self.u0 * 0.5) or
                                    (self.depth[i, j] < 0.1 * self.h0)):
                                # see if it is shoreline

                                it0 = it

                            dH = 0

                        else:

                            if self.uw[i, j] == 0:

                                dH = 0
                                # if no velocity
                                # no change in water surface elevation

                            else:

                                dH = (self.S0 *
                                      (self.ux[i, j] * (ip - i) * self.dx +
                                       self.uy[i, j] * (jp - j) * self.dx) /
                                      self.uw[i, j])
                                # difference between streamline and
                                # parcel path

                    Hnew[i, j] = Hnew[ip, jp] + dH
                    # previous cell's surface plus difference in H

                    self.sfc_visit[i, j] = self.sfc_visit[i, j] + 1
                    # add up # of cell visits

                    # sum of all water surface elevations
                    self.sfc_sum[i, j] = self.sfc_sum[i, j] + Hnew[i, j]

    def finalize_water_iteration(self, timestep, iteration):
        """
        Finish updating flow fields
        Clean up at end of water iteration
        """

        self.update_water(timestep, iteration)

        self.stage[:] = np.maximum(self.stage, self.H_SL)
        self.depth[:] = np.maximum(self.stage - self.eta, 0)

        self.update_flow_field(iteration)
        self.update_velocity_field()

    def check_size_of_indices_matrix(self, it):
        if it >= self.indices.shape[1]:
            """
            Initial size of self.indices is half of self.itmax
            because the number of iterations doesn't go beyond
            that for many timesteps.

            Once it reaches it > self.itmax/2 once, make the size
            self.iter for all further timesteps
            """

            if self.verbose >= 2:
                self.logger.info('Increasing size of self.indices')

            indices_blank = np.zeros(
                (np.int(self.Np_water), np.int(self.itmax / 4)), dtype=np.int)

            self.indices = np.hstack((self.indices, indices_blank))

    def get_water_weight_array(self):

        self.water_weights = np.zeros(shape = (self.L, self.W, 9))

        for i in range(self.L):
            for j in range(self.W):
                self.water_weights[i, j] = self.get_weight_at_cell((i, j))

    def get_weight_at_cell(self, ind):

        stage_ind = self.pad_stage[
            ind[0] - 1 + 1:ind[0] + 2 + 1, ind[1] - 1 + 1:ind[1] + 2 + 1]

        weight_sfc = np.maximum(0,
                                (self.stage[ind] - stage_ind) / self.distances)

        weight_int = np.maximum(0, (self.qx[ind] * self.jvec +
                                    self.qy[ind] * self.ivec) / self.distances)

        if ind[0] == 0:
            weight_sfc[0, :] = 0
            weight_int[0, :] = 0

        depth_ind = self.pad_depth[
            ind[0] - 1 + 1:ind[0] + 2 + 1, ind[1] - 1 + 1:ind[1] + 2 + 1]
        ct_ind = self.pad_cell_type[
            ind[0] - 1 + 1:ind[0] + 2 + 1, ind[1] - 1 + 1:ind[1] + 2 + 1]

        weight_sfc[(depth_ind <= self.dry_depth) | (ct_ind == -2)] = 0
        weight_int[(depth_ind <= self.dry_depth) | (ct_ind == -2)] = 0

        if np.nansum(weight_sfc) > 0:
            weight_sfc = weight_sfc / np.nansum(weight_sfc)

        if np.nansum(weight_int) > 0:
            weight_int = weight_int / np.nansum(weight_int)

        weight = self.gamma * weight_sfc + (1 - self.gamma) * weight_int
        weight = depth_ind ** self.theta_water * weight
        weight[depth_ind <= self.dry_depth] = 0
        if np.sum(weight) != 0:
            weight = weight / np.sum(weight)
        else:
            weight = weight

        return np.cumsum(weight.flatten())

    def get_new_cell(self, ind):
        weight = self.water_weights[ind[0], ind[1]]
        new_cell = self.random_pick(weight)
        return new_cell

    def calculate_new_ind(self, ind, new_cell):

        new_ind = (ind[0] + self.jwalk.flat[new_cell],
                   ind[1] + self.iwalk.flat[new_cell])
        # added wrap mode to fct to resolve ValueError due to negative numbers
        new_ind_flat = np.ravel_multi_index(
            new_ind, self.depth.shape, mode='wrap')

        return new_ind_flat

    def calculate_new_indices(self, inds_tuple, new_cells):
        newbies = []
        for i, j in zip(inds_tuple, new_cells):
            if j != 4:
                newbies.append(self.calculate_new_ind(i, j))
            else:
                newbies.append(0)
        return newbies

    def step_update(self, ind, new_ind, new_cell):

        istep = self.iwalk.flat[new_cell]
        jstep = self.jwalk.flat[new_cell]
        dist = np.sqrt(istep * istep + jstep * jstep)

        if dist > 0:

            self.qxn.flat[ind] += jstep / dist
            self.qyn.flat[ind] += istep / dist
            self.qwn.flat[ind] += self.Qp_water / self.dx / 2

            self.qxn.flat[new_ind] += jstep / dist
            self.qyn.flat[new_ind] += istep / dist
            self.qwn.flat[new_ind] += self.Qp_water / self.dx / 2

        return dist

    def update_steps(self, current_inds, new_inds, new_cells):
        newbies = []
        for i, j, k in zip(current_inds, new_inds, new_cells):
            if i > 0:
                newbies.append(self.step_update(i, j, k))
            else:
                newbies.append(0)
        return newbies

    def check_for_loops(self, inds, it):

        looped = [len(i[i > 0]) != len(set(i[i > 0])) for i in self.indices]

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

                nind = np.ravel_multi_index((px, py), self.depth.shape)

                inds[n] = nind

                self.free_surf_flag[n] = -1

        return inds

    def check_for_boundary(self, inds):

        self.free_surf_flag[(self.cell_type.flat[inds] == -1)
                            & (self.free_surf_flag == 0)] = 1

        self.free_surf_flag[(self.cell_type.flat[inds] == -1)
                            & (self.free_surf_flag == -1)] = 2

        inds[self.free_surf_flag == 2] = 0

        return inds

    def update_water(self, timestep, itr):
        """Update surface after routing all parcels.

        Could divide into 3 functions for cleanliness.
        """

        Hnew = self.eta + self.depth
        Hnew[Hnew < self.H_SL] = self.H_SL # water surface height not under sea level

        Hnew[self.sfc_visit > 0] = (self.sfc_sum[self.sfc_visit > 0] /
                                    self.sfc_visit[self.sfc_visit > 0])
        # find average water surface elevation for a cell

        Hnew_pad = np.pad(Hnew, 1, 'edge')

        # smooth newly calculated free surface
        Htemp = Hnew

        for itsmooth in range(self.Nsmooth):

            Hsmth = Htemp

            for i in range(self.L):

                for j in range(self.W):

                    if self.cell_type[i, j] > -2:
                        # locate non-boundary cells
                        sumH = 0
                        nbcount = 0

                        ct_ind = self.pad_cell_type[
                            i - 1 + 1:i + 2 + 1, j - 1 + 1:j + 2 + 1]
                        Hnew_ind = Hnew_pad[
                            i - 1 + 1:i + 2 + 1, j - 1 + 1:j + 2 + 1]

                        Hnew_ind[1, 1] = 0
                        Hnew_ind[ct_ind == -2] = 0

                        sumH = np.sum(Hnew_ind)
                        nbcount = np.sum(Hnew_ind > 0)

                        if nbcount > 0:

                            Htemp[i, j] = (self.Csmooth * Hsmth[i, j] +
                                           (1 - self.Csmooth) * sumH / nbcount)
                            # smooth if are not wall cells

        Hsmth = Htemp

        if timestep > 0:
            self.stage = ((1 - self.omega_sfc) * self.stage +
                          self.omega_sfc * Hsmth)

        self.flooding_correction()

    def flooding_correction(self):
        """
        Flood dry cells along the shore if necessary

        Check the neighbors of all dry cells. If any dry cells have wet
        neighbors, check that their stage is not higher than the bed elevation
        of the center cell.
        If it is, flood the dry cell.
        """

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

            stage_nh = wet_mask_nh[:, shore_ind[0][i], shore_ind[1][i]] * \
                stage_nhs[:, shore_ind[0][i], shore_ind[1][i]]

            if (stage_nh > eta_shore[i]).any():
                self.stage[shore_ind[0][i], shore_ind[1][i]] = max(stage_nh)

    def get_wet_mask_nh(self):
        """
        Returns np.array((8,L,W)), for each neighbor around a cell
        with 1 if the neighbor is wet and 0 if dry
        """

        wet_mask = (self.depth > self.dry_depth) * 1
        wet_mask_nh = self.build_weight_array(wet_mask, fix_edges=True)

        return wet_mask_nh

    def build_weight_array(self, array, fix_edges=False, normalize=False):
        """
        Create np.array((8,L,W)) of quantity a
        in each of the neighbors to a cell
        """

        a_shape = array.shape

        wgt_array = np.zeros((8, a_shape[0], a_shape[1]))
        nums = list(range(8))

        wgt_array[nums[0], :, :-1] = array[:, 1:]  # E
        wgt_array[nums[1], 1:, :-1] = array[:-1, 1:]  # NE
        wgt_array[nums[2], 1:, :] = array[:-1, :]  # N
        wgt_array[nums[3], 1:, 1:] = array[:-1, :-1]  # NW
        wgt_array[nums[4], :, 1:] = array[:, :-1]  # W
        wgt_array[nums[5], :-1, 1:] = array[1:, :-1]  # SW
        wgt_array[nums[6], :-1, :] = array[1:, :]  # S
        wgt_array[nums[7], :-1, :-1] = array[1:, 1:]  # SE

        if fix_edges:
            wgt_array[nums[0], :, -1] = wgt_array[nums[0], :, -2]
            wgt_array[nums[1], :, -1] = wgt_array[nums[1], :, -2]
            wgt_array[nums[7], :, -1] = wgt_array[nums[7], :, -2]
            wgt_array[nums[1], 0, :] = wgt_array[nums[1], 1, :]
            wgt_array[nums[2], 0, :] = wgt_array[nums[2], 1, :]
            wgt_array[nums[3], 0, :] = wgt_array[nums[3], 1, :]
            wgt_array[nums[3], :, 0] = wgt_array[nums[3], :, 1]
            wgt_array[nums[4], :, 0] = wgt_array[nums[4], :, 1]
            wgt_array[nums[5], :, 0] = wgt_array[nums[5], :, 1]
            wgt_array[nums[5], -1, :] = wgt_array[nums[5], -2, :]
            wgt_array[nums[6], -1, :] = wgt_array[nums[6], -2, :]
            wgt_array[nums[7], -1, :] = wgt_array[nums[7], -2, :]

        if normalize:
            a_sum = np.sum(wgt_array, axis=0)
            wgt_array[:, a_sum != 0] = wgt_array[
                :, a_sum != 0] / a_sum[a_sum != 0]

        return wgt_array

    def update_flow_field(self, iteration):
        """
        Update water discharge after one water iteration
        """

        timestep = self._time

        dloc = (self.qxn**2 + self.qyn**2)**(0.5)

        qwn_div = np.ones((self.L, self.W))
        qwn_div[dloc > 0] = self.qwn[dloc > 0] / dloc[dloc > 0]

        self.qxn *= qwn_div
        self.qyn *= qwn_div

        if timestep > 0:

            omega = self.omega_flow_iter

            if iteration == 0:
                omega = self.omega_flow

            self.qx = self.qxn * omega + self.qx * (1 - omega)
            self.qy = self.qyn * omega + self.qy * (1 - omega)

        else:
            self.qx = self.qxn.copy()
            self.qy = self.qyn.copy()

        self.qw = (self.qx**2 + self.qy**2)**(0.5)

        self.qx[0, self.inlet] = self.qw0
        self.qy[0, self.inlet] = 0
        self.qw[0, self.inlet] = self.qw0

    def update_velocity_field(self):
        """
        Update the flow velocity field after one water iteration
        """

        mask = (self.depth > self.dry_depth) * (self.qw > 0)

        self.uw[mask] = np.minimum(
            self.u_max, self.qw[mask] / self.depth[mask])
        self.uw[~mask] = 0
        self.ux[mask] = self.uw[mask] * self.qx[mask] / self.qw[mask]
        self.ux[~mask] = 0
        self.uy[mask] = self.uw[mask] * self.qy[mask] / self.qw[mask]
        self.uy[~mask] = 0
