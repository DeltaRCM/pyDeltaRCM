
import numpy as np

from . import shared_tools

# tools for water routing algorithms


class water_tools(object):

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

        _iter = 0
        inlet_weights = np.ones_like(self.inlet)
        start_indices = [
            self.inlet[shared_tools.random_pick(inlet_weights / sum(inlet_weights))]
            for x in range(self.Np_water)]

        self.qxn.flat[start_indices] += 1
        self.qwn.flat[start_indices] += self.Qp_water / self.dx / 2

        self.indices[:, 0] = start_indices
        current_inds = np.array(start_indices)

        self.looped[:] = 0

        self.get_water_weight_array()

        while (sum(current_inds) > 0) & (_iter < self.itmax):

            _iter += 1

            self.check_size_of_indices_matrix(_iter)

            inds = np.unravel_index(current_inds, self.depth.shape)
            inds_tuple = list(zip(*inds))

            new_cells = [self.get_new_cell(x)
                         if x != (0, 0) else 4
                         for x in inds_tuple]

            new_cells = np.array(new_cells)

            next_index = shared_tools.calculate_new_ind(
                current_inds,
                new_cells,
                self.iwalk.flatten(),
                self.jwalk.flatten(),
                self.eta.shape)

            dist, istep, jstep, astep = shared_tools.get_steps(
                new_cells,
                self.iwalk.flat[:],
                self.jwalk.flat[:])

            self.update_Q(dist, current_inds, next_index, astep, jstep, istep)

            current_inds, self.looped, self.free_surf_flag = shared_tools.check_for_loops(
                self.indices, next_index, _iter, self.L0, self.looped, self.eta.shape,
                self.CTR, self.free_surf_flag)

            current_inds = self.check_for_boundary(current_inds)

            self.indices[:, _iter] = current_inds

            current_inds[self.free_surf_flag > 0] = 0

    def free_surf(self, it):
        """Calculate free surface after routing one water parcel."""
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
        """Finish updating flow fields.

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
            _msg = 'Increasing size of self.indices'
            self.logger.info(_msg)
            if self.verbose >= 2:
                print(_msg)

            indices_blank = np.zeros(
                (np.int(self.Np_water), np.int(self.itmax / 4)), dtype=np.int)

            self.indices = np.hstack((self.indices, indices_blank))

    def get_water_weight_array(self):

        self.water_weights = np.zeros(shape = (self.L, self.W, 9))

        for i in range(self.L):
            for j in range(self.W):
                stage_nbrs = self.pad_stage[i - 1 + 1:i + 2 + 1, j - 1 + 1:j + 2 + 1]
                depth_nbrs = self.pad_depth[i - 1 + 1:i + 2 + 1, j - 1 + 1:j + 2 + 1]
                ct_nbrs = self.pad_cell_type[i - 1 + 1:i + 2 + 1, j - 1 + 1:j + 2 + 1]
                self.water_weights[i, j] = shared_tools.get_weight_at_cell(
                    (i, j),
                    stage_nbrs.flatten(), depth_nbrs.flatten(), ct_nbrs.flatten(),
                    self.stage[i, j], self.qx[i, j], self.qy[i, j],
                    self.ivec.flatten(), self.jvec.flatten(), self.distances.flatten(),
                    self.dry_depth, self.gamma, self.theta_water)

    def get_new_cell(self, ind):
        weight = self.water_weights[ind[0], ind[1]]
        return shared_tools.random_pick(weight)

    def update_Q(self, dist, current_inds, next_index, astep, jstep, istep):

        self.qxn = shared_tools.update_dirQfield(self.qxn.flat[:], dist, current_inds, astep, jstep
                                          ).reshape(self.qxn.shape)
        self.qyn = shared_tools.update_dirQfield(self.qyn.flat[:], dist, current_inds, astep, istep
                                          ).reshape(self.qyn.shape)
        self.qwn = shared_tools.update_absQfield(self.qwn.flat[:], dist, current_inds, astep, self.Qp_water, self.dx
                                          ).reshape(self.qwn.shape)
        self.qxn = shared_tools.update_dirQfield(self.qxn.flat[:], dist, next_index, astep, jstep
                                          ).reshape(self.qxn.shape)
        self.qyn = shared_tools.update_dirQfield(self.qyn.flat[:], dist, next_index, astep, istep
                                          ).reshape(self.qyn.shape)
        self.qwn = shared_tools.update_absQfield(self.qwn.flat[:], dist, next_index, astep, self.Qp_water, self.dx
                                          ).reshape(self.qwn.shape)

    def check_for_boundary(self, inds):

        self.free_surf_flag[(self.cell_type.flat[inds] == -1) & (self.free_surf_flag == 0)] = 1

        self.free_surf_flag[(self.cell_type.flat[inds] == -1) & (self.free_surf_flag == -1)] = 2

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
        """Flood dry cells along the shore if necessary.

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
        """Get wet mask.

        Returns np.array((8,L,W)), for each neighbor around a cell
        with 1 if the neighbor is wet and 0 if dry
        """
        wet_mask = (self.depth > self.dry_depth) * 1
        wet_mask_nh = self.build_weight_array(wet_mask, fix_edges=True)

        return wet_mask_nh

    def build_weight_array(self, array, fix_edges=False, normalize=False):
        """Weighting array of neighbors.

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
        """Update water discharge after one water iteration."""
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
        """Update the flow velocity field after one water iteration."""
        mask = (self.depth > self.dry_depth) * (self.qw > 0)

        self.uw[mask] = np.minimum(
            self.u_max, self.qw[mask] / self.depth[mask])
        self.uw[~mask] = 0
        self.ux[mask] = self.uw[mask] * self.qx[mask] / self.qw[mask]
        self.ux[~mask] = 0
        self.uy[mask] = self.uw[mask] * self.qy[mask] / self.qw[mask]
        self.uy[~mask] = 0
