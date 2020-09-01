
import numpy as np
from numba import njit, jit, typed

from . import shared_tools

# tools for water routing algorithms


class water_tools(object):

    def init_water_iteration(self):

        self.qxn[:] = 0
        self.qyn[:] = 0
        self.qwn[:] = 0

        self.free_surf_flag[:] = 0
        self.free_surf_walk_indices[:] = 0
        self.sfc_visit[:] = 0
        self.sfc_sum[:] = 0

        self.pad_stage = np.pad(self.stage, 1, 'edge')
        self.pad_depth = np.pad(self.depth, 1, 'edge')
        self.pad_cell_type = np.pad(self.cell_type, 1, 'edge')

    def run_water_iteration(self):
        """Run a single iteration of travel paths for all water parcels.

        Runs all water  parcels (`Np_water` parcels) for `stepmax` steps, or
        until the parcels reach a boundary.

        All parcels are processed in parallel, taking one step for each loop
        of the ``while`` loop.
        """

        _step = 0  # the step number of parcels
        inlet_weights = np.ones_like(self.inlet)
        start_indices = shared_tools.get_start_indices(self.inlet,
                                                       inlet_weights,
                                                       self.Np_water)

        self.qxn.flat[start_indices] += 1
        self.qwn.flat[start_indices] += self.Qp_water / self.dx / 2

        self.free_surf_walk_indices[:, 0] = start_indices
        current_inds = np.copy(start_indices)

        self.looped[:] = 0

        self.get_water_weight_array()
        water_weights_flat = self.water_weights.reshape(-1, 9)

        while (sum(current_inds) > 0) & (_step < self.stepmax):

            _step += 1

            self.check_size_of_indices_matrix(_step)

            # use water weights and random pick to determine d8 direction
            new_direction = pick_d8_direction(current_inds, water_weights_flat)
            new_direction = new_direction.astype(np.int)

            new_indices = calculate_new_ind(
                current_inds,
                new_direction,
                self.iwalk_flat,
                self.jwalk_flat,
                self.eta.shape)

            dist, istep, jstep, astep = shared_tools.get_steps(
                new_direction,
                self.iwalk_flat,
                self.jwalk_flat)

            self.update_Q(dist, current_inds, new_indices, astep, jstep, istep)

            current_inds, self.looped, self.free_surf_flag = check_for_loops(
                self.free_surf_walk_indices, new_indices, _step, self.L0, self.looped,
                self.eta.shape, self.CTR, self.free_surf_flag)

            # Record the parcel pathways for computing the free surface
            #     Parcels that have reached the boundary are updated to
            #     ``ind==0``, effectively ending the routing of these parcels.
            curent_inds = self.check_for_boundary(current_inds)  # changes `free_surf_flag`
            self.free_surf_walk_indices[:, _step] = current_inds  # record indices
            current_inds[self.free_surf_flag > 0] = 0

    def compute_free_surface(self):
        """Calculate free surface after routing one water parcel."""

        self.sfc_visit, self.sfc_sum = _accumulate_free_surface_walks(
            self.free_surf_walk_indices, self.looped, self.cell_type,
            self.uw, self.ux, self.uy, self.depth,
            self.dx, self.u0, self.h0, self.H_SL, self.S0)

    def finalize_water_iteration(self, iteration):
        """Finish updating flow fields.

        Clean up at end of water iteration
        """
        self.update_water()

        self.stage[:] = np.maximum(self.stage, self.H_SL)
        self.depth[:] = np.maximum(self.stage - self.eta, 0)

        self.update_flow_field(iteration)
        self.update_velocity_field()

    def check_size_of_indices_matrix(self, it):
        if it >= self.free_surf_walk_indices.shape[1]:
            """
            Initial size of self.free_surf_walk_indices is half of self.stepmax
            because the number of iterations doesn't go beyond
            that for many timesteps.

            Once it reaches it > self.stepmax/2 once, make the size
            self.iter for all further timesteps
            """
            _msg = 'Increasing size of self.free_surf_walk_indices'
            self.logger.info(_msg)
            if self.verbose >= 2:
                print(_msg)

            indices_blank = np.zeros(
                (np.int(self.Np_water), np.int(self.stepmax / 4)), dtype=np.int)

            self.free_surf_walk_indices = np.hstack((self.free_surf_walk_indices, indices_blank))

    def get_water_weight_array(self):

        self.water_weights = np.zeros(shape=(self.L, self.W, 9))

        for i in range(self.L):
            for j in range(self.W):
                stage_nbrs = self.pad_stage[i - 1 + 1:i + 2 + 1, j - 1 + 1:j + 2 + 1]
                depth_nbrs = self.pad_depth[i - 1 + 1:i + 2 + 1, j - 1 + 1:j + 2 + 1]
                ct_nbrs = self.pad_cell_type[i - 1 + 1:i + 2 + 1, j - 1 + 1:j + 2 + 1]

                weight_sfc, weight_int = shared_tools.get_weight_sfc_int(
                    self.stage[i, j], stage_nbrs.ravel(),
                    self.qx[i, j], self.qy[i, j], self.ivec_flat, self.jvec_flat,
                    self.distances_flat)

                self.water_weights[i, j] = shared_tools.get_weight_at_cell(
                    (i, j), weight_sfc, weight_int,
                    depth_nbrs.ravel(), ct_nbrs.ravel(),
                    self.dry_depth, self.gamma, self.theta_water)


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
        """Check whether parcels have reached the boundary.

        Checks whether any parcels have reached the model boundaries. If they
        have, then update the information in :obj:`~pyDeltaRCM.DeltaModel.free_surf_flag`.

        Parameters
        ----------
        inds : :obj:`ndarray`
            Unraveled indicies of parcels.
        """
        # where cell type is "edge" and free_surf_flag is currently valid
        self.free_surf_flag[(self.cell_type.flat[inds] == -1) & (self.free_surf_flag == 0)] = 1

        # where cell type is "edge" and free_surf_flag is currently valid
        self.free_surf_flag[(self.cell_type.flat[inds] == -1) & (self.free_surf_flag == -1)] = 2

        # below is not needed, because of update at end of `iteration`. I'm
        #   leaving it here for now. If removed, can also remove the return?
        inds[self.free_surf_flag == 2] = 0
        return inds

    def update_water(self):
        """Update surface after routing all parcels.

        Could divide into 3 functions for cleanliness.
        """
        Hnew = self.eta + self.depth

        # water surface height not under sea level
        Hnew[Hnew < self.H_SL] = self.H_SL

        # find average water surface elevation for a cell
        Hnew[self.sfc_visit > 0] = (self.sfc_sum[self.sfc_visit > 0] /
                                    self.sfc_visit[self.sfc_visit > 0])

        # smooth newly calculated free surface
        Hnew_pad = np.pad(Hnew, 1, 'edge')
        Hsmth = _smooth_water_surface(
            Hnew, Hnew_pad, self.cell_type, self.pad_cell_type,
            self.Nsmooth, self.Csmooth)

        if self._time_iter > 0:
            self.stage = ((1 - self.omega_sfc) * self.stage +
                          self.omega_sfc * Hsmth)

        self.flooding_correction()

    def update_flow_field(self, iteration):
        """Update water discharge after one water iteration."""
        dloc = (self.qxn**2 + self.qyn**2)**(0.5)

        qwn_div = np.ones((self.L, self.W))
        qwn_div[dloc > 0] = self.qwn[dloc > 0] / dloc[dloc > 0]

        self.qxn *= qwn_div
        self.qyn *= qwn_div

        if self._time_iter > 0:

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
            #    so they cannot be > eta_shore[i]
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


@njit
def pick_d8_direction(inds, water_weights):
    """Get new cell locations, based on water weights.

    Algorithm is to:
        1. loop through each parcel, which is described by a pair in the
           `inds` array.
        2. determine the water weights for that location (from `water_weights`)
        3. choose a new cell based on the probabilities of the weights (using
           the `random_pick` function)

    Parameters
    ----------
    inds : :obj:`ndarray`
        Current unraveled indices of the parcels. ``(N,)  `ndarray` containing
        the unraveled indices.

    water_weights : :obj:`ndarray`
        Weights of every water cell. ``(LxW, 9)`` `ndarray`, uses unraveled
        indicies along 0th dimension; 9 cells represent self and 8 neighboring
        cells.

    Returns
    -------
    new_cells : :obj:`ndarray`
        The new cell for water parcels, relative to the current location.
        I.e., this is the D8 direction the parcel is going to travel in the
        next stage, :obj:`pyDeltaRCM.shared_tools.calculate_new_ind`.
    """
    new_cells = []
    for i in np.arange(inds.shape[0]):
        ind = inds[i]
        if ind != 0:
            weight = water_weights[ind, :]
            new_cells.append(shared_tools.random_pick(weight))
        else:
            new_cells.append(4)

    new_cells = np.array(new_cells)
    return new_cells


@njit
def calculate_new_ind(indices, new_cells, iwalk, jwalk, domain_shape):
    """Calculate the new location (indices) of parcels.

    Use the information of the current parcel (`indices`) in conjunction with
    the D8 direction the parcel needs to travel (`new_cells`) to determine the
    new indices of each parcel.
    """
    newbies = []
    for p, q in zip(indices, new_cells):
        if q != 4:
            ind_tuple = shared_tools.custom_unravel(p, domain_shape)
            new_ind = (ind_tuple[0] + jwalk[q],
                       ind_tuple[1] + iwalk[q])
            newbies.append(shared_tools.custom_ravel(new_ind, domain_shape))
        else:
            newbies.append(0)

    return np.array(newbies)


@njit
def check_for_loops(indices, inds, it, L0, loopedout, domain_shape, CTR, free_surf_flag):

    looped = typed.List()  # numba typed list for iteration
    for i in np.arange(len(indices)):
        row = indices[i, :]
        v = len(row[row > 0]) != len(set(row[row > 0]))
        looped.append(v)
    travel = (0, it)

    for n in range(indices.shape[0]):
        ind = inds[n]
        if looped[n] and (ind > 0) and (max(travel) > L0):
            loopedout[n] += 1
            px, py = shared_tools.custom_unravel(ind, domain_shape)
            Fx = px - 1
            Fy = py - CTR

            Fw = np.sqrt(Fx**2 + Fy**2)

            if Fw != 0:
                px = px + np.round(Fx / Fw * 5.)
                py = py + np.round(Fy / Fw * 5.)

            px = max(px, L0)
            px = int(min(domain_shape[0] - 2, px))

            py = max(1, py)
            py = int(min(domain_shape[1] - 2, py))

            nind = shared_tools.custom_ravel((px, py), domain_shape)

            inds[n] = nind

            free_surf_flag[n] = -1

    return inds, loopedout, free_surf_flag


@njit
def _accumulate_free_surface_walks(free_surf_walk_indices, looped, cell_type,
                      uw, ux, uy, depth,
                      dx, u0, h0, H_SL, S0):

    _shape = uw.shape
    Hnew = np.zeros(_shape)
    sfc_visit = np.zeros(_shape)
    sfc_sum = np.zeros(_shape)

    # for every parcel, walk the path of the parcel
    for p, inds in enumerate(free_surf_walk_indices):

        # unravel the indices of the parcel
        inds_whr = inds[inds > 0]  # where the path has meaningful values
        xs = np.zeros_like(inds_whr)  # x coordinates
        ys = np.zeros_like(inds_whr)  # x coordinates
        for pp, ind_whr in np.ndenumerate(inds_whr):
            xs[pp], ys[pp] = shared_tools.custom_unravel(ind_whr, _shape)

        # determine whether the pathway contributes to the free surface
        Hnew[:] = 0
        if ((cell_type[xs[-1], ys[-1]] == -1) and
                (looped[p] == 0)):

            # if cell is in ocean, H = H_SL (downstream boundary condition)
            Hnew[xs[-1], ys[-1]] = H_SL

            # counting back from last cell visited
            in_ocean = True  # whether we are in the ocean or not
            dH = 0
            for it in range(len(xs) - 2, -1, -1):

                i = int(xs[it])
                ip = int(xs[it + 1])
                j = int(ys[it])
                jp = int(ys[it + 1])
                dist = np.sqrt((ip - i)**2 + (jp - j)**2)

                if dist > 0:

                    # if in the ocean (not reached the shoreline yet)
                    if in_ocean:
                        # see if it is shoreline
                        if ((uw[i, j] > u0 * 0.5) or (depth[i, j] < 0.1 * h0)):
                            in_ocean = False  # passed the shoreline
                    # otherwise, in the delta
                    else:
                        # if no velocity
                        if uw[i, j] == 0:
                            dH = 0  # no change in water surface elevation
                        else:
                            # diff between streamline and parcel path
                            dH = (S0 * (ux[i, j] * (ip - i) * dx +
                                        uy[i, j] * (jp - j) * dx) / uw[i, j])

                # previous cell's surface plus difference in H
                Hnew[i, j] = Hnew[ip, jp] + dH

                # add up # of cell visits
                sfc_visit[i, j] = sfc_visit[i, j] + 1

                # sum of all water surface elevations
                sfc_sum[i, j] = sfc_sum[i, j] + Hnew[i, j]

    return sfc_visit, sfc_sum


@njit
def _smooth_water_surface(Hnew, Hnew_pad, cell_type, pad_cell_type,
                          Nsmooth, Csmooth):

    L, W = cell_type.shape
    Htemp = Hnew
    for itsmooth in range(Nsmooth):

        Hsmth = Htemp
        for i in range(L):
            for j in range(W):

                if cell_type[i, j] > -2:
                    # locate non-boundary cells
                    sumH = 0
                    nbcount = 0

                    ct_ind = pad_cell_type[
                        i - 1 + 1:i + 2 + 1, j - 1 + 1:j + 2 + 1]
                    Hnew_ind = Hnew_pad[
                        i - 1 + 1:i + 2 + 1, j - 1 + 1:j + 2 + 1]

                    Hnew_ind[1, 1] = 0
                    Hnew_ind = Hnew_ind.ravel()
                    _log = ct_ind.ravel() == -2
                    Hnew_ind[_log] = 0

                    sumH = np.sum(Hnew_ind)
                    nbcount = np.sum(Hnew_ind > 0)

                    if nbcount > 0:
                        # smooth if are not wall cells
                        Htemp[i, j] = (Csmooth * Hsmth[i, j] +
                                       (1 - Csmooth) * sumH / nbcount)

    Hsmth = Htemp
    return Hsmth
