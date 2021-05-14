
import numpy as np
from numba import njit
from numba import float32, int64
from numba.experimental import jitclass
from scipy import ndimage
import abc

import warnings

from . import shared_tools

# tools for sediment routing algorithms and deposition/erosion


class sed_tools(abc.ABC):

    def route_sediment(self):
        """Sediment routing main method.

        This is the main method for sediment routing in the model. It is
        called once per `update()` call.

        Internally, this method calls:
            * :obj:`route_all_sand_parcels`
            * :obj:`topo_diffusion`
            * :obj:`route_all_mud_parcels`
        """
        _msg = 'Beginning sediment iteration'
        self.log_info(_msg, verbosity=2)

        # initialize the relevant fields and parcel trackers
        self.hook_init_sediment_iteration()
        self.init_sediment_iteration()

        _msg = 'Beginning sand parcel routing'
        self.log_info(_msg, verbosity=2)
        self.hook_route_all_sand_parcels()
        self.route_all_sand_parcels()

        _msg = 'Beginning topographic diffusion'
        self.log_info(_msg, verbosity=2)
        self.hook_topo_diffusion()
        self.topo_diffusion()

        _msg = 'Beginning mud parcel routing'
        self.log_info(_msg, verbosity=2)
        self.hook_route_all_mud_parcels()
        self.route_all_mud_parcels()

    def sed_route(self):
        """Deprecated, since v1.3.1. Use :obj:`route_sediment`."""
        _msg = ('`sed_route` and `hook_sed_route` are deprecated and '
                'have been replaced with `route_sediment`. '
                'Running `route_sediment` now, but '
                'this will be removed in future release.')
        self.logger.warning(_msg)
        warnings.warn(UserWarning(_msg))
        self.route_sediment()

    def init_sediment_iteration(self):
        """Init the water iteration routine.

        Clear and pad fields in preparation for iterating parcels.
        """
        _msg = 'Initializing water iteration'
        self.log_info(_msg, verbosity=2)

        # pad with edge on depth
        self.pad_depth = np.pad(self.depth, 1, 'edge')

        # clear sediment flux field and deposit volumes
        self.qs[:] = 0
        self.Vp_dep_sand[:] = 0
        self.Vp_dep_mud[:] = 0

    def route_all_sand_parcels(self):
        """Route sand parcels; topo diffusion.

        This method largely wraps the :obj:`SandRouter`. First, the number of
        parcels and sand fraction (:obj:`~pyDeltaRCM.DeltaModel.f_bedload`)
        are used to determine starting locations for sand parcels. Next, these
        locations are sent to the `SandRouter`, along with many other model
        state variables.

        Finally, variables are unpacked from the `SandRouter` and updated in
        the model fields, where they are later used by the `MudRouter` and the
        water parcel routing.

        Examples
        --------

        The cumulative effect of routing all sand parcels:

        +-------------------------------------------+------------------------------------------------+
        | .. plot:: sed_tools/_initial_bed_state.py | .. plot:: sed_tools/route_all_sand_parcels.py  |
        +-------------------------------------------+------------------------------------------------+
        """
        _msg = 'Determining sand parcel start indicies'
        self.log_info(_msg, verbosity=2)

        num_starts = int(self._Np_sed * self._f_bedload)
        inlet_weights = np.ones_like(self.inlet)
        start_indices = shared_tools.get_start_indices(self.inlet,
                                                       inlet_weights,
                                                       num_starts)

        _msg = 'Supplying model state to SandRouter for iteration'
        self.log_info(_msg, verbosity=2)

        self._sr.run(start_indices, self.eta, self.stage, self.depth,
                     self.cell_type, self.uw, self.ux, self.uy,
                     self.Vp_dep_mud, self.Vp_dep_sand,
                     self.qw, self.qx, self.qy, self.qs)

        # These are the variables updated at the end of the `SandRouter`. If
        # you attempt to drop in a replacement SandRouter, you will need to
        # update these fields!!
        _msg = 'Updating DeltaModel based on SandRouter change'
        self.log_info(_msg, verbosity=2)

        self.Vp_dep_mud = self._sr.Vp_dep_mud
        self.Vp_dep_sand = self._sr.Vp_dep_sand
        self.eta = self._sr.eta  # update bed
        self.depth = self._sr.depth  # update depth
        self.pad_depth = self._sr.pad_depth  # update depth in padded array
        self.uw = self._sr.uw  # update absolute flow field
        self.ux = self._sr.ux  # update component flow field
        self.uy = self._sr.uy  # update component flow fielda
        self.qs = self._sr.qs

    def route_all_mud_parcels(self):
        """Route mud parcels.

        This method largely wraps the :obj:`MudRouter`. First, the number of
        parcels and sand fraction (:obj:`~pyDeltaRCM.DeltaModel.f_bedload`)
        are used to determine starting locations for mud parcels. Next, these
        locations are sent to the `MudRouter`, along with many other model
        state variables.

        Finally, variables are unpacked from the `MudRouter` and updated in
        the model fields, where they are later used by the water parcel
        routing.

        Examples
        --------

        The cumulative effect of routing all sand parcels:

        +-------------------------------------------+------------------------------------------------+
        | .. plot:: sed_tools/_initial_bed_state.py | .. plot:: sed_tools/route_all_mud_parcels.py   |
        +-------------------------------------------+------------------------------------------------+
        """
        _msg = 'Determining mud parcel start indicies'
        self.log_info(_msg, verbosity=2)

        num_starts = int(self._Np_sed * (1 - self._f_bedload))
        inlet_weights = np.ones_like(self.inlet)
        start_indices = shared_tools.get_start_indices(self.inlet,
                                                       inlet_weights,
                                                       num_starts)

        _msg = 'Supplying model state to MudRouter for iteration'
        self.log_info(_msg, verbosity=2)

        self._mr.run(start_indices, self.eta, self.stage, self.depth,
                     self.cell_type, self.uw, self.ux, self.uy,
                     self.Vp_dep_mud, self.Vp_dep_sand,
                     self.qw, self.qx, self.qy)

        # These are the variables updated at the end of the `MudRouter`. If
        # you attempt to drop in a replacement MudRouter, you will need to
        # update these fields!!
        _msg = 'Updating DeltaModel based on MudRouter change'
        self.log_info(_msg, verbosity=2)

        self.Vp_dep_mud = self._mr.Vp_dep_mud
        self.Vp_dep_sand = self._mr.Vp_dep_sand
        self.eta = self._mr.eta  # update bed
        self.depth = self._mr.depth  # update depth
        self.pad_depth = self._mr.pad_depth  # update depth in padded array
        self.uw = self._mr.uw  # update absolute flow field
        self.ux = self._mr.ux  # update component flow field
        self.uy = self._mr.uy  # update component flow field

    def topo_diffusion(self):
        """Diffuse topography after routing.

        Diffuse topography after routing all coarse sediment parcels. The
        operation is repeated `N_crossdiff` times.
        """
        for _ in range(self.N_crossdiff):

            a = ndimage.convolve(self.eta, self.kernel1, mode='constant')
            b = ndimage.convolve(self.qs, self.kernel2, mode='constant')
            c = ndimage.convolve(self.qs * self.eta, self.kernel2,
                                 mode='constant')

            self.cf = (self.diffusion_multiplier *
                       (self.qs * a - self.eta * b + c))

            self.cf[self.cell_type == -2] = 0
            self.cf[0, :] = 0

            self.eta += self.cf


@njit
def _get_weight_at_cell_sediment(ind, weight_int, depth_nbrs, ct_nbrs,
                                 dry_depth, theta, distances_flat):
    """Get neighbor weight array for sediment routing.

    .. todo::

        Expand description. Include example? Equation? Link to Morphodynamics
        document.
    """
    dry = (depth_nbrs <= dry_depth)
    wall = (ct_nbrs == -2)
    ctr = (np.arange(9) == 4)
    drywall = np.logical_or(dry, wall)

    # always set ctr to 0 before rebalancing
    weight_int[ctr] = 0

    weight = np.copy(weight_int)  # no gamma weighting here
    weight = (depth_nbrs ** theta) * weight

    # ALWAYS disallow the choice to not move
    weight[ctr] = 0

    # ALWAYS disallow the choice to go into land or dry cells
    weight[drywall] = 0

    # sanity check
    if np.any(np.isnan(weight)):
        raise RuntimeError('NaN encountered in sediment weighting.'
                           'Please report error.')

    # correct the weights for random choice
    if np.sum(weight) == 0:
        # if no weights
        #  convert to random walk into any non-wall,
        #  (incl dry cells), weighted by distance
        weight[:] = 1 / distances_flat

        # disallow movement into walls
        weight[wall] = 0

    weight[ctr] = 0  # enforce

    # sanity check
    weight_sum = np.sum(weight)
    if weight_sum == 0:
        raise RuntimeError('No weights encountered in sediment weighting.'
                           'Please report error.')

    # final rebalance
    weight = weight / weight_sum

    return weight


r_spec = [('_dt', float32), ('_dx', float32),
          ('num_starts', int64), ('start_indices', int64[:]),
          ('stepmax', float32), ('px', int64), ('py', int64),
          ('eta', float32[:, :]), ('stage', float32[:, :]),
          ('depth', float32[:, :]), ('cell_type', int64[:, :]),
          ('uw', float32[:, :]), ('ux', float32[:, :]), ('uy', float32[:, :]),
          ('pad_stage', float32[:, :]), ('pad_depth', float32[:, :]),
          ('pad_cell_type', int64[:, :]), ('qw', float32[:, :]),
          ('qx', float32[:, :]), ('qy', float32[:, :]), ('qs', float32[:, :]),
          ('ivec_flat', float32[:]), ('jvec_flat', float32[:]),
          ('iwalk_flat', int64[:]), ('jwalk_flat', int64[:]),
          ('distances_flat', float32[:]),
          ('dry_depth', float32), ('_lambda', float32),
          ('_beta', float32),  ('_f_bedload', float32),
          ('theta_sed', float32), ('u_max', float32),
          ('qs0', float32), ('_u0', float32), ('Vp_sed', float32),
          ('Vp_res', float32), ('Vp_dep_mud', float32[:, :]),
          ('Vp_dep_sand', float32[:, :]),
          ('U_dep_mud', float32), ('U_ero_mud', float32),
          ('U_ero_sand', float32)]


class BaseRouter(object):
    """BaseRouter.

    Defines common methods for jitted routers.

    Subclasses need to define `run`, `_route_one_parcel`, and
    `_deposit_or_erode`.

    .. note::

        Although this class is configured with `@abc.abstractmethod` methods,
        we cannot actually enforce the implementation of this `BaseRouter` as
        an abstract class, because Numba does not support abstract classes
        (yet; https://github.com/numba/numba/issues/6033). The classes are
        included in the base class with abstract decorators to help make it
        clear that these methods need to be implemented in subclassing
        `Router`.
    """
    @abc.abstractmethod
    def run(self, *args, **kwargs):
        ...

    @abc.abstractmethod
    def _route_one_parcel(self):
        ...

    def _choose_next_location(self, px, py):

        # choose next location with weights
        stage_nbrs = self.pad_stage[
            px:px + 3, py:py + 3]
        depth_nbrs = self.pad_depth[
            px:px + 3, py:py + 3]
        cell_type_ind = self.pad_cell_type[
            px:px + 3, py:py + 3]

        _, weight_int = shared_tools.get_weight_sfc_int(
            self.stage[px, py], stage_nbrs.ravel(), self.qx[px, py],
            self.qy[px, py], self.ivec_flat, self.jvec_flat,
            self.distances_flat)

        if np.any(np.isnan(weight_int)):
            raise RuntimeError('NaN in weight_int.')

        if not np.all(np.isfinite(depth_nbrs.ravel())):
            raise RuntimeError('nonfinite in depth_nbrs.')

        weights = _get_weight_at_cell_sediment(
            (px, py), weight_int, depth_nbrs.ravel(),
            cell_type_ind.ravel(), self.dry_depth, self.theta_sed, self.distances_flat)

        new_cell = shared_tools.random_pick(weights)

        dist, istep, jstep, _ = shared_tools.get_steps(
            new_cell, self.iwalk_flat, self.jwalk_flat)

        return istep, jstep, dist

    @abc.abstractmethod
    def _deposit_or_erode(self, px, py):
        """Determine whether to erode or deposit.

        This is the decision making component of the routine, and will be
        different for sand vs mud.
        """
        ...

    def _update_fields(self, Vp_change, px, py):
        """Execute deposit of sand or mud.

        Deposit sediment volume `Vp_change`. The change in bed elevation
        depends on the sediment mass conservation (i.e., Exner equation) and
        is equal to:

        .. code::

            Vp_change / (dx * dx)

        Following the sediment deposition/erosion, the new values for flow
        depth and flow velocity fields are determined.

        .. note::

            Total sediment mass is preserved, but individual categories
            of sand and mud are not. I.e., it is assumed that there is infinite
            sand and/or mud to erode at any location where erosion is
            occurring.

        Parameters
        ----------
        Vp_change : :obj:`float`
            Volume of sediment to deposit / erode. If erosion, `Vp_change`
            should be negative.

        px : :obj:`int`
            Index.

        py : :obj:`int`
            Index.

        Returns
        -------
        """
        # Compute the changes in the bed elevation, and some flow fields
        #    The :obj:`_update_helper` function is a jitted routine which
        #    computes the new values for the bed elevation and then determines
        #    the updated values of the depth and flow velocity fields. These
        #    determinations require several comparisons and repeated indexing,
        #    so we use a jitted "helper" function to do the operations.
        qw0 = self.qw[px, py]
        eta_change = Vp_change / (self._dx * self._dx)

        eta = self.eta[px, py] + eta_change  # new bed
        depth = self.stage[px, py] - eta  # new depth
        depth = np.maximum(0, depth)  # force >=0

        if depth > 0:
            uw = np.minimum(self.u_max, qw0 / depth)
        else:
            uw = self.uw[px, py]

        # now apply the computed updated values
        self.eta[px, py] = eta  # update bed
        self.depth[px, py] = depth  # update depth
        self.pad_depth[px + 1, py + 1] = depth  # update depth in padded array
        self.uw[px, py] = uw  # update absolute flow field
        # update component flow fields
        if qw0 > 0:
            self.ux[px, py] = uw * self.qx[px, py] / qw0
            self.uy[px, py] = uw * self.qy[px, py] / qw0
        else:
            self.ux[px, py] = 0
            self.uy[px, py] = 0

    def _compute_Vp_ero(self, Vp_sed, U_loc, U_ero, beta):
        """Compute volume erorded based on velocity.

        The volume of sediment eroded depends on the local flow velocity
        (:obj:`U_loc`), the critical velocity for erosion (:obj:`U_ero`), and
        `beta`.

        Function is used by both sand and mud pathways, but the input value of
        U_ero depends on the pathway.

        Parameters
        ----------
        Vp_sed : :obj:`float`
            Volume of a sediment parcel.

        U_loc : :obj:`float`
            Local flow velocity.

        U_ero : :obj:`float`
            Critical velocity for erosion.

        beta : :obj:`float`
            unknown.

        Returns
        -------
        Vp_ero : :obj:`float`
            Volume of eroded sediment.
        """
        return (Vp_sed * (U_loc**beta - U_ero**beta) /
                U_ero**beta)

    def _limit_Vp_change(self, Vp, stage, eta, dx, dep_ero):
        """Limit change in volume to 1/4 of a cell volume.

        Function is used by multiple pathways in `mud_dep_ero` and `sand_dep_ero`
        but with different inputs for the sediment volume (:obj:`Vp`).

        dep_ero indicates which pathway we are in, dep==0, ero==1
        """
        depth = stage - eta
        if depth < 0:
            if dep_ero == 0:
                return 0
            else:
                fourth = np.abs(depth) / 4 * (dx * dx)
                return np.minimum(Vp, fourth)
        else:
            fourth = depth / 4 * (dx * dx)
            return np.minimum(Vp, fourth)


@jitclass(r_spec)
class SandRouter(BaseRouter):
    """Jitted class to route sand.

    Initialized in
    :obj:`~pyDeltaRCM.init_tools.init_tools.init_sediment_routers` with a
    multitude of constants. The `Router` is then called via the public
    :obj:`run` method in the :obj:`~pyDeltaRCM.sed_tools.sed_tools.sed_route`
    method.

    .. important::

        A new Router must be initialized if any constants are changed to the
        underlying `DeltaModel` object. Convention is to call
        :obj:`~pyDeltaRCM.init_tools.init_tools.init_sediment_routers` in any
        property `setter`.
    """
    def __init__(self, _dt, dx, Vp_sed, u_max, qs0, u0, U_ero_sand, f_bedload,
                 ivec_flat, jvec_flat, iwalk_flat, jwalk_flat, distances_flat,
                 dry_depth, beta, stepmax, theta_sed):

        self._dt = _dt
        self._dx = dx
        self.Vp_sed = Vp_sed

        self.u_max = u_max
        self.qs0 = qs0
        self._u0 = u0
        self.U_ero_sand = U_ero_sand
        self._f_bedload = f_bedload

        self.ivec_flat, self.jvec_flat,  = ivec_flat, jvec_flat
        self.iwalk_flat, self.jwalk_flat = iwalk_flat, jwalk_flat
        self.distances_flat = distances_flat

        self.dry_depth = dry_depth
        self._beta = beta
        self.stepmax = stepmax
        self.theta_sed = theta_sed

    def run(self, start_indices, eta, stage, depth, cell_type,
            uw, ux, uy, Vp_dep_mud, Vp_dep_sand,
            qw, qx, qy, qs):
        """The main function to route and deposit/erode sand parcels.

        Algorithm is to:

            1. as input, receive the current status of fields from the model.
            Additionally, receive a list of the starting points to use to run
            parcels, as :obj:`px` and :obj:`py`.

            2. begin a `for` loop to run each parcel in series.

            3. in the :obj:`SandRouter`, the sediment partitioning from a
            ghost node is executed. This step is skipped for
            :obj:`MudRouter`.

            4. call :obj:`_route_one_parcel` method to run a single parcel of
            sediment through all iterations.

            5. repeat from 3, until the correct number of parcels have been
            routed. Note that the number of *total* parcels is
            :obj:`~pyDeltaRCM.DeltaModel.Np_sed`, and the number of sand or
            mud parcels will depend on the value of
            :obj:`~pyDeltaRCM.DeltaModel.f_bedload`.

        .. note::

            We are unable to precompute the routing weights, in the way
            we do in :obj:`~pyDeltaRCM.water_tools.get_water_weight_array`,
            because the weighting changes with each parcel step (i.e.,
            morphodynamics).
        """
        self.eta = eta
        self.stage = stage
        self.depth = depth
        self.cell_type = cell_type
        self.uw = uw
        self.ux = ux
        self.uy = uy
        self.pad_stage = shared_tools.custom_pad(stage)
        self.pad_depth = shared_tools.custom_pad(depth)
        self.pad_cell_type = shared_tools.custom_pad(cell_type)
        self.Vp_dep_mud = Vp_dep_mud
        self.Vp_dep_sand = Vp_dep_sand
        self.qw = qw
        self.qx = qx
        self.qy = qy
        self.qs = qs

        num_starts = start_indices.shape[0]
        for np_sed in range(num_starts):

            self.Vp_res = self.Vp_sed

            px = 0
            py = start_indices[np_sed]

            self.qs[px, py] = (self.qs[px, py] +
                               self.Vp_res / 2. / self._dt / self._dx)
            self._route_one_parcel(px, py)

    def _route_one_parcel(self, px, py):
        """Route one parcel.

        Algorithm is to:

            1. as input, receive the starting points to use to run a single
            parcel, as :obj:`px` and :obj:`py`.

            2. begin a while loop, which counts the number of steps the parcel
            has taken

            3. determine, based on the water surface and flow velocity field,
            which location to travel to next. This determination is named as
            :obj:`choose_next_location`, which utilizes the `shared_tools`
            methods to pick the next location.

            4. call the :obj:`_deposit_or_erode` method to detemine whether to
            deposit or erode sediment. This method is implemented
            *differently* for sand and mud routing, and depends on a multitude
            of different model variables. This step modifies the bed elevation
            and flow depth fields (in subfunction :obj:`_update_fields`),
            which necessitates finding the new weights for routing on each
            step. Also, the volume of sediment is either increased or
            decreased (erosion or deposition).

            5. repeat from 3, until `stepmax` is reached, or an "edge" cell is
            reached.
        """
        it = 0
        sed_continue = True

        while sed_continue:

            px0 = px
            py0 = py

            # Choose the next location for the parcel to travel
            istep, jstep, dist = self._choose_next_location(px0, py0)

            px = (px0 + jstep)
            py = (py0 + istep)

            self._partition_sediment(px0, py0, px, py, dist)
            self._deposit_or_erode(px, py)

            it += 1
            if self.cell_type[px, py] == -1:  # check for "edge" cell
                sed_continue = False  # kill the `while` loop
            if (it == self.stepmax):
                sed_continue = False

    def _partition_sediment(self, px0, py0, px, py, dist):
        """Spread sediment flux between two cells.
        """
        partition = self.Vp_res / 2. / self._dt / self._dx
        if dist > 0:
            self.qs[px0, py0] += partition  # deposition in current cell
            self.qs[px, py] += partition  # deposition in new cell

    def _deposit_or_erode(self, px, py):
        """Decide if deposit or erode sand.

        .. note:: Volumetric change is limited to 1/4 local cell water volume.

        Sand deposition:
            If more sediment is in transport (`qs_loc`) than the determined
            transport capacity of the cell (`qs_cap`), sediment needs to
            deposit on the bed.
        Sand erosion:
            Can only occur if local velocity (`U_loc`) is greater than the
            critical erosion threshold for sand
            (:obj:`~pyDeltaRCM.DeltaModel.U_ero_sand`), *and* if the local
            transport capacity is not yet reached (`qs_loc < qs_cap`).
        """
        U_loc = self.uw[px, py]
        qs_cap = (self.qs0 * self._f_bedload / self._u0**self._beta *
                  U_loc**self._beta)
        qs_loc = self.qs[px, py]

        Vp_change = 0
        if qs_loc > qs_cap:
            # Sand deposition
            #     If more sediment is in transport than the determined
            #     transport capacity of the cell (`qs_cap`), sediment needs to
            #     deposit on the bed.
            Vp_change = self._limit_Vp_change(self.Vp_res, self.stage[px, py],
                                              self.eta[px, py], self._dx, 0)

        elif (U_loc > self.U_ero_sand) and (qs_loc < qs_cap):
            # Sand erosion
            #     Can only occur if local velocity is greater than the
            #     critical erosion threshold for sand, *and* if the local
            #     transport capacity is not yet reached.
            Vp_change = self._compute_Vp_ero(self.Vp_sed, U_loc,
                                             self.U_ero_sand, self._beta)
            Vp_change = self._limit_Vp_change(Vp_change, self.stage[px, py],
                                              self.eta[px, py], self._dx, 1)
            Vp_change = Vp_change * -1

        if Vp_change > 0:  # if deposition
            self.Vp_dep_sand[px, py] = self.Vp_dep_sand[px, py] + Vp_change

        self.Vp_res = self.Vp_res - Vp_change  # update sed volume in parcel

        self._update_fields(Vp_change, px, py)  # update other fields as needed


@jitclass(r_spec)
class MudRouter(BaseRouter):
    """Jitted class to route mud.

    Initialized in
    :obj:`~pyDeltaRCM.init_tools.init_tools.init_sediment_routers` with a
    multitude of constants. The `Router` is then called via the public
    :obj:`run` method in the :obj:`~pyDeltaRCM.sed_tools.sed_tools.sed_route`
    method.

    .. important::

        A new Router must be initialized if any constants are changed to the
        underlying `DeltaModel` object. Convention is to call
        :obj:`~pyDeltaRCM.init_tools.init_tools.init_sediment_routers` in any
        property `setter`.
    """
    def __init__(self, _dt, dx, Vp_sed, u_max, U_dep_mud, U_ero_mud,
                 ivec_flat, jvec_flat, iwalk_flat, jwalk_flat, distances_flat,
                 dry_depth, _lambda, beta, stepmax, theta_sed):

        self._dt = _dt
        self._dx = dx
        self.Vp_sed = Vp_sed

        self.u_max = u_max
        self.U_dep_mud = U_dep_mud
        self.U_ero_mud = U_ero_mud

        self.ivec_flat, self.jvec_flat,  = ivec_flat, jvec_flat
        self.iwalk_flat, self.jwalk_flat = iwalk_flat, jwalk_flat
        self.distances_flat = distances_flat

        self.dry_depth = dry_depth
        self._lambda = _lambda
        self._beta = beta
        self.stepmax = stepmax
        self.theta_sed = theta_sed

    def run(self, start_indices, eta, stage, depth, cell_type,
            uw, ux, uy, Vp_dep_mud, Vp_dep_sand,
            qw, qx, qy):
        """The main function to route and deposit/erode mud parcels.

        """

        self.eta = eta
        self.stage = stage
        self.depth = depth
        self.cell_type = cell_type
        self.uw = uw
        self.ux = ux
        self.uy = uy
        self.pad_stage = shared_tools.custom_pad(stage)
        self.pad_depth = shared_tools.custom_pad(depth)
        self.pad_cell_type = shared_tools.custom_pad(cell_type)
        self.Vp_dep_mud = Vp_dep_mud
        self.Vp_dep_sand = Vp_dep_sand
        self.qw = qw
        self.qx = qx
        self.qy = qy

        num_starts = start_indices.shape[0]
        for np_sed in range(num_starts):

            self.Vp_res = self.Vp_sed

            px = 0
            py = start_indices[np_sed]

            self._route_one_parcel(px, py)

    def _route_one_parcel(self, px, py):
        """Route one parcel.

        """
        it = 0
        sed_continue = True

        while sed_continue:

            # Choose the next location for the parcel to travel
            istep, jstep, _ = self._choose_next_location(px, py)

            px = px + jstep
            py = py + istep

            self._deposit_or_erode(px, py)

            it += 1
            if self.cell_type[px, py] == -1:  # check for "edge" cell
                sed_continue = False  # kill the `while` loop
            if (it == self.stepmax):
                sed_continue = False

    def _deposit_or_erode(self, px, py):
        """Decide if deposit or erode mud.

        .. note:: Volumetric change is limited to 1/4 local cell water volume.

        .. important:: TODO: complete description specific for mud transport
        """
        U_loc = self.uw[px, py]

        Vp_change = 0
        if U_loc < self.U_dep_mud:
            Vp_change = (self._lambda * self.Vp_res *
                         (self.U_dep_mud**self._beta - U_loc**self._beta) /
                         (self.U_dep_mud**self._beta))
            Vp_change = self._limit_Vp_change(Vp_change, self.stage[px, py],
                                              self.eta[px, py], self._dx, 0)

        if U_loc > self.U_ero_mud:
            Vp_change = self._compute_Vp_ero(self.Vp_sed, U_loc,
                                             self.U_ero_mud, self._beta)
            Vp_change = self._limit_Vp_change(Vp_change, self.stage[px, py],
                                              self.eta[px, py], self._dx, 1)
            Vp_change = Vp_change * -1

        if Vp_change > 0:  # if deposition
            self.Vp_dep_mud[px, py] = self.Vp_dep_mud[px, py] + Vp_change

        self.Vp_res = self.Vp_res - Vp_change  # update sed volume in parcel

        self._update_fields(Vp_change, px, py)  # update other fields as needed
