# unit tests for init_tools.py

import pytest

import sys
import os
import numpy as np
sys.path.append(os.path.realpath(os.path.dirname(__file__) + "/.."))

from pyDeltaRCM.deltaRCM_driver import pyDeltaRCM
from pyDeltaRCM import init_tools

# need to create a simple case of pydeltarcm object to test these functions
np.random.seed(0)  # fix the random seed
delta = pyDeltaRCM(input_file=os.path.join(os.getcwd(), 'tests', 'test.yaml'))

# now that it is initiated can access the init_tools via the inherited object
# delta.**init_tools_function**


# Tests for all of the constants
delta.set_constants()


def test_set_constant_g():
    """
    check gravity
    """
    assert delta.g == 9.81


def test_set_constant_distances():
    """
    check distances
    """
    assert delta.distances[0, 0] == np.sqrt(2)


def test_set_ivec():
    assert delta.ivec[0, 0] == -np.sqrt(0.5)


def test_set_jvec():
    assert delta.jvec[0, 0] == -np.sqrt(0.5)


def test_set_iwalk():
    assert delta.iwalk[0, 0] == -1


def test_set_jwalk():
    assert delta.jwalk[0, 0] == -1


def test_set_dxn_iwalk():
    assert delta.dxn_iwalk[0] == 1


def test_set_dxn_jwalk():
    assert delta.dxn_jwalk[0] == 0


def test_set_dxn_dist():
    assert delta.dxn_dist[0] == 1


def test_set_walk_flat():
    assert delta.walk_flat[0] == 1


def test_kernel1():
    assert delta.kernel1[0, 0] == 1


def test_kernel2():
    assert delta.kernel2[0, 0] == 1

# Tests for other variables
delta.create_other_variables()


def test_init_Np_water():
    assert delta.init_Np_water == 10


def test_init_Np_sed():
    assert delta.init_Np_sed == 10


def test_dx():
    assert delta.dx == float(1)


def test_theta_sand():
    assert delta.theta_sand == 2


def test_theta_mud():
    assert delta.theta_mud == 1


def test_Nsmooth():
    assert delta.Nsmooth == 1


def test_U_dep_mud():
    assert delta.U_dep_mud == 0.3


def test_U_ero_sand():
    assert delta.U_ero_sand == 1.05


def test_U_ero_mud():
    assert delta.U_ero_mud == 1.5


def test_L0():
    assert delta.L0 == 1


def test_N0():
    assert delta.N0 == 3


def test_L():
    assert delta.L == 10


def test_W():
    assert delta.W == 10


def test_u_max():
    assert delta.u_max == 2.0


def test_C0():
    assert delta.C0 == 0.001


def test_dry_depth():
    assert delta.dry_depth == 0.1


def test_CTR():
    assert delta.CTR == 4


def test_gamma():
    assert delta.gamma == 0.001962


def test_V0():
    assert delta.V0 == 1


def test_Qw0():
    assert delta.Qw0 == 3.0


def test_qw0():
    assert delta.qw0 == 1.0


def test_Qp_water():
    assert delta.Qp_water == 3.0 / 10


def test_qw0():
    assert delta.qs0 == 0.001


def test_dVs():
    assert delta.dVs == 0.9


def test_Qs0():
    assert delta.Qs0 == 3.0 * 0.001


def test_Vp_sed():
    assert delta.Vp_sed == 0.9 / 10


def test_itmax():
    assert delta.itmax == 40


def test_size_indices():
    assert delta.size_indices == int(20)


def test_dt():
    assert delta.dt == 0.9 / 0.003


def test_omega_flow():
    assert delta.omega_flow == 0.9


def test_omega_flow_iter():
    assert delta.omega_flow_iter == 2.


def test_N_crossdiff():
    assert delta.N_crossdiff == int(1)


def test_lambda():
    assert delta._lambda == 1.


def test_diffusion_multiplier():
    assert delta.diffusion_multiplier == 15.0


# test definition of the model domain
delta.create_domain()


def test_x():
    assert delta.x[0][-1] == 9


def test_y():
    assert delta.y[0][-1] == 0


def test_cell_type():
    # wall type in corner
    assert delta.cell_type[0, 0] == -2


def test_eta():
    assert delta.eta[2, 2] == -1


def test_stage():
    assert delta.stage[2, 2] == 0.0


def test_depth():
    assert delta.depth[2, 2] == 1


def test_qx():
    # prescribe the qx at the inlet
    assert delta.qx[0, 4] == 1


def test_qy():
    assert delta.qy[0, 4] == 0


def test_qxn():
    assert delta.qxn[0, 0] == 0


def test_qyn():
    assert delta.qyn[0, 0] == 0


def test_qwn():
    assert delta.qwn[0, 0] == 0


def test_ux():
    assert delta.ux[0, 4] == 1


def test_uy():
    assert delta.uy[0, 4] == 0


def test_uw():
    assert delta.uw[0, 4] == 1


def test_qs():
    assert delta.qs[5, 5] == 0


def test_Vp_dep_sand():
    assert np.any(delta.Vp_dep_sand) == 0


def test_Vp_dep_mud():
    assert np.any(delta.Vp_dep_mud) == 0


def test_free_surf_flag():
    assert np.any(delta.free_surf_flag) == 0


def test_looped():
    assert np.any(delta.looped) == 0


def test_indices():
    assert np.any(delta.indices) == 0


def test_sfc_visit():
    assert np.any(delta.sfc_visit) == 0


def test_sfc_sum():
    assert np.any(delta.sfc_sum) == 0


def test_clim_eta():
    assert delta.clim_eta == (-2, 0.05)
