## unit tests for init_tools.py

import pytest

import sys, os
import numpy as np
sys.path.append(os.path.realpath(os.path.dirname(__file__)+"/.."))

from pyDeltaRCM import BmiDelta
from pyDeltaRCM import init_tools

# need to create a simple case of pydeltarcm object to test these functions
# will fix the random seed
np.random.seed(0)

# use test.yaml to create a small test case
delta = BmiDelta()
delta.initialize(os.getcwd() + '/tests/test.yaml')

# now that it is initiated can access the init_tools via the inherited object
# delta._delta.**init_tools_function**


# Tests for all of the constants
delta._delta.set_constants()

def test_set_constant_g():
    """
    check gravity
    """
    assert delta._delta.g == 9.81

def test_set_constant_distances():
    """
    check distances
    """
    assert delta._delta.distances[0,0] == np.sqrt(2)

def test_set_ivec():
    assert delta._delta.ivec[0,0] == -np.sqrt(0.5)

def test_set_jvec():
    assert delta._delta.jvec[0,0] == -np.sqrt(0.5)

def test_set_iwalk():
    assert delta._delta.iwalk[0,0] == -1

def test_set_jwalk():
    assert delta._delta.jwalk[0,0] == -1

def test_set_dxn_iwalk():
    assert delta._delta.dxn_iwalk[0] == 1

def test_set_dxn_jwalk():
    assert delta._delta.dxn_jwalk[0] == 0

def test_set_dxn_dist():
    assert delta._delta.dxn_dist[0] == 1

def test_set_walk_flat():
    assert delta._delta.walk_flat[0] == 1

def test_kernel1():
    assert delta._delta.kernel1[0,0] == 1

def test_kernel2():
    assert delta._delta.kernel2[0,0] == 1

# Tests for other variables
delta._delta.create_other_variables()

def test_init_Np_water():
    assert delta._delta.init_Np_water == 10

def test_init_Np_sed():
    assert delta._delta.init_Np_sed == 10

def test_dx():
    assert delta._delta.dx == float(1)

def test_theta_sand():
    assert delta._delta.theta_sand == 2

def test_theta_mud():
    assert delta._delta.theta_mud == 1

def test_Nsmooth():
    assert delta._delta.Nsmooth == 1

def test_U_dep_mud():
    assert delta._delta.U_dep_mud == 0.3

def test_U_ero_sand():
    assert delta._delta.U_ero_sand == 1.05

def test_U_ero_mud():
    assert delta._delta.U_ero_mud == 1.5

def test_L0():
    assert delta._delta.L0 == 1

def test_N0():
    assert delta._delta.N0 == 3

def test_L():
    assert delta._delta.L == 10

def test_W():
    assert delta._delta.W == 10

def test_u_max():
    assert delta._delta.u_max == 2.0

def test_C0():
    assert delta._delta.C0 == 0.001

def test_dry_depth():
    assert delta._delta.dry_depth == 0.1

def test_CTR():
    assert delta._delta.CTR == 4

def test_gamma():
    assert delta._delta.gamma == 0.001962

def test_V0():
    assert delta._delta.V0 == 1

def test_Qw0():
    assert delta._delta.Qw0 == 3.0

def test_qw0():
    assert delta._delta.qw0 == 1.0

def test_Qp_water():
    assert delta._delta.Qp_water == 3.0 / 10

def test_qw0():
    assert delta._delta.qs0 == 0.001

def test_dVs():
    assert delta._delta.dVs == 0.9

def test_Qs0():
    assert delta._delta.Qs0 == 3.0 * 0.001

def test_Vp_sed():
    assert delta._delta.Vp_sed == 0.9 / 10

def test_itmax():
    assert delta._delta.itmax == 40

def test_size_indices():
    assert delta._delta.size_indices == int(20)

def test_dt():
    assert delta._delta.dt == 0.9 / 0.003

def test_omega_flow():
    assert delta._delta.omega_flow == 0.9

def test_omega_flow_iter():
    assert delta._delta.omega_flow_iter == 2.

def test_N_crossdiff():
    assert delta._delta.N_crossdiff == int(1)

def test_lambda():
    assert delta._delta._lambda == 1.

def test_diffusion_multiplier():
    assert delta._delta.diffusion_multiplier == 15.0


# test definition of the model domain
delta._delta.create_domain()

def test_x():
    assert delta._delta.x[0][-1] == 9

def test_y():
    assert delta._delta.y[0][-1] == 0

def test_cell_type():
    # wall type in corner
    assert delta._delta.cell_type[0,0] == -2

def test_eta():
    assert delta._delta.eta[2,2] == -1

def test_stage():
    assert delta._delta.stage[2,2] == 0.0

def test_depth():
    assert delta._delta.depth[2,2] == 1

def test_qx():
    # prescribe the qx at the inlet
    assert delta._delta.qx[0,4] == 1

def test_qy():
    assert delta._delta.qy[0,4] == 0

def test_qxn():
    assert delta._delta.qxn[0,0] == 0

def test_qyn():
    assert delta._delta.qyn[0,0] == 0

def test_qwn():
    assert delta._delta.qwn[0,0] == 0

def test_ux():
    assert delta._delta.ux[0,4] == 1

def test_uy():
    assert delta._delta.uy[0,4] == 0

def test_uw():
    assert delta._delta.uw[0,4] == 1

def test_qs():
    assert delta._delta.qs[5,5] == 0

def test_Vp_dep_sand():
    assert np.any(delta._delta.Vp_dep_sand) == 0

def test_Vp_dep_mud():
    assert np.any(delta._delta.Vp_dep_mud) == 0

def test_free_surf_flag():
    assert np.any(delta._delta.free_surf_flag) == 0

def test_looped():
    assert np.any(delta._delta.looped) == 0

def test_indices():
    assert np.any(delta._delta.indices) == 0

def test_sfc_visit():
    assert np.any(delta._delta.sfc_visit) == 0

def test_sfc_sum():
    assert np.any(delta._delta.sfc_sum) == 0

def test_clim_eta():
    assert delta._delta.clim_eta == (-2, 0.05)
