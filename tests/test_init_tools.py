# unit tests for init_tools.py

import pytest

import sys
import os
import numpy as np

from pyDeltaRCM.model import DeltaModel
from pyDeltaRCM import init_tools

from utilities import test_DeltaModel
import utilities


def test_inlet_size_specified(tmp_path):
    file_name = 'user_parameters.yaml'
    p, f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(f, 'Length', 4000.)
    utilities.write_parameter_to_file(f, 'Width', 8000.)
    utilities.write_parameter_to_file(f, 'dx', 20)
    utilities.write_parameter_to_file(f, 'N0_meters', 150)
    utilities.write_parameter_to_file(f, 'L0_meters', 200)
    f.close()
    delta = DeltaModel(input_file=p)
    assert delta.N0 == 8
    assert delta.L0 == 10


def test_inlet_size_set_to_one_fourth_domain(tmp_path):
    file_name = 'user_parameters.yaml'
    p, f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(f, 'Length', 4000.)
    utilities.write_parameter_to_file(f, 'Width', 8000.)
    utilities.write_parameter_to_file(f, 'dx', 20)
    utilities.write_parameter_to_file(f, 'N0_meters', 5500)
    utilities.write_parameter_to_file(f, 'L0_meters', 3300)
    f.close()
    delta = DeltaModel(input_file=p)
    assert delta.N0 == 100
    assert delta.L0 == 50


# tests for attrs set during yaml parsing

def test_set_verbose(test_DeltaModel):
    assert test_DeltaModel.verbose == 0


def test_set_seed_zero(test_DeltaModel):
    assert test_DeltaModel.seed == 0


# tests for all of the constants

def test_set_constant_g(test_DeltaModel):
    """
    check gravity
    """
    assert test_DeltaModel.g == 9.81


def test_set_constant_distances(test_DeltaModel):
    """
    check distances
    """
    assert test_DeltaModel.distances[0, 0] == np.sqrt(2)


def test_set_ivec(test_DeltaModel):
    assert test_DeltaModel.ivec[0, 0] == -np.sqrt(0.5)


def test_set_jvec(test_DeltaModel):
    assert test_DeltaModel.jvec[0, 0] == -np.sqrt(0.5)


def test_set_iwalk(test_DeltaModel):
    assert test_DeltaModel.iwalk[0, 0] == -1


def test_set_jwalk(test_DeltaModel):
    assert test_DeltaModel.jwalk[0, 0] == -1


def test_set_dxn_iwalk(test_DeltaModel):
    assert test_DeltaModel.dxn_iwalk[0] == 1


def test_set_dxn_jwalk(test_DeltaModel):
    assert test_DeltaModel.dxn_jwalk[0] == 0


def test_set_dxn_dist(test_DeltaModel):
    assert test_DeltaModel.dxn_dist[0] == 1


def test_set_walk_flat(test_DeltaModel):
    assert test_DeltaModel.walk_flat[0] == 1


def test_kernel1(test_DeltaModel):
    assert test_DeltaModel.kernel1[0, 0] == 1


def test_kernel2(test_DeltaModel):
    assert test_DeltaModel.kernel2[0, 0] == 1


# Tests for other variables
def test_init_Np_water(tmp_path):
    p = utilities.yaml_from_dict(tmp_path, 'input.yaml', {'Np_water': 50})
    _delta = DeltaModel(input_file=p)
    assert _delta.init_Np_water == 50


def test_init_Np_sed(tmp_path):
    p = utilities.yaml_from_dict(tmp_path, 'input.yaml', {'Np_sed': 60})
    _delta = DeltaModel(input_file=p)
    assert _delta.init_Np_sed == 60


def test_dx(tmp_path):
    p = utilities.yaml_from_dict(tmp_path, 'input.yaml', {'dx': 20})
    _delta = DeltaModel(input_file=p)
    assert _delta.dx == 20


def test_theta_sand(tmp_path):
    p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                 {'coeff_theta_sand': 1.4,
                                  'theta_water': 1.2})
    _delta = DeltaModel(input_file=p)
    assert _delta.theta_sand == 1.68


def test_theta_mud(tmp_path):
    p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                 {'coeff_theta_mud': 0.8,
                                  'theta_water': 1.3})
    _delta = DeltaModel(input_file=p)
    assert _delta.theta_mud == 1.04


def test_Nsmooth(tmp_path):
    p = utilities.yaml_from_dict(tmp_path, 'input.yaml', {'Nsmooth': 6})
    _delta = DeltaModel(input_file=p)
    assert _delta.Nsmooth == 6


def test_U_dep_mud(tmp_path):
    p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                 {'coeff_U_dep_mud': 0.4325,
                                  'u0': 2.2})
    _delta = DeltaModel(input_file=p)
    assert _delta.U_dep_mud == 0.9515


def test_U_ero_sand(tmp_path):
    p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                 {'coeff_U_ero_sand': 1.23,
                                  'u0': 2.2})
    _delta = DeltaModel(input_file=p)
    assert _delta.U_ero_sand == 2.706


def test_U_ero_mud(tmp_path):
    p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                 {'coeff_U_ero_mud': 1.67,
                                  'u0': 2.2})
    _delta = DeltaModel(input_file=p)
    assert _delta.U_ero_mud == 3.674


def test_L0(tmp_path):
    p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                 {'L0_meters': 100,
                                  'Length': 6000,
                                  'dx': 5})
    _delta = DeltaModel(input_file=p)
    assert _delta.L0 == 20


def test_N0(tmp_path):
    p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                 {'N0_meters': 500,
                                  'Width': 6000,
                                  'dx': 5})
    _delta = DeltaModel(input_file=p)
    assert _delta.N0 == 100


def test_L(tmp_path):
    p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                 {'Length': 1600,
                                  'dx': 20})
    _delta = DeltaModel(input_file=p)
    assert _delta.L == 80


def test_W(tmp_path):
    p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                 {'Width': 1200,
                                  'dx': 20})
    _delta = DeltaModel(input_file=p)
    assert _delta.W == 60


def test_SLR(tmp_path):
    p = utilities.yaml_from_dict(tmp_path, 'input.yaml', {'SLR': 0.01})
    _delta = DeltaModel(input_file=p)
    assert _delta.SLR == 0.01


def test_u_max(test_DeltaModel):
    assert test_DeltaModel.u_max == 2.0


def test_C0(test_DeltaModel):
    assert test_DeltaModel.C0 == 0.001


def test_dry_depth(test_DeltaModel):
    assert test_DeltaModel.dry_depth == 0.1


def test_CTR(test_DeltaModel):
    assert test_DeltaModel.CTR == 4


def test_gamma(test_DeltaModel):
    assert test_DeltaModel.gamma == 0.001962


def test_V0(test_DeltaModel):
    assert test_DeltaModel.V0 == 1


def test_Qw0(test_DeltaModel):
    assert test_DeltaModel.Qw0 == 3.0


def test_qw0(test_DeltaModel):
    assert test_DeltaModel.qw0 == 1.0


def test_Qp_water(test_DeltaModel):
    assert test_DeltaModel.Qp_water == 3.0 / 10


def test_qw0(test_DeltaModel):
    assert test_DeltaModel.qs0 == 0.001


def test_dVs(test_DeltaModel):
    assert test_DeltaModel.dVs == 0.9


def test_Qs0(test_DeltaModel):
    assert test_DeltaModel.Qs0 == 3.0 * 0.001


def test_Vp_sed(test_DeltaModel):
    assert test_DeltaModel.Vp_sed == 0.9 / 10


def test_itmax(test_DeltaModel):
    assert test_DeltaModel.itmax == 40


def test_size_indices(test_DeltaModel):
    assert test_DeltaModel.size_indices == int(20)


def test_dt(test_DeltaModel):
    assert test_DeltaModel.dt == 0.9 / 0.003


def test_omega_flow(tmp_path):
    p = utilities.yaml_from_dict(tmp_path, 'input.yaml', {'omega_flow': 0.8})
    _delta = DeltaModel(input_file=p)
    assert _delta.omega_flow == 0.8


def test_omega_flow_iter(test_DeltaModel):
    assert test_DeltaModel.omega_flow_iter == 2.


def test_N_crossdiff(test_DeltaModel):
    assert test_DeltaModel.N_crossdiff == int(1)


def test_lambda(test_DeltaModel):
    assert test_DeltaModel._lambda == 1.


def test_diffusion_multiplier(test_DeltaModel):
    assert test_DeltaModel.diffusion_multiplier == 15.0


# test definition of the model domain

def test_x(test_DeltaModel):
    assert test_DeltaModel.x[0][-1] == 9


def test_y(test_DeltaModel):
    assert test_DeltaModel.y[0][-1] == 0


def test_cell_type(test_DeltaModel):
    # wall type in corner
    assert test_DeltaModel.cell_type[0, 0] == -2


def test_eta(test_DeltaModel):
    assert test_DeltaModel.eta[2, 2] == -1


def test_stage(test_DeltaModel):
    assert test_DeltaModel.stage[2, 2] == 0.0


def test_depth(test_DeltaModel):
    assert test_DeltaModel.depth[2, 2] == 1


def test_qx(test_DeltaModel):
    # prescribe the qx at the inlet
    assert test_DeltaModel.qx[0, 4] == 1


def test_qy(test_DeltaModel):
    assert test_DeltaModel.qy[0, 4] == 0


def test_qxn(test_DeltaModel):
    assert test_DeltaModel.qxn[0, 0] == 0


def test_qyn(test_DeltaModel):
    assert test_DeltaModel.qyn[0, 0] == 0


def test_qwn(test_DeltaModel):
    assert test_DeltaModel.qwn[0, 0] == 0


def test_ux(test_DeltaModel):
    assert test_DeltaModel.ux[0, 4] == 1


def test_uy(test_DeltaModel):
    assert test_DeltaModel.uy[0, 4] == 0


def test_uw(test_DeltaModel):
    assert test_DeltaModel.uw[0, 4] == 1


def test_qs(test_DeltaModel):
    assert test_DeltaModel.qs[5, 5] == 0


def test_Vp_dep_sand(test_DeltaModel):
    assert np.any(test_DeltaModel.Vp_dep_sand) == 0


def test_Vp_dep_mud(test_DeltaModel):
    assert np.any(test_DeltaModel.Vp_dep_mud) == 0


def test_free_surf_flag(test_DeltaModel):
    assert np.any(test_DeltaModel.free_surf_flag) == 0


def test_looped(test_DeltaModel):
    assert np.any(test_DeltaModel.looped) == 0


def test_indices(test_DeltaModel):
    assert np.any(test_DeltaModel.indices) == 0


def test_sfc_visit(test_DeltaModel):
    assert np.any(test_DeltaModel.sfc_visit) == 0


def test_sfc_sum(test_DeltaModel):
    assert np.any(test_DeltaModel.sfc_sum) == 0
