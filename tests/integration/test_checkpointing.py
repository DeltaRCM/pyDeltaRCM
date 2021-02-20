# unit tests for consistent model outputs

import pytest

import os
import platform
import numpy as np
from netCDF4 import Dataset

import unittest.mock as mock

from pyDeltaRCM import DeltaModel
from pyDeltaRCM import preprocessor

from .. import utilities


# @mock.patch(
#     'pyDeltaRCM.iteration_tools.iteration_tools.run_one_timestep',
#     new=utilities.FastIteratingDeltaModel.run_one_timestep)
def test_simple_checkpoint(tmp_path):
    """Test checkpoint vs a base run, and against another checkpoint run."""
    # define a yaml for the longer model run
    file_name = 'base_run.yaml'
    base_p, base_f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(base_f, 'Length', 10.0)
    utilities.write_parameter_to_file(base_f, 'Width', 10.0)
    utilities.write_parameter_to_file(base_f, 'seed', 0)
    utilities.write_parameter_to_file(base_f, 'dx', 1.0)
    utilities.write_parameter_to_file(base_f, 'L0_meters', 1.0)
    utilities.write_parameter_to_file(base_f, 'Np_water', 10)
    utilities.write_parameter_to_file(base_f, 'u0', 1.0)
    utilities.write_parameter_to_file(base_f, 'N0_meters', 2.0)
    utilities.write_parameter_to_file(base_f, 'h0', 1.0)
    utilities.write_parameter_to_file(base_f, 'SLR', 0.001)
    utilities.write_parameter_to_file(base_f, 'Np_sed', 10)
    utilities.write_parameter_to_file(base_f, 'save_dt', 50)
    utilities.write_parameter_to_file(base_f, 'save_strata', True)
    utilities.write_parameter_to_file(base_f, 'save_checkpoint', True)
    utilities.write_parameter_to_file(base_f, 'checkpoint_dt', 600)
    base_f.close()
    longModel = DeltaModel(input_file=base_p)

    for _ in range(0, 3):
        longModel.update()
    longModel.finalize()

    # try defining a new model but plan to load checkpoint from longModel
    file_name = 'base_run.yaml'
    base_p, base_f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(base_f, 'Length', 10.0)
    utilities.write_parameter_to_file(base_f, 'Width', 10.0)
    utilities.write_parameter_to_file(base_f, 'seed', 0)
    utilities.write_parameter_to_file(base_f, 'dx', 1.0)
    utilities.write_parameter_to_file(base_f, 'L0_meters', 1.0)
    utilities.write_parameter_to_file(base_f, 'Np_water', 10)
    utilities.write_parameter_to_file(base_f, 'u0', 1.0)
    utilities.write_parameter_to_file(base_f, 'N0_meters', 2.0)
    utilities.write_parameter_to_file(base_f, 'h0', 1.0)
    utilities.write_parameter_to_file(base_f, 'SLR', 0.001)
    utilities.write_parameter_to_file(base_f, 'Np_sed', 10)
    utilities.write_parameter_to_file(base_f, 'save_dt', 50)
    utilities.write_parameter_to_file(base_f, 'save_strata', True)
    utilities.write_parameter_to_file(base_f, 'save_checkpoint', True)
    utilities.write_parameter_to_file(base_f, 'checkpoint_dt', 600)
    utilities.write_parameter_to_file(base_f, 'resume_checkpoint', True)
    base_f.close()
    resumeModel = DeltaModel(input_file=base_p)

    # advance it one step to catch up to longModel
    resumeModel.update()
    resumeModel.finalize()

    # the longModel and resumeModel should match
    assert longModel.time == resumeModel.time
    assert np.all(longModel.uw == resumeModel.uw)
    assert np.all(longModel.ux == resumeModel.ux)
    assert np.all(longModel.uy == resumeModel.uy)
    assert np.all(longModel.depth == resumeModel.depth)
    assert np.all(longModel.stage == resumeModel.stage)
    assert np.all(longModel.strata_eta.todense() ==
                  resumeModel.strata_eta.todense())
    # assert np.all(longModel.strata_sand_frac.todense() ==
    #               resumeModel.strata_sand_frac.todense())

    # define another model that loads the checkpoint
    file_name = 'base_run.yaml'
    base_p, base_f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(base_f, 'Length', 10.0)
    utilities.write_parameter_to_file(base_f, 'Width', 10.0)
    utilities.write_parameter_to_file(base_f, 'seed', 0)
    utilities.write_parameter_to_file(base_f, 'dx', 1.0)
    utilities.write_parameter_to_file(base_f, 'L0_meters', 1.0)
    utilities.write_parameter_to_file(base_f, 'Np_water', 10)
    utilities.write_parameter_to_file(base_f, 'u0', 1.0)
    utilities.write_parameter_to_file(base_f, 'N0_meters', 2.0)
    utilities.write_parameter_to_file(base_f, 'h0', 1.0)
    utilities.write_parameter_to_file(base_f, 'SLR', 0.001)
    utilities.write_parameter_to_file(base_f, 'Np_sed', 10)
    utilities.write_parameter_to_file(base_f, 'save_dt', 50)
    utilities.write_parameter_to_file(base_f, 'save_strata', True)
    utilities.write_parameter_to_file(base_f, 'save_checkpoint', True)
    utilities.write_parameter_to_file(base_f, 'checkpoint_dt', 600)
    utilities.write_parameter_to_file(base_f, 'resume_checkpoint', True)
    base_f.close()
    resumeModel2 = DeltaModel(input_file=base_p)

    # advance it one step to catch up to resumeModel
    resumeModel2.update()
    resumeModel2.finalize()

    # the two models that resumed from the checkpoint should be the same
    assert resumeModel2.time == resumeModel.time
    assert np.all(resumeModel2.uw == resumeModel.uw)
    assert np.all(resumeModel2.ux == resumeModel.ux)
    assert np.all(resumeModel2.uy == resumeModel.uy)
    assert np.all(resumeModel2.depth == resumeModel.depth)
    assert np.all(resumeModel2.stage == resumeModel.stage)
    assert np.all(resumeModel2.strata_eta.todense() ==
                  resumeModel.strata_eta.todense())
    assert np.all(resumeModel2.strata_sand_frac.todense() ==
                  resumeModel.strata_sand_frac.todense())


def test_longer_checkpoint(tmp_path):
    """Test checkpoint with longer run."""
    # define a yaml for the longer model run
    file_name = 'base_run.yaml'
    base_p, base_f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(base_f, 'Length', 10.0)
    utilities.write_parameter_to_file(base_f, 'Width', 10.0)
    utilities.write_parameter_to_file(base_f, 'seed', 0)
    utilities.write_parameter_to_file(base_f, 'dx', 1.0)
    utilities.write_parameter_to_file(base_f, 'L0_meters', 1.0)
    utilities.write_parameter_to_file(base_f, 'Np_water', 10)
    utilities.write_parameter_to_file(base_f, 'u0', 1.0)
    utilities.write_parameter_to_file(base_f, 'N0_meters', 2.0)
    utilities.write_parameter_to_file(base_f, 'h0', 1.0)
    utilities.write_parameter_to_file(base_f, 'SLR', 0.001)
    utilities.write_parameter_to_file(base_f, 'Np_sed', 10)
    utilities.write_parameter_to_file(base_f, 'save_dt', 50)
    utilities.write_parameter_to_file(base_f, 'save_strata', True)
    utilities.write_parameter_to_file(base_f, 'save_checkpoint', True)
    utilities.write_parameter_to_file(base_f, 'checkpoint_dt', 1200)
    base_f.close()
    longModel = DeltaModel(input_file=base_p)

    for _ in range(0, 7):
        longModel.update()
    longModel.finalize()

    # try defining a new model but plan to load checkpoint from longModel
    file_name = 'base_run.yaml'
    base_p, base_f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(base_f, 'Length', 10.0)
    utilities.write_parameter_to_file(base_f, 'Width', 10.0)
    utilities.write_parameter_to_file(base_f, 'seed', 0)
    utilities.write_parameter_to_file(base_f, 'dx', 1.0)
    utilities.write_parameter_to_file(base_f, 'L0_meters', 1.0)
    utilities.write_parameter_to_file(base_f, 'Np_water', 10)
    utilities.write_parameter_to_file(base_f, 'u0', 1.0)
    utilities.write_parameter_to_file(base_f, 'N0_meters', 2.0)
    utilities.write_parameter_to_file(base_f, 'h0', 1.0)
    utilities.write_parameter_to_file(base_f, 'SLR', 0.001)
    utilities.write_parameter_to_file(base_f, 'Np_sed', 10)
    utilities.write_parameter_to_file(base_f, 'save_dt', 50)
    utilities.write_parameter_to_file(base_f, 'save_strata', True)
    utilities.write_parameter_to_file(base_f, 'save_checkpoint', True)
    utilities.write_parameter_to_file(base_f, 'checkpoint_dt', 1200)
    utilities.write_parameter_to_file(base_f, 'resume_checkpoint', True)
    base_f.close()
    resumeModel = DeltaModel(input_file=base_p)

    # advance it three steps to catch up to longModel
    for _ in range(0, 3):
        resumeModel.update()
    resumeModel.finalize()

    # the longModel and resumeModel should match
    assert longModel.time == resumeModel.time
    assert np.all(longModel.uw == resumeModel.uw)
    assert np.all(longModel.ux == resumeModel.ux)
    assert np.all(longModel.uy == resumeModel.uy)
    assert np.all(longModel.depth == resumeModel.depth)
    assert np.all(longModel.stage == resumeModel.stage)
    assert np.all(longModel.strata_eta.todense() ==
                  resumeModel.strata_eta.todense())
    assert pytest.approx(longModel.strata_sand_frac.todense() ==
                         resumeModel.strata_sand_frac.todense())


def test_checkpoint_nc(tmp_path):
    """Test the netCDF that is written to by the checkpointing."""
    # define a yaml for the base model run
    file_name = 'base_run.yaml'
    base_p, base_f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(base_f, 'Length', 10.0)
    utilities.write_parameter_to_file(base_f, 'Width', 10.0)
    utilities.write_parameter_to_file(base_f, 'seed', 0)
    utilities.write_parameter_to_file(base_f, 'dx', 1.0)
    utilities.write_parameter_to_file(base_f, 'L0_meters', 1.0)
    utilities.write_parameter_to_file(base_f, 'Np_water', 10)
    utilities.write_parameter_to_file(base_f, 'u0', 1.0)
    utilities.write_parameter_to_file(base_f, 'N0_meters', 2.0)
    utilities.write_parameter_to_file(base_f, 'h0', 1.0)
    utilities.write_parameter_to_file(base_f, 'SLR', 0.001)
    utilities.write_parameter_to_file(base_f, 'Np_sed', 10)
    utilities.write_parameter_to_file(base_f, 'save_dt', 50)
    utilities.write_parameter_to_file(base_f, 'save_strata', True)
    utilities.write_parameter_to_file(base_f, 'save_eta_grids', True)
    utilities.write_parameter_to_file(base_f, 'save_depth_grids', True)
    utilities.write_parameter_to_file(base_f, 'save_discharge_grids', True)
    utilities.write_parameter_to_file(base_f, 'save_checkpoint', True)
    utilities.write_parameter_to_file(base_f, 'checkpoint_dt', 600)
    utilities.write_parameter_to_file(base_f, 'out_dir', tmp_path / 'test')
    base_f.close()
    baseModel = DeltaModel(input_file=base_p)

    for _ in range(0, 4):
        baseModel.update()
    baseModel.finalize()

    assert baseModel.time == 1200.0  # dt=300 * 4 = 1200

    # try defining a new model but plan to load checkpoint from baseModel
    file_name = 'base_run.yaml'
    base_p, base_f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(base_f, 'Length', 10.0)
    utilities.write_parameter_to_file(base_f, 'Width', 10.0)
    utilities.write_parameter_to_file(base_f, 'seed', 0)
    utilities.write_parameter_to_file(base_f, 'dx', 1.0)
    utilities.write_parameter_to_file(base_f, 'L0_meters', 1.0)
    utilities.write_parameter_to_file(base_f, 'Np_water', 10)
    utilities.write_parameter_to_file(base_f, 'u0', 1.0)
    utilities.write_parameter_to_file(base_f, 'N0_meters', 2.0)
    utilities.write_parameter_to_file(base_f, 'h0', 1.0)
    utilities.write_parameter_to_file(base_f, 'SLR', 0.001)
    utilities.write_parameter_to_file(base_f, 'Np_sed', 10)
    utilities.write_parameter_to_file(base_f, 'save_dt', 50)
    utilities.write_parameter_to_file(base_f, 'save_strata', True)
    utilities.write_parameter_to_file(base_f, 'save_eta_grids', True)
    utilities.write_parameter_to_file(base_f, 'save_depth_grids', True)
    utilities.write_parameter_to_file(base_f, 'save_discharge_grids', True)
    utilities.write_parameter_to_file(base_f, 'save_checkpoint', False)
    utilities.write_parameter_to_file(base_f, 'resume_checkpoint', True)
    utilities.write_parameter_to_file(base_f, 'out_dir', tmp_path / 'test')
    base_f.close()
    resumeModel = DeltaModel(input_file=base_p)

    assert resumeModel.time == baseModel.time  # should be same when resumed

    # advance it six steps
    for _ in range(0, 6):
        resumeModel.update()
    resumeModel.finalize()

    # assert that ouput netCDF4 exists
    exp_path_nc = os.path.join(tmp_path / 'test', 'pyDeltaRCM_output.nc')
    assert os.path.isfile(exp_path_nc)

    # load it into memory and check values in the netCDF4
    output = Dataset(exp_path_nc, 'r', allow_pickle=True)
    out_vars = output.variables.keys()
    # check that expected variables are in the file
    assert 'x' in out_vars
    assert 'y' in out_vars
    assert 'time' in out_vars
    assert 'eta' in out_vars
    assert 'depth' in out_vars
    assert 'discharge' in out_vars
    # check attributes of variables
    assert output['time'][0].tolist() == 0.0
    assert output['time'][-1].tolist() == 3000.0
    assert output['eta'][0].shape == (10, 10)
    assert output['eta'][-1].shape == (10, 10)
    assert output['depth'][-1].shape == (10, 10)
    assert output['discharge'][-1].shape == (10, 10)
    # check time
    assert baseModel.dt == 300.0  # base model timestep
    assert baseModel.time == 1200.0  # ran 4 steps, so 300*4=1200
    assert resumeModel.dt == 300.0  # resume model timestep size
    assert resumeModel.time == 3000.0  # ran 6 steps on top of base model
    # should be 1200 + (6*300) = 1200 + 1800 = 3000

    # checkpoint interval aligns w/ timestep dt so these should match
    assert output['time'][-1].tolist() == resumeModel.time


def test_checkpoint_diff_dt(tmp_path):
    """Test when checkpoint_dt does not match dt or save_dt."""
    # define a yaml for the base model run
    file_name = 'base_run.yaml'
    base_p, base_f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(base_f, 'Length', 10.0)
    utilities.write_parameter_to_file(base_f, 'Width', 10.0)
    utilities.write_parameter_to_file(base_f, 'seed', 0)
    utilities.write_parameter_to_file(base_f, 'dx', 1.0)
    utilities.write_parameter_to_file(base_f, 'L0_meters', 1.0)
    utilities.write_parameter_to_file(base_f, 'Np_water', 10)
    utilities.write_parameter_to_file(base_f, 'u0', 1.0)
    utilities.write_parameter_to_file(base_f, 'N0_meters', 2.0)
    utilities.write_parameter_to_file(base_f, 'h0', 1.0)
    utilities.write_parameter_to_file(base_f, 'SLR', 0.001)
    utilities.write_parameter_to_file(base_f, 'Np_sed', 10)
    utilities.write_parameter_to_file(base_f, 'save_dt', 50)
    utilities.write_parameter_to_file(base_f, 'save_strata', True)
    utilities.write_parameter_to_file(base_f, 'save_eta_grids', True)
    utilities.write_parameter_to_file(base_f, 'save_depth_grids', True)
    utilities.write_parameter_to_file(base_f, 'save_discharge_grids', True)
    utilities.write_parameter_to_file(base_f, 'save_checkpoint', True)
    utilities.write_parameter_to_file(base_f, 'checkpoint_dt', 500)
    utilities.write_parameter_to_file(base_f, 'out_dir', tmp_path / 'test')
    base_f.close()
    baseModel = DeltaModel(input_file=base_p)

    for _ in range(0, 2):
        baseModel.update()
    baseModel.finalize()

    assert baseModel.time == 600.0  # dt=300 * 2 = 600

    # try defining a new model but plan to load checkpoint from baseModel
    file_name = 'base_run.yaml'
    base_p, base_f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(base_f, 'Length', 10.0)
    utilities.write_parameter_to_file(base_f, 'Width', 10.0)
    utilities.write_parameter_to_file(base_f, 'seed', 0)
    utilities.write_parameter_to_file(base_f, 'dx', 1.0)
    utilities.write_parameter_to_file(base_f, 'L0_meters', 1.0)
    utilities.write_parameter_to_file(base_f, 'Np_water', 10)
    utilities.write_parameter_to_file(base_f, 'u0', 1.0)
    utilities.write_parameter_to_file(base_f, 'N0_meters', 2.0)
    utilities.write_parameter_to_file(base_f, 'h0', 1.0)
    utilities.write_parameter_to_file(base_f, 'SLR', 0.001)
    utilities.write_parameter_to_file(base_f, 'Np_sed', 10)
    utilities.write_parameter_to_file(base_f, 'save_dt', 50)
    utilities.write_parameter_to_file(base_f, 'save_strata', True)
    utilities.write_parameter_to_file(base_f, 'save_eta_grids', True)
    utilities.write_parameter_to_file(base_f, 'save_depth_grids', True)
    utilities.write_parameter_to_file(base_f, 'save_discharge_grids', True)
    utilities.write_parameter_to_file(base_f, 'save_checkpoint', False)
    utilities.write_parameter_to_file(base_f, 'resume_checkpoint', True)
    utilities.write_parameter_to_file(base_f, 'out_dir', tmp_path / 'test')
    base_f.close()
    resumeModel = DeltaModel(input_file=base_p)

    assert resumeModel.time == 600.0  # checkpoint dt was 500 but smaller than
    # saving dt which was 300, so 300*2=600 should be time we resume from
    assert resumeModel.time == baseModel.time  # should be same when resumed

    # advance it two more steps
    for _ in range(0, 2):
        resumeModel.update()
    resumeModel.finalize()

    # assert that ouput netCDF4 exists
    exp_path_nc = os.path.join(tmp_path / 'test', 'pyDeltaRCM_output.nc')
    assert os.path.isfile(exp_path_nc)

    # load it into memory and check values in the netCDF4
    output = Dataset(exp_path_nc, 'r', allow_pickle=True)
    out_vars = output.variables.keys()
    # check that expected variables are in the file
    assert 'x' in out_vars
    assert 'y' in out_vars
    assert 'time' in out_vars
    assert 'eta' in out_vars
    assert 'depth' in out_vars
    assert 'discharge' in out_vars
    # check attributes of variables
    assert output['time'][0].tolist() == 0.0
    assert output['time'][-1].tolist() == 1200.0
    assert output['eta'][0].shape == (10, 10)
    assert output['eta'][-1].shape == (10, 10)
    assert output['depth'][-1].shape == (10, 10)
    assert output['discharge'][-1].shape == (10, 10)
    # check time
    assert baseModel.dt == 300.0  # base model timestep
    assert baseModel.time == 600.0  # ran 4 steps, so 300*2=600
    assert resumeModel.dt == 300.0  # resume model timestep size
    assert resumeModel.time == 1200.0  # ran 2 steps on top of base model
    # should be 600 + (2*300) = 1200
    assert len(output['time'][:].tolist()) == 5  # 0-300-600-900-1200
    # checkpoint interval aligns w/ timestep dt so these should match
    assert output['time'][-1].tolist() == resumeModel.time


def test_multi_checkpoints(tmp_path):
    """Test using checkpoints multiple times for a given model run."""
    # define a yaml for the base model run
    file_name = 'base_run.yaml'
    base_p, base_f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(base_f, 'Length', 10.0)
    utilities.write_parameter_to_file(base_f, 'Width', 10.0)
    utilities.write_parameter_to_file(base_f, 'seed', 0)
    utilities.write_parameter_to_file(base_f, 'dx', 1.0)
    utilities.write_parameter_to_file(base_f, 'L0_meters', 1.0)
    utilities.write_parameter_to_file(base_f, 'Np_water', 10)
    utilities.write_parameter_to_file(base_f, 'u0', 1.0)
    utilities.write_parameter_to_file(base_f, 'N0_meters', 2.0)
    utilities.write_parameter_to_file(base_f, 'h0', 1.0)
    utilities.write_parameter_to_file(base_f, 'SLR', 0.001)
    utilities.write_parameter_to_file(base_f, 'Np_sed', 10)
    utilities.write_parameter_to_file(base_f, 'save_dt', 50)
    utilities.write_parameter_to_file(base_f, 'save_strata', True)
    utilities.write_parameter_to_file(base_f, 'save_eta_grids', True)
    utilities.write_parameter_to_file(base_f, 'save_depth_grids', True)
    utilities.write_parameter_to_file(base_f, 'save_discharge_grids', True)
    utilities.write_parameter_to_file(base_f, 'save_checkpoint', True)
    utilities.write_parameter_to_file(base_f, 'checkpoint_dt', 600)
    utilities.write_parameter_to_file(base_f, 'out_dir', tmp_path / 'test')
    base_f.close()
    baseModel = DeltaModel(input_file=base_p)

    # run base for 2 timesteps
    for _ in range(0, 2):
        baseModel.update()
    baseModel.finalize()

    assert baseModel.time == 600.0  # dt=300 * 2 = 600

    # try defining a new model but plan to load checkpoint from baseModel
    file_name = 'base_run.yaml'
    base_p, base_f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(base_f, 'Length', 10.0)
    utilities.write_parameter_to_file(base_f, 'Width', 10.0)
    utilities.write_parameter_to_file(base_f, 'seed', 0)
    utilities.write_parameter_to_file(base_f, 'dx', 1.0)
    utilities.write_parameter_to_file(base_f, 'L0_meters', 1.0)
    utilities.write_parameter_to_file(base_f, 'Np_water', 10)
    utilities.write_parameter_to_file(base_f, 'u0', 1.0)
    utilities.write_parameter_to_file(base_f, 'N0_meters', 2.0)
    utilities.write_parameter_to_file(base_f, 'h0', 1.0)
    utilities.write_parameter_to_file(base_f, 'SLR', 0.001)
    utilities.write_parameter_to_file(base_f, 'Np_sed', 10)
    utilities.write_parameter_to_file(base_f, 'save_dt', 50)
    utilities.write_parameter_to_file(base_f, 'save_strata', True)
    utilities.write_parameter_to_file(base_f, 'save_eta_grids', True)
    utilities.write_parameter_to_file(base_f, 'save_depth_grids', True)
    utilities.write_parameter_to_file(base_f, 'save_discharge_grids', True)
    utilities.write_parameter_to_file(base_f, 'save_checkpoint', True)
    utilities.write_parameter_to_file(base_f, 'resume_checkpoint', True)
    utilities.write_parameter_to_file(base_f, 'checkpoint_dt', 600)
    utilities.write_parameter_to_file(base_f, 'out_dir', tmp_path / 'test')
    base_f.close()
    resumeModel = DeltaModel(input_file=base_p)

    assert resumeModel.time == 600.0  # checkpoint dt was 500 but smaller than
    # saving dt which was 300, so 300*2=600 should be time we resume from
    assert resumeModel.time == baseModel.time  # should be same when resumed

    # advance it two more steps
    for _ in range(0, 2):
        resumeModel.update()
    resumeModel.finalize()

    # create another resume model
    resumeModel02 = DeltaModel(input_file=base_p)

    assert resumeModel02.time == resumeModel.time  # should be same
    assert resumeModel02.time == 1200.0  # 300*4 = 1200

    # step it twice
    for _ in range(0, 2):
        resumeModel02.update()
    resumeModel02.finalize()

    # assert that ouput netCDF4 exists
    exp_path_nc = os.path.join(tmp_path / 'test', 'pyDeltaRCM_output.nc')
    assert os.path.isfile(exp_path_nc)

    # load it into memory and check values in the netCDF4
    output = Dataset(exp_path_nc, 'r', allow_pickle=True)
    out_vars = output.variables.keys()
    # check that expected variables are in the file
    assert 'x' in out_vars
    assert 'y' in out_vars
    assert 'time' in out_vars
    assert 'eta' in out_vars
    assert 'depth' in out_vars
    assert 'discharge' in out_vars
    # check attributes of variables
    assert output['time'][0].tolist() == 0.0
    assert output['time'][-1].tolist() == 1800.0
    assert output['eta'][0].shape == (10, 10)
    assert output['eta'][-1].shape == (10, 10)
    assert output['depth'][-1].shape == (10, 10)
    assert output['discharge'][-1].shape == (10, 10)
    # check time
    assert baseModel.dt == 300.0  # base model timestep
    assert baseModel.time == 600.0  # ran 2 steps, so 300*2=600
    assert resumeModel.dt == 300.0  # resume model timestep size
    assert resumeModel.time == 1200.0  # ran 2 steps on top of base model
    # should be 600 + (2*300) = 1200
    assert resumeModel02.dt == 300.0  # same dt
    assert resumeModel02.time == 1800.0  # ran 2 steps + resume, 1200+600=1800

    assert len(output['time'][:].tolist()) == 7  # 0-300-600-900-1200-1500-1800
    # checkpoint interval aligns w/ timestep dt so these should match
    assert output['time'][-1].tolist() == resumeModel02.time


def test_load_nocheckpoint(tmp_path):
    """Try loading a checkpoint file when one doesn't exist."""
    # define a yaml
    file_name = 'trial_run.yaml'
    base_p, base_f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(base_f, 'Length', 10.0)
    utilities.write_parameter_to_file(base_f, 'Width', 10.0)
    utilities.write_parameter_to_file(base_f, 'seed', 0)
    utilities.write_parameter_to_file(base_f, 'dx', 1.0)
    utilities.write_parameter_to_file(base_f, 'L0_meters', 1.0)
    utilities.write_parameter_to_file(base_f, 'Np_water', 10)
    utilities.write_parameter_to_file(base_f, 'u0', 1.0)
    utilities.write_parameter_to_file(base_f, 'N0_meters', 2.0)
    utilities.write_parameter_to_file(base_f, 'h0', 1.0)
    utilities.write_parameter_to_file(base_f, 'SLR', 0.001)
    utilities.write_parameter_to_file(base_f, 'Np_sed', 10)
    utilities.write_parameter_to_file(base_f, 'save_dt', 50)
    utilities.write_parameter_to_file(base_f, 'save_strata', True)
    utilities.write_parameter_to_file(base_f, 'save_eta_grids', True)
    utilities.write_parameter_to_file(base_f, 'save_depth_grids', True)
    utilities.write_parameter_to_file(base_f, 'save_discharge_grids', True)
    utilities.write_parameter_to_file(base_f, 'save_checkpoint', True)
    utilities.write_parameter_to_file(base_f, 'resume_checkpoint', True)
    utilities.write_parameter_to_file(base_f, 'out_dir', tmp_path / 'test')
    base_f.close()

    # try loading the model yaml despite no checkpoint existing
    with pytest.raises(FileNotFoundError):
        _ = DeltaModel(input_file=base_p)


@pytest.mark.skipif(
    platform.system() != 'Linux',
    reason='Parallel support only on Linux OS.')
def test_py_hlvl_parallel_checkpoint(tmp_path):
    """Test checkpointing in parallel."""
    file_name = 'user_parameters.yaml'
    p, f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(f, 'verbose', 2)
    utilities.write_parameter_to_file(f, 'ensemble', 2)
    utilities.write_parameter_to_file(f, 'Length', 10.0)
    utilities.write_parameter_to_file(f, 'Width', 10.0)
    utilities.write_parameter_to_file(f, 'dx', 1.0)
    utilities.write_parameter_to_file(f, 'L0_meters', 1.0)
    utilities.write_parameter_to_file(f, 'N0_meters', 1.0)
    utilities.write_parameter_to_file(f, 'Np_water', 10)
    utilities.write_parameter_to_file(f, 'Np_sed', 10)
    utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'test')
    utilities.write_parameter_to_file(f, 'parallel', 2)
    utilities.write_parameter_to_file(f, 'save_dt', 300)
    utilities.write_parameter_to_file(f, 'save_checkpoint', True)
    utilities.write_parameter_to_file(f, 'save_eta_grids', True)
    f.close()
    pp = preprocessor.Preprocessor(input_file=p, timesteps=2)
    # assertions for job creation
    assert type(pp.file_list) is list
    assert len(pp.file_list) == 2
    assert pp._is_completed is False
    # assertions after running jobs
    pp.run_jobs()
    assert isinstance(pp.job_list[0], preprocessor._ParallelJob)
    assert pp._is_completed is True
    exp_path_nc0 = os.path.join(
        tmp_path / 'test', 'job_000', 'pyDeltaRCM_output.nc')
    exp_path_nc1 = os.path.join(
        tmp_path / 'test', 'job_001', 'pyDeltaRCM_output.nc')
    assert os.path.isfile(exp_path_nc0)
    assert os.path.isfile(exp_path_nc1)
    # check that checkpoint files exist
    exp_path_ckpt0 = os.path.join(
        tmp_path / 'test', 'job_000', 'checkpoint.npz')
    exp_path_ckpt1 = os.path.join(
        tmp_path / 'test', 'job_001', 'checkpoint.npz')
    assert os.path.isfile(exp_path_ckpt0)
    assert os.path.isfile(exp_path_ckpt1)
    # load one output files and check values
    out_old = netCDF4.Dataset(exp_path_nc1)
    assert 'meta' in out_old.groups.keys()
    assert out_old['time'][0].tolist() == 0.0
    assert out_old['time'][-1].tolist() == 600.0  # 2 timesteps of 300.0s run

    # close netCDF file
    out_old.close()
    # try to resume jobs
    file_name = 'user_parameters.yaml'
    p, f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(f, 'verbose', 2)
    utilities.write_parameter_to_file(f, 'ensemble', 2)
    utilities.write_parameter_to_file(f, 'Length', 10.0)
    utilities.write_parameter_to_file(f, 'Width', 10.0)
    utilities.write_parameter_to_file(f, 'dx', 1.0)
    utilities.write_parameter_to_file(f, 'L0_meters', 1.0)
    utilities.write_parameter_to_file(f, 'N0_meters', 1.0)
    utilities.write_parameter_to_file(f, 'Np_water', 10)
    utilities.write_parameter_to_file(f, 'Np_sed', 10)
    utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'test')
    utilities.write_parameter_to_file(f, 'parallel', 2)
    utilities.write_parameter_to_file(f, 'save_dt', 300)
    utilities.write_parameter_to_file(f, 'resume_checkpoint', True)
    utilities.write_parameter_to_file(f, 'save_eta_grids', True)
    f.close()
    pp = preprocessor.Preprocessor(input_file=p, timesteps=1)
    # assertions for job creation
    assert type(pp.file_list) is list
    assert len(pp.file_list) == 2
    assert pp._is_completed is False
    # assertions after running jobs
    pp.run_jobs()
    assert isinstance(pp.job_list[0], preprocessor._ParallelJob)
    assert pp._is_completed is True
    exp_path_nc0 = os.path.join(
        tmp_path / 'test', 'job_000', 'pyDeltaRCM_output.nc')
    exp_path_nc1 = os.path.join(
        tmp_path / 'test', 'job_001', 'pyDeltaRCM_output.nc')
    assert os.path.isfile(exp_path_nc0)
    assert os.path.isfile(exp_path_nc1)
    # check that checkpoint files still exist
    exp_path_ckpt0 = os.path.join(
        tmp_path / 'test', 'job_000', 'checkpoint.npz')
    exp_path_ckpt1 = os.path.join(
        tmp_path / 'test', 'job_001', 'checkpoint.npz')
    assert os.path.isfile(exp_path_ckpt0)
    assert os.path.isfile(exp_path_ckpt1)
    # load one output file to check it out
    out_fin = netCDF4.Dataset(exp_path_nc1)
    assert 'meta' in out_old.groups.keys()
    assert out_fin['time'][0].tolist() == 0
    assert out_fin['time'][-1].tolist() == 900.0  # +1 timestep of 300.0s
    # close netcdf file
    out_fin.close()