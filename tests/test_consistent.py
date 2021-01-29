# unit tests for consistent model outputs

import pytest

import sys
import os
import numpy as np
from netCDF4 import Dataset

from pyDeltaRCM import DeltaModel

from utilities import test_DeltaModel
import utilities

# need to create a simple case of pydeltarcm object to test these functions


def test_bed_after_one_update(test_DeltaModel):
    test_DeltaModel.update()
    # slice is: test_DeltaModel.eta[:5, 4]
    # print(test_DeltaModel.eta[:5, 4])

    _exp = np.array([-1., -0.840265, -0.9976036, -1., -1.])
    assert np.all(test_DeltaModel.eta[:5, 4] == pytest.approx(_exp))


def test_bed_after_ten_updates(test_DeltaModel):

    for _ in range(0, 10):
        test_DeltaModel.update()

    # slice is: test_DeltaModel.eta[:5, 4]
    # print(test_DeltaModel.eta[:5, 4])

    _exp = np.array([1.7, 0.83358884, -0.9256229,  -1., -1.])
    assert np.all(test_DeltaModel.eta[:5, 4] == pytest.approx(_exp))


def test_long_multi_validation(tmp_path):
    # IndexError on corner.

    file_name = 'user_parameters.yaml'
    p, f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(f, 'seed', 42)
    utilities.write_parameter_to_file(f, 'Length', 600.)
    utilities.write_parameter_to_file(f, 'Width', 600.)
    utilities.write_parameter_to_file(f, 'dx', 5)
    utilities.write_parameter_to_file(f, 'Np_water', 10)
    utilities.write_parameter_to_file(f, 'Np_sed', 10)
    utilities.write_parameter_to_file(f, 'f_bedload', 0.05)
    f.close()
    delta = DeltaModel(input_file=p)

    for _ in range(0, 3):
        delta.update()

    # slice is: test_DeltaModel.eta[:5, 62]
    print(delta.eta[:5, 62])

    _exp1 = np.array([-4.9709163, -4.972, -3.722989, -7.786886, -3.7249935])
    assert np.all(delta.eta[:5, 62] == pytest.approx(_exp1))

    for _ in range(0, 10):
        delta.update()

    # slice is: test_DeltaModel.eta[:5, 4]
    print(delta.eta[:5, 62])

    _exp2 = np.array([-4.9709163, -1.5536911, -3.268889, -3.2696986, -2.0843806])
    assert np.all(delta.eta[:5, 62] == pytest.approx(_exp2))


def test_limit_inds_error_inlet_size_fixed_bug_example_1(tmp_path):
    """IndexError due to inlet size being too large by default.

    If the domain was made small (30x60), but the `N0_meters` and `L0_meters`
    parameters were not adjusted, the model domain was filled with landscape
    above sea level and water routing failed due to trying to access a cell
    outside the domain. This produced an IndexError.

    We now limit the size of the inlet to 1/4 of the domain edge length in
    both directions (length and width). This change fixed this case example,
    and it is now here as a consistency check, and to make sure the bug is not
    recreated by mistake.
    """
    file_name = 'user_parameters.yaml'
    p, f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(f, 'seed', 42)
    utilities.write_parameter_to_file(f, 'Length', 30.)
    utilities.write_parameter_to_file(f, 'Width', 60.)
    utilities.write_parameter_to_file(f, 'dx', 1)
    utilities.write_parameter_to_file(f, 'verbose', 1)
    utilities.write_parameter_to_file(f, 'Np_water', 20)
    utilities.write_parameter_to_file(f, 'Np_sed', 20)
    utilities.write_parameter_to_file(f, 'f_bedload', 0.65)
    f.close()
    delta = DeltaModel(input_file=p)

    for _ in range(0, 2):
        delta.update()

    # slice is: test_DeltaModel.eta[:5, 30]
    print(delta.eta[:5, 30])

    _exp = np.array([-4.9988008, -4.7794013, -4.5300136, -4.4977293, -4.56228])
    assert np.all(delta.eta[:5, 30] == pytest.approx(_exp))


def test_limit_inds_error_inlet_size_fixed_bug_example_2(tmp_path):
    """IndexError due to inlet size being too large by default.

    If the domain was made small (30x60), but the `N0_meters` and `L0_meters`
    parameters were not adjusted, the model domain was filled with landscape
    above sea level and water routing failed due to trying to access a cell
    outside the domain. This produced an IndexError.

    We now limit the size of the inlet to 1/4 of the domain edge length in
    both directions (length and width). This change fixed this case example,
    and it is now here as a consistency check, and to make sure the bug is not
    recreated by mistake.
    """
    file_name = 'user_parameters.yaml'
    p, f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(f, 'seed', 43)
    utilities.write_parameter_to_file(f, 'Length', 30.)
    utilities.write_parameter_to_file(f, 'Width', 60.)
    utilities.write_parameter_to_file(f, 'dx', 1)
    utilities.write_parameter_to_file(f, 'verbose', 1)
    utilities.write_parameter_to_file(f, 'Np_water', 20)
    utilities.write_parameter_to_file(f, 'Np_sed', 20)
    utilities.write_parameter_to_file(f, 'f_bedload', 0.15)
    f.close()
    delta = DeltaModel(input_file=p)

    for _ in range(0, 7):
        delta.update()

    # slice is: test_DeltaModel.eta[:5, 30]
    print(delta.eta[:5, 30])

    _exp = np.array([-4.9975486, -4.9140935, -5.15276, -5.3690896, -5.1903167])
    assert np.all(delta.eta[:5, 30] == pytest.approx(_exp))


def test_limit_inds_error_fixed_bug_example_3(tmp_path):
    """IndexError due to inlet width resolving to an edge cell.

    If the domain was made small and long (20x10), then the configuration that
    determined the center cell to hinge the inlet location on, would resolve
    to place the inlet at an edge cell. This led to some index error at the
    end of the water iteration and an IndexError.

    We now recalcualte the value of the self.CTR parameter if the cell is
    chosen as ind = 0 or 1. This is really only relevant to the testing suite,
    where domains are very small. This change fixed this case example, and it
    is now here as a consistency check, and to make sure the bug is not
    recreated by mistake.
    """
    file_name = 'user_parameters.yaml'
    p, f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(f, 'seed', 42)
    utilities.write_parameter_to_file(f, 'Length', 20.)
    utilities.write_parameter_to_file(f, 'Width', 10.)
    utilities.write_parameter_to_file(f, 'dx', 2)
    utilities.write_parameter_to_file(f, 'verbose', 1)
    utilities.write_parameter_to_file(f, 'Np_water', 20)
    utilities.write_parameter_to_file(f, 'Np_sed', 20)
    utilities.write_parameter_to_file(f, 'f_bedload', 0.65)
    f.close()
    delta = DeltaModel(input_file=p)

    for _ in range(0, 2):
        delta.update()

    # slice is: test_DeltaModel.eta[:5, 2]
    print(delta.eta[:5, 2])

    _exp = np.array([-4.99961, -4.605685, -3.8314152, -4.9007816, -5.])
    assert np.all(delta.eta[:5, 2] == pytest.approx(_exp))


def test_model_similarity(tmp_path):
    """Test consistency of two models initialized from same yaml."""
    file_name = 'base_run.yaml'
    p, f = utilities.create_temporary_file(tmp_path, file_name)
    utilities.write_parameter_to_file(f, 'Length', 10.0)
    utilities.write_parameter_to_file(f, 'Width', 10.0)
    utilities.write_parameter_to_file(f, 'seed', 0)
    utilities.write_parameter_to_file(f, 'dx', 1.0)
    utilities.write_parameter_to_file(f, 'L0_meters', 1.0)
    utilities.write_parameter_to_file(f, 'Np_water', 10)
    utilities.write_parameter_to_file(f, 'u0', 1.0)
    utilities.write_parameter_to_file(f, 'N0_meters', 2.0)
    utilities.write_parameter_to_file(f, 'h0', 1.0)
    utilities.write_parameter_to_file(f, 'SLR', 0.001)
    utilities.write_parameter_to_file(f, 'Np_sed', 10)
    utilities.write_parameter_to_file(f, 'save_dt', 50)
    utilities.write_parameter_to_file(f, 'save_strata', True)
    f.close()

    # create and update first model
    ModelA = DeltaModel(input_file=p)
    ModelA.update()
    ModelA.output_netcdf.close()
    # create and update second model
    ModelB = DeltaModel(input_file=p)
    ModelB.update()
    ModelB.output_netcdf.close()

    # fields should be the same
    assert ModelA.time == ModelB.time
    assert ModelA._time_iter == ModelB._time_iter
    assert ModelA._save_iter == ModelB._save_iter
    assert ModelA._save_time_since_last == ModelB._save_time_since_last
    assert np.all(ModelA.uw == ModelB.uw)
    assert np.all(ModelA.ux == ModelB.ux)
    assert np.all(ModelA.uy == ModelB.uy)
    assert np.all(ModelA.depth == ModelB.depth)
    assert np.all(ModelA.stage == ModelB.stage)
    assert np.all(ModelA.strata_eta.todense() ==
                  ModelB.strata_eta.todense())
    assert np.all(ModelA.strata_sand_frac.todense() ==
                  ModelB.strata_sand_frac.todense())


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
