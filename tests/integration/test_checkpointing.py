# unit tests for consistent model outputs

import pytest

import os
import platform
import shutil
import numpy as np
from netCDF4 import Dataset

import unittest.mock as mock

from pyDeltaRCM import DeltaModel
from pyDeltaRCM import preprocessor

from .. import utilities


@mock.patch(
    'pyDeltaRCM.iteration_tools.iteration_tools.solve_water_and_sediment_timestep',
    new=utilities.FastIteratingDeltaModel.solve_water_and_sediment_timestep)
class TestCheckpointingIntegrations:
    """
    The above patch implements an augmented DeltaModel from `utilities`. In
    this modified DeltaModel, the `solve_water_and_sediment_timestep`
    operations (i.e., the time consuming part of the model) is replaced with
    an updating random field. This guarantees that the random-repeatedness of
    checkpointing is validated, but it is much faster and easier to isolate
    checkpointing-related issues from model issues.
    """

    def test_simple_checkpoint(self, tmp_path):
        """Test checkpoint vs a base run.

        Also, checks resumed model against another checkpoint run.
        """
        # define a yaml for the longer model run
        file_name = 'base_run.yaml'
        base_p, base_f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(base_f, 'out_dir', tmp_path / 'test')
        utilities.write_parameter_to_file(base_f, 'save_checkpoint', True)
        base_f.close()
        longModel = DeltaModel(input_file=base_p)

        # run for some number of updates
        for _ in range(0, 50):
            longModel.update()
        longModel.finalize()

        # try defining a new model but plan to load checkpoint from longModel
        file_name = 'base_run.yaml'
        base_p, base_f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(base_f, 'out_dir', tmp_path / 'test')
        utilities.write_parameter_to_file(base_f, 'resume_checkpoint', True)
        base_f.close()
        resumeModel = DeltaModel(input_file=base_p)

        # advance the resumed model until it catch up to longModel
        assert resumeModel.time < longModel.time
        while resumeModel._time < longModel._time:
            resumeModel.update()
        resumeModel.finalize()

        # the longModel and resumeModel should match
        assert longModel.time == resumeModel.time
        assert np.all(longModel.eta == resumeModel.eta)
        assert np.all(longModel.uw == resumeModel.uw)
        assert np.all(longModel.ux == resumeModel.ux)
        assert np.all(longModel.uy == resumeModel.uy)
        assert np.all(longModel.depth == resumeModel.depth)
        assert np.all(longModel.stage == resumeModel.stage)
        assert np.all(longModel.sand_frac == resumeModel.sand_frac)
        assert np.all(longModel.active_layer == resumeModel.active_layer)

        # define another model that loads the checkpoint
        file_name = 'base_run.yaml'
        base_p, base_f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(base_f, 'out_dir', tmp_path / 'test')
        utilities.write_parameter_to_file(base_f, 'resume_checkpoint', True)
        base_f.close()
        resumeModel2 = DeltaModel(input_file=base_p)

        # advance the resumed model until it catch up to longModel
        while resumeModel2._time < resumeModel._time:
            resumeModel2.update()
        resumeModel2.finalize()

        # the two models that resumed from the checkpoint should be the same
        assert resumeModel2.time == resumeModel.time
        assert np.all(resumeModel2.uw == resumeModel.uw)
        assert np.all(resumeModel2.ux == resumeModel.ux)
        assert np.all(resumeModel2.uy == resumeModel.uy)
        assert np.all(resumeModel2.depth == resumeModel.depth)
        assert np.all(resumeModel2.stage == resumeModel.stage)
        assert np.all(resumeModel2.sand_frac == resumeModel.sand_frac)
        assert np.all(resumeModel2.active_layer == resumeModel.active_layer)

    def test_checkpoint_nc(self, tmp_path):
        """Test the netCDF that is written to by the checkpointing."""
        # define a yaml for the base model run
        file_name = 'base_run.yaml'
        base_p, base_f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(base_f, 'out_dir', tmp_path / 'test')
        utilities.write_parameter_to_file(base_f, 'save_eta_grids', True)
        utilities.write_parameter_to_file(base_f, 'save_depth_grids', True)
        utilities.write_parameter_to_file(base_f, 'save_discharge_grids', True)
        utilities.write_parameter_to_file(base_f, 'save_sandfrac_grids', True)
        utilities.write_parameter_to_file(base_f, 'save_checkpoint', True)
        base_f.close()
        baseModel = DeltaModel(input_file=base_p)

        # run for some base number of steps
        nt_base = 50
        for _ in range(0, 50):
            baseModel.update()

        # force the model run to end immmediately after exporting a checkpoint
        nt_var = 0
        while (baseModel._save_time_since_checkpoint != 0):
            baseModel.update()
            nt_var += 1

        # then finalize
        baseModel.finalize()

        # check that the time makes sense
        assert baseModel.time == baseModel._dt * (nt_base + nt_var)

        # try defining a new model but plan to load checkpoint from baseModel
        file_name = 'base_run.yaml'
        base_p, base_f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(base_f, 'out_dir', tmp_path / 'test')
        utilities.write_parameter_to_file(base_f, 'save_eta_grids', True)
        utilities.write_parameter_to_file(base_f, 'save_depth_grids', True)
        utilities.write_parameter_to_file(base_f, 'save_discharge_grids', True)
        utilities.write_parameter_to_file(base_f, 'save_sandfrac_grids', True)
        utilities.write_parameter_to_file(base_f, 'save_checkpoint', False)
        utilities.write_parameter_to_file(base_f, 'resume_checkpoint', True)
        base_f.close()
        resumeModel = DeltaModel(input_file=base_p)

        assert resumeModel.time == baseModel.time  # same when resumed

        # advance it until output_data has been called again
        nt_resume = 0
        while (resumeModel._save_time_since_data != 0) or (nt_resume < 50):
            resumeModel.update()
            nt_resume += 1
        resumeModel.finalize()

        assert nt_resume > 0
        assert resumeModel.time > baseModel.time

        # assert that output netCDF4 exists
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
        assert 'sandfrac' in out_vars

        # check attributes of variables
        assert output['time'][0].tolist() == 0.0
        assert output['time'][-1] == resumeModel.time
        assert output['time'][-1].tolist() == resumeModel._dt * \
            (nt_base + nt_var + nt_resume)
        assert output['eta'][0].shape == resumeModel.eta.shape
        assert output['eta'][-1].shape == resumeModel.eta.shape
        assert output['depth'][-1].shape == resumeModel.eta.shape
        assert output['discharge'][-1].shape == resumeModel.eta.shape
        assert output['sandfrac'][-1].shape == resumeModel.eta.shape
        # check the metadata
        assert output['meta']['L0'][:] == resumeModel.L0
        assert output['meta']['N0'][:] == resumeModel.N0
        assert output['meta']['CTR'][:] == resumeModel.CTR
        assert output['meta']['dx'][:] == resumeModel.dx
        assert output['meta']['h0'][:] == resumeModel.h0
        assert np.all(output['meta']['cell_type'][:] == resumeModel.cell_type)
        assert output['meta']['H_SL'][-1].data == resumeModel.H_SL
        assert output['meta']['f_bedload'][-1].data == resumeModel.f_bedload
        assert pytest.approx(float(output['meta']['C0_percent'][-1].data) ==
                             resumeModel.C0_percent)
        assert output['meta']['u0'][-1].data == resumeModel.u0

        # checkpoint interval aligns w/ timestep dt so these should match
        assert output['time'][-1].tolist() == resumeModel.time

    def test_checkpoint_diff_dt(self, tmp_path):
        """Test when checkpoint_dt does not match dt or save_dt."""
        # define a yaml for the base model run
        file_name = 'base_run.yaml'
        base_p, base_f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(base_f, 'save_eta_grids', True)
        utilities.write_parameter_to_file(base_f, 'save_depth_grids', True)
        utilities.write_parameter_to_file(base_f, 'save_discharge_grids', True)
        utilities.write_parameter_to_file(base_f, 'save_checkpoint', True)
        utilities.write_parameter_to_file(base_f, 'out_dir', tmp_path / 'test')
        base_f.close()
        baseModel = DeltaModel(input_file=base_p)

        # modify the checkpoint dt to be different than save_dt
        baseModel._checkpoint_dt = (baseModel.save_dt * 0.65)

        for _ in range(0, 50):
            baseModel.update()
        baseModel.finalize()

        assert baseModel.time == baseModel._dt * 50
        baseModelSavedTime = (baseModel.time -
                              baseModel._save_time_since_checkpoint)
        assert baseModelSavedTime > 0

        # try defining a new model but plan to load checkpoint from baseModel
        file_name = 'base_run.yaml'
        base_p, base_f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(base_f, 'save_eta_grids', True)
        utilities.write_parameter_to_file(base_f, 'save_depth_grids', True)
        utilities.write_parameter_to_file(base_f, 'save_discharge_grids', True)
        utilities.write_parameter_to_file(base_f, 'save_checkpoint', False)
        utilities.write_parameter_to_file(base_f, 'resume_checkpoint', True)
        utilities.write_parameter_to_file(base_f, 'out_dir', tmp_path / 'test')
        base_f.close()
        resumeModel = DeltaModel(input_file=base_p)

        assert resumeModel.time == baseModelSavedTime

        # advance until some steps and just saved
        nt_resume = 0
        while (resumeModel._save_time_since_data != 0) or (nt_resume < 50):
            resumeModel.update()
            nt_resume += 1
        resumeModel.finalize()

        # assert that output netCDF4 exists
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
        assert output['time'][-1].tolist() == resumeModel.time

    def test_multi_checkpoints(self, tmp_path):
        """Test using checkpoints multiple times for a given model run."""
        # define a yaml for the base model run
        file_name = 'base_run.yaml'
        base_p, base_f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(base_f, 'save_eta_grids', True)
        utilities.write_parameter_to_file(base_f, 'save_checkpoint', True)
        utilities.write_parameter_to_file(base_f, 'out_dir', tmp_path / 'test')
        base_f.close()
        baseModel = DeltaModel(input_file=base_p)

        # run base for 2 timesteps
        for _ in range(0, 50):
            baseModel.update()
        baseModel.finalize()

        # try defining a new model but plan to load checkpoint from baseModel
        file_name = 'base_run.yaml'
        base_p, base_f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(base_f, 'save_eta_grids', True)
        utilities.write_parameter_to_file(base_f, 'save_checkpoint', True)
        utilities.write_parameter_to_file(base_f, 'resume_checkpoint', True)
        utilities.write_parameter_to_file(base_f, 'out_dir', tmp_path / 'test')
        base_f.close()
        resumeModel = DeltaModel(input_file=base_p)

        assert resumeModel.time <= baseModel.time

        # advance it more steps
        for _ in range(0, 25):
            resumeModel.update()
        resumeModel.finalize()

        # create another resume model
        resumeModel02 = DeltaModel(input_file=base_p)

        assert resumeModel02.time <= resumeModel.time  # should be same

        # step it some more
        nt_resume02 = 0
        while (resumeModel02._save_time_since_data != 0) or (nt_resume02 < 50):
            resumeModel02.update()
            nt_resume02 += 1

        # assert that output netCDF4 exists
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
        # check attributes of variables
        assert output['time'][0].tolist() == 0.0
        assert output['time'][-1].tolist() == resumeModel02.time

    def test_load_nocheckpoint(self, tmp_path):
        """Try loading a checkpoint file when one doesn't exist."""
        # define a yaml
        file_name = 'trial_run.yaml'
        base_p, base_f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(base_f, 'resume_checkpoint', True)
        utilities.write_parameter_to_file(base_f, 'out_dir', tmp_path / 'test')
        base_f.close()

        # try loading the model yaml despite no checkpoint existing
        with pytest.raises(FileNotFoundError):
            _ = DeltaModel(input_file=base_p)

    @pytest.mark.skipif(
        platform.system() != 'Linux',
        reason='Parallel support only on Linux OS.')
    def test_py_hlvl_parallel_checkpoint(self, tmp_path):
        """Test checkpointing in parallel."""
        file_name = 'user_parameters.yaml'
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, 'ensemble', 2)
        utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'test')
        utilities.write_parameter_to_file(f, 'parallel', 2)
        utilities.write_parameter_to_file(f, 'save_checkpoint', True)
        utilities.write_parameter_to_file(f, 'save_eta_grids', True)
        f.close()
        pp = preprocessor.Preprocessor(input_file=p, timesteps=50)
        # assertions for job creation
        assert len(pp.file_list) == 2
        assert pp._is_completed is False

        # run the jobs, mocked deltas
        pp.run_jobs()

        # compute the expected final time recorded
        _dt = pp.job_list[1].deltamodel._dt
        _checkpoint_dt = pp.job_list[1].deltamodel._checkpoint_dt
        expected_save_interval = (((_checkpoint_dt // _dt) + 1) * _dt)
        expected_last_save_time = (((50 * _dt) // expected_save_interval) *
                                   expected_save_interval)

        # assertions after running jobs
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
        out_old = Dataset(exp_path_nc1)
        assert 'meta' in out_old.groups.keys()
        assert out_old['time'][0].tolist() == 0.0
        assert out_old['time'][-1].tolist() == expected_last_save_time

        # close netCDF file
        out_old.close()

        # try to resume jobs
        file_name = 'user_parameters.yaml'
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, 'ensemble', 2)
        utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'test')
        utilities.write_parameter_to_file(f, 'parallel', 2)
        utilities.write_parameter_to_file(f, 'resume_checkpoint', True)
        utilities.write_parameter_to_file(f, 'save_eta_grids', True)
        f.close()
        pp = preprocessor.Preprocessor(input_file=p, timesteps=50)
        # assertions for job creation
        assert len(pp.file_list) == 2
        assert pp._is_completed is False

        # run the jobs, mocked deltas
        pp.run_jobs()

        # assertions after running jobs
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
        out_fin = Dataset(exp_path_nc1)
        assert 'meta' in out_old.groups.keys()
        assert out_fin['time'][0].tolist() == 0
        assert out_fin['time'][-1].tolist() == expected_last_save_time * 2
        # close netcdf file
        out_fin.close()


class TestCheckpointingCreatingLoading:

    def test_load_checkpoint_with_netcdf(self, tmp_path):
        """Test that a run can be resumed when there are outputs.
        """
        # define a yaml with outputs (defaults will output strata)
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'save_checkpoint': True,
                                      'save_eta_grids': True})
        _delta = DeltaModel(input_file=p)

        # replace eta with a random field for checkpointing success check
        _rand_field = np.random.uniform(0, 1, size=_delta.eta.shape)
        _delta.eta = _rand_field

        _delta._save_time_since_checkpoint = float("inf")
        _delta.output_checkpoint()  # force another checkpoint
        _delta.finalize()

        # paths exists
        assert os.path.isfile(os.path.join(
            _delta.prefix, 'pyDeltaRCM_output.nc'))
        assert os.path.isfile(os.path.join(
            _delta.prefix, 'checkpoint.npz'))
        _delta = []  # clear

        # can be resumed
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'save_checkpoint': True,
                                      'resume_checkpoint': True})
        _delta = DeltaModel(input_file=p)

        # check that fields match
        assert np.all(_delta.eta == _rand_field)

    def test_create_checkpoint_without_netcdf(self, tmp_path):
        """Test that a checkpoint can be created when there are no outputs
        """
        # define a yaml with NO outputs, but checkpoint
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'save_checkpoint': True})

        _delta = DeltaModel(input_file=p)

        # replace eta with a random field for checkpointing success check
        _rand_field = np.random.uniform(0, 1, size=_delta.eta.shape)
        _delta.eta = _rand_field

        _delta._save_time_since_checkpoint = float("inf")
        _delta.output_checkpoint()  # force another checkpoint
        _delta.finalize()

        # should be no file
        assert not os.path.isfile(os.path.join(
            _delta.prefix, 'pyDeltaRCM_output.nc'))

        # can be resumed
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'save_checkpoint': True,
                                      'resume_checkpoint': True})
        _delta = DeltaModel(input_file=p)

        # check that fields match
        assert np.all(_delta.eta == _rand_field)

    def test_load_checkpoint_without_netcdf(self, tmp_path):
        """Test that a run can be resumed when there are outputs.
        """
        # define a yaml with NO outputs, but checkpoint
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'save_checkpoint': True})
        _delta = DeltaModel(input_file=p)

        # replace eta with a random field for checkpointing success check
        _rand_field = np.random.uniform(0, 1, size=_delta.eta.shape)
        _delta.eta = _rand_field

        _delta._save_time_since_checkpoint = float("inf")
        _delta.output_checkpoint()  # force another checkpoint
        _delta.finalize()

        # should be no nc file but should be a checkpoint file
        assert not os.path.isfile(os.path.join(
            _delta.prefix, 'pyDeltaRCM_output.nc'))
        assert os.path.isfile(os.path.join(
            _delta.prefix, 'checkpoint.npz'))

        # now try to resume, will WARN on not finding netcdf
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'save_checkpoint': True,
                                      'save_eta_grids': True,
                                      'resume_checkpoint': True})
        with pytest.warns(UserWarning, match=r'NetCDF4 output *.'):
            _delta = DeltaModel(input_file=p)

        # assert that a new output file exists file exists
        assert os.path.isfile(os.path.join(
            _delta.prefix, 'pyDeltaRCM_output.nc'))
        assert os.path.isfile(os.path.join(
            _delta.prefix, 'checkpoint.npz'))

        # check that fields match
        assert np.all(_delta.eta == _rand_field)

    @mock.patch(
        'pyDeltaRCM.iteration_tools.iteration_tools.solve_water_and_sediment_timestep',
        new=utilities.FastIteratingDeltaModel.solve_water_and_sediment_timestep)
    @pytest.mark.skipif(
        platform.system() != 'Linux',
        reason='Parallel support only on Linux OS.')
    def test_load_ckpt_wo_netcdf_parallel_spinup_to_matrix(self, tmp_path):
        """
        Test that multiple matrix runs can be resumed from a single checkpoint
        file, and take advantage of the preprocessor parallel infrastructure.
        """
        # define a yaml with an output and checkpoint
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'save_checkpoint': True,
                                      'save_eta_grids': True})
        baseModel = DeltaModel(input_file=p)

        # run base for 2 timesteps
        for _ in range(0, 50):
            baseModel.update()
        baseModel.finalize()

        # get the number of times the data has been output in the base file
        bmsi = baseModel._save_iter
        assert bmsi > 0

        # check that files exist, and then delete nc
        assert os.path.isfile(os.path.join(
            baseModel.prefix, 'pyDeltaRCM_output.nc'))
        assert os.path.isfile(os.path.join(
            baseModel.prefix, 'checkpoint.npz'))

        # open the file and check dimensions
        exp_nc_path = os.path.join(baseModel.prefix, 'pyDeltaRCM_output.nc')
        output = Dataset(exp_nc_path, 'r', allow_pickle=True)
        out_vars = output.variables.keys()
        # check that expected variables are in the file
        assert 'x' in out_vars
        assert 'y' in out_vars
        assert 'time' in out_vars
        assert 'eta' in out_vars
        # check attributes of variables
        assert output['time'].shape[0] == bmsi

        ########################

        # set up a matrix of runs
        resume_dict = {'save_checkpoint': False,
                       'resume_checkpoint': True,
                       'save_eta_grids': True,
                       'out_dir': os.path.join(tmp_path, 'matrix'),
                       'parallel': 4}
        _matrix = {'f_bedload': [0.1, 0.2, 0.5, 1]}
        resume_dict['matrix'] = _matrix

        # let the preprocessor write the initial matrix and
        #    create a new output netcdf file
        pp = preprocessor.Preprocessor(
            resume_dict,
            timesteps=25)  # 2000

        # now copy the checkpoint
        for j, j_file in enumerate(pp.file_list):
            # get folder
            j_folder = j_file.parent

            # copy the spinup checkpoint to each of the folders
            shutil.copy(
                src=os.path.join(baseModel.prefix, 'checkpoint.npz'),
                dst=os.path.join(tmp_path, 'matrix', j_folder))

        pp.run_jobs()

        # first job save dimension
        fjsi = pp.job_list[0].deltamodel._save_iter

        # proceed with assertions
        for j, j_job in enumerate(pp.job_list):
            exp_nc_path = os.path.join(
                tmp_path, 'matrix', j_folder, 'pyDeltaRCM_output.nc')
            assert os.path.isfile(exp_nc_path)

            # check all jobs have same dimensionality
            jsi = j_job.deltamodel._save_iter
            assert jsi == fjsi

            # open the file and check dimensions
            output = Dataset(exp_nc_path, 'r', allow_pickle=True)
            out_vars = output.variables.keys()
            # check that expected variables are in the file
            assert 'x' in out_vars
            assert 'y' in out_vars
            assert 'time' in out_vars
            assert 'eta' in out_vars
            # check attributes of variables

            # this is the critical check, that the dimension of the second
            # netcdf is not expanded to begin at _save_ter from the initial
            # basemodel
            assert output['time'].shape[0] <= (bmsi // 2)

    @pytest.mark.skipif(
        platform.system() == 'Windows',
        reason='OS differences regarding netCDF permissions.')
    def test_load_checkpoint_with_open_netcdf(self, tmp_path):
        """Test what happens if output netCDF file is actually open.

        This is not the same as when the netCDF file is open in another
        process. That situation raises an error for all OS.
        """
        # define a yaml with outputs (defaults will output strata)
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'save_checkpoint': True,
                                      'save_eta_grids': True})
        _delta = DeltaModel(input_file=p)

        # replace eta with a random field for checkpointing success check
        _rand_field = np.random.uniform(0, 1, size=_delta.eta.shape)
        _delta.eta = _rand_field

        _delta._save_time_since_checkpoint = float("inf")
        _delta.output_checkpoint()  # force another checkpoint
        _delta.finalize()

        # paths exists
        assert os.path.isfile(os.path.join(
            _delta.prefix, 'pyDeltaRCM_output.nc'))
        assert os.path.isfile(os.path.join(
            _delta.prefix, 'checkpoint.npz'))
        _prefix = _delta.prefix
        _delta = []  # clear

        # open the netCDF file
        _opened = Dataset(os.path.join(_prefix, 'pyDeltaRCM_output.nc'),
                          'r+', format='NETCDF4')
        assert type(_opened) == Dataset
        assert 'eta' in _opened.variables.keys()
        # saved grid is the initial one, before random field was assigned
        assert np.all(_opened.variables['eta'][:].data != _rand_field)

        # can be resumed
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'save_checkpoint': True,
                                      'resume_checkpoint': True,
                                      'save_eta_grids': True})
        _delta = DeltaModel(input_file=p)
        # force save grids/figs
        _delta.save_grids_and_figs()

        # check that fields match
        assert np.all(_delta.eta == _rand_field)
        # assert that old netCDF object is still around and hasn't changed
        assert type(_opened) == Dataset
        assert 'eta' in _opened.variables.keys()
        # grid from old netCDF is initial one, before random field was assigned
        assert np.all(_opened.variables['eta'][:].data != _rand_field)

        # clear delta
        _delta = []
        # open the new netCDF file
        _new = Dataset(os.path.join(_prefix, 'pyDeltaRCM_output.nc'),
                       'r+', format='NETCDF4')
        # first grid should be the OG one
        assert np.all(_opened['eta'][:].data == _new['eta'][0, :, :].data)
        # random field should be saved in the new netCDF file
        # some rounding/truncation happens in the netCDF so we use approx
        assert pytest.approx(_rand_field == _new['eta'][1, :, :].data)

    @pytest.mark.skipif(
        platform.system() != 'Windows',
        reason='OS differences regarding netCDF permissions.')
    def test_load_checkpoint_with_open_netcdf_win(self, tmp_path):
        """Test what happens if output netCDF file is actually open.

        This is not the same as when the netCDF file is open in another
        process. That situation raises an error for all OS.
        """
        # define a yaml with outputs (defaults will output strata)
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'save_checkpoint': True,
                                      'save_eta_grids': True})
        _delta = DeltaModel(input_file=p)

        # replace eta with a random field for checkpointing success check
        _rand_field = np.random.uniform(0, 1, size=_delta.eta.shape)
        _delta.eta = _rand_field

        _delta._save_time_since_checkpoint = float("inf")
        _delta.output_checkpoint()  # force another checkpoint
        _delta.finalize()

        # paths exists
        assert os.path.isfile(os.path.join(
            _delta.prefix, 'pyDeltaRCM_output.nc'))
        assert os.path.isfile(os.path.join(
            _delta.prefix, 'checkpoint.npz'))
        _prefix = _delta.prefix
        _delta = []  # clear

        # open the netCDF file
        _opened = Dataset(os.path.join(_prefix, 'pyDeltaRCM_output.nc'),
                          'r+', format='NETCDF4')
        assert type(_opened) == Dataset
        assert 'eta' in _opened.variables.keys()
        # saved grid is the initial one, before random field was assigned
        assert np.all(_opened.variables['eta'][:].data != _rand_field)

        # can be resumed
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'save_checkpoint': True,
                                      'resume_checkpoint': True,
                                      'save_eta_grids': True})
        # raises a permissions error on Windows
        with pytest.raises(PermissionError):
            _ = DeltaModel(input_file=p)
