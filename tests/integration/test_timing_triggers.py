import glob
import os
import netCDF4
import numpy as np

import unittest.mock as mock

from pyDeltaRCM.model import DeltaModel
from .. import utilities


class TestTimingStratigraphy:

    def test_expand_stratigraphy(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'toggle_subsidence': True,
                                      'start_subsidence': 20000,
                                      'sigma_max': 1e-8,
                                      'save_dt': 50000})
        _delta = DeltaModel(input_file=p)

        # mock the timestep computations
        _delta.run_one_timestep = mock.MagicMock()

        assert _delta.dt == 25000
        assert _delta.n_steps == 10
        assert _delta.strata_counter == 0
        assert _delta.strata_eta[:, _delta.strata_counter].getnnz() == 0
        for _t in range(19):
            assert _delta.strata_eta[:, _delta.strata_counter].getnnz() == 0
            _delta.update()
            assert _delta.time == _delta.dt * (_t + 1)
            assert _delta.strata_eta.shape[1] == 10
        assert _delta.time == 19 * 25000
        assert _delta.strata_counter == 10  # stored 10, invalid next store
        assert _delta.strata_eta.shape[1] == 10
        # nothing occurs on next  update, because save_dt = 2 * dt
        _delta.update()
        assert _delta.time == 20 * 25000
        assert _delta.strata_counter == 10
        assert _delta.strata_eta.shape[1] == 10
        # expansion occurs when model tries to save strata after next update
        _delta.update()
        assert _delta.time == 21 * 25000
        assert _delta.strata_counter == 11
        assert _delta.strata_eta.shape[1] == 20
        # run to bring to even 100 steps, check status again
        for _t in range(79):
            _delta.update()
        assert _delta.time == 100 * 25000
        assert _delta.strata_counter == 50
        assert _delta.strata_eta.shape[1] == 50


class TestTimingOutputData:

    def test_save_one_fig_no_grids(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'save_dt': 1,
                                      'save_strata': False,
                                      'save_eta_figs': True})
        _delta = DeltaModel(input_file=p)

        # mock the timestep computations
        _delta.run_one_timestep = mock.MagicMock()

        # check nothing created at the start
        img_glob = glob.glob(os.path.join(_delta.prefix, '*.png'))
        nc_glob = glob.glob(os.path.join(_delta.prefix, '*.nc'))
        assert len(img_glob) == 0
        assert len(nc_glob) == 0

        # update the delta a few times
        for _ in range(0, 2):
            _delta.update()

        assert _delta.time_iter == 2.0
        _delta.run_one_timestep.call_count == 2

        # check for output eta files
        exp_path_nc = os.path.join(
            tmp_path / 'out_dir', 'pyDeltaRCM_output.nc')
        exp_path_png0 = os.path.join(tmp_path / 'out_dir', 'eta_00000.png')
        exp_path_png1 = os.path.join(tmp_path / 'out_dir', 'eta_00001.png')
        exp_path_png2 = os.path.join(tmp_path / 'out_dir', 'eta_00002.png')
        assert not os.path.isfile(exp_path_nc)
        assert os.path.isfile(exp_path_png0)
        assert os.path.isfile(exp_path_png1)
        assert not os.path.isfile(exp_path_png2)

    def test_save_one_fig_one_grid(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'save_dt': 1,
                                      'save_eta_grids': True,
                                      'save_discharge_figs': True})
        _delta = DeltaModel(input_file=p)

        # mock the timestep computations
        _delta.run_one_timestep = mock.MagicMock()

        exp_path_nc = os.path.join(
            tmp_path / 'out_dir', 'pyDeltaRCM_output.nc')
        assert os.path.isfile(exp_path_nc)
        nc_size_before = os.path.getsize(exp_path_nc)
        assert nc_size_before > 0

        # update a couple times, should increase on each save
        for _ in range(0, 2):
            _delta.update()
        nc_size_middle = os.path.getsize(exp_path_nc)
        assert _delta.time_iter == 2.0
        assert nc_size_middle > nc_size_before

        # now finalize, and file size should increase over middle again
        _delta.finalize()
        nc_size_after = os.path.getsize(exp_path_nc)
        assert _delta.time_iter == 2.0
        assert nc_size_after > nc_size_middle
        assert nc_size_after > nc_size_before

    def test_save_metadata_no_grids(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'save_dt': 1,
                                      'save_metadata': True})
        _delta = DeltaModel(input_file=p)

        # mock the timestep computations
        _delta.run_one_timestep = mock.MagicMock()

        exp_path_nc = os.path.join(
            tmp_path / 'out_dir', 'pyDeltaRCM_output.nc')
        assert os.path.isfile(exp_path_nc)

        for _ in range(0, 2):
            _delta.update()

        assert _delta.time_iter == 2.0
        _delta.run_one_timestep.call_count == 2

        _delta.finalize()

        ds = netCDF4.Dataset(exp_path_nc, "r", format="NETCDF4")
        assert not ('eta' in ds.variables)
        assert ds['meta']['H_SL'].shape[0] == 3
        assert ds['meta']['L0'][:] == 3

    def test_save_one_grid_metadata_by_default(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'save_dt': 1,
                                      'save_metadata': False,
                                      'save_eta_grids': True,
                                      'C0_percent': 0.2})
        _delta = DeltaModel(input_file=p)

        # mock the timestep computations
        _delta.run_one_timestep = mock.MagicMock()

        exp_path_nc = os.path.join(
            tmp_path / 'out_dir', 'pyDeltaRCM_output.nc')
        assert os.path.isfile(exp_path_nc)

        for _ in range(0, 6):
            _delta.update()

        assert _delta.time_iter == 6.0
        _delta.run_one_timestep.call_count == 6

        _delta.finalize()
        ds = netCDF4.Dataset(exp_path_nc, "r", format="NETCDF4")
        _arr = ds.variables['eta']
        assert _arr.shape[1] == _delta.eta.shape[0]
        assert _arr.shape[2] == _delta.eta.shape[1]
        assert ('meta' in ds.groups)  # if any grids, save meta too
        assert ds.groups['meta']['H_SL'].shape[0] == _arr.shape[0]
        assert np.all(ds.groups['meta']['C0_percent'][:] == 0.2)
        assert np.all(ds.groups['meta']['f_bedload'][:] == 0.5)
