from pathlib import Path
import platform

import pytest
import numpy as np

from .. import utilities
from pyDeltaRCM import DeltaModel


class TestConsistentOutputsBetweenMerges:
    def test_bed_after_one_update(self, tmp_path: Path) -> None:
        file_name = "user_parameters.yaml"
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, "out_dir", tmp_path / "out_dir")
        utilities.write_parameter_to_file(f, "Length", 10.0)
        utilities.write_parameter_to_file(f, "Width", 10.0)
        utilities.write_parameter_to_file(f, "seed", 0)
        utilities.write_parameter_to_file(f, "dx", 1.0)
        utilities.write_parameter_to_file(f, "L0_meters", 1.0)
        utilities.write_parameter_to_file(f, "S0", 0.0002)
        utilities.write_parameter_to_file(f, "itermax", 1)
        utilities.write_parameter_to_file(f, "Np_water", 10)
        utilities.write_parameter_to_file(f, "u0", 1.0)
        utilities.write_parameter_to_file(f, "N0_meters", 2.0)
        utilities.write_parameter_to_file(f, "h0", 1.0)
        utilities.write_parameter_to_file(f, "H_SL", 0.0)
        utilities.write_parameter_to_file(f, "SLR", 0.001)
        utilities.write_parameter_to_file(f, "Np_sed", 10)
        utilities.write_parameter_to_file(f, "f_bedload", 0.5)
        utilities.write_parameter_to_file(f, "C0_percent", 0.1)
        utilities.write_parameter_to_file(f, "toggle_subsidence", False)
        utilities.write_parameter_to_file(f, "start_subsidence", 50.0)
        utilities.write_parameter_to_file(f, "save_eta_figs", False)
        utilities.write_parameter_to_file(f, "save_stage_figs", False)
        utilities.write_parameter_to_file(f, "save_depth_figs", False)
        utilities.write_parameter_to_file(f, "save_discharge_figs", False)
        utilities.write_parameter_to_file(f, "save_velocity_figs", False)
        utilities.write_parameter_to_file(f, "save_eta_grids", False)
        utilities.write_parameter_to_file(f, "save_stage_grids", False)
        utilities.write_parameter_to_file(f, "save_depth_grids", False)
        utilities.write_parameter_to_file(f, "save_discharge_grids", False)
        utilities.write_parameter_to_file(f, "save_velocity_grids", False)
        utilities.write_parameter_to_file(f, "save_dt", 500)
        f.close()
        _delta = DeltaModel(input_file=p)

        _delta.update()

        # slice is: _delta.eta[:5, 4]
        _exp = np.array([-1.0, -0.9152762, -1.0004134, -1.0, -1.0])
        assert np.all(_delta.eta[:5, 4] == pytest.approx(_exp))

    def test_bed_after_ten_updates(self, tmp_path: Path) -> None:
        file_name = "user_parameters.yaml"
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, "out_dir", tmp_path / "out_dir")
        utilities.write_parameter_to_file(f, "Length", 10.0)
        utilities.write_parameter_to_file(f, "Width", 10.0)
        utilities.write_parameter_to_file(f, "seed", 0)
        utilities.write_parameter_to_file(f, "dx", 1.0)
        utilities.write_parameter_to_file(f, "L0_meters", 1.0)
        utilities.write_parameter_to_file(f, "S0", 0.0002)
        utilities.write_parameter_to_file(f, "itermax", 1)
        utilities.write_parameter_to_file(f, "Np_water", 10)
        utilities.write_parameter_to_file(f, "u0", 1.0)
        utilities.write_parameter_to_file(f, "N0_meters", 2.0)
        utilities.write_parameter_to_file(f, "h0", 1.0)
        utilities.write_parameter_to_file(f, "H_SL", 0.0)
        utilities.write_parameter_to_file(f, "SLR", 0.001)
        utilities.write_parameter_to_file(f, "Np_sed", 10)
        utilities.write_parameter_to_file(f, "f_bedload", 0.5)
        utilities.write_parameter_to_file(f, "C0_percent", 0.1)
        utilities.write_parameter_to_file(f, "toggle_subsidence", False)
        utilities.write_parameter_to_file(f, "start_subsidence", 50.0)
        utilities.write_parameter_to_file(f, "save_eta_figs", False)
        utilities.write_parameter_to_file(f, "save_stage_figs", False)
        utilities.write_parameter_to_file(f, "save_depth_figs", False)
        utilities.write_parameter_to_file(f, "save_discharge_figs", False)
        utilities.write_parameter_to_file(f, "save_velocity_figs", False)
        utilities.write_parameter_to_file(f, "save_eta_grids", False)
        utilities.write_parameter_to_file(f, "save_stage_grids", False)
        utilities.write_parameter_to_file(f, "save_depth_grids", False)
        utilities.write_parameter_to_file(f, "save_discharge_grids", False)
        utilities.write_parameter_to_file(f, "save_velocity_grids", False)
        utilities.write_parameter_to_file(f, "save_dt", 500)
        f.close()
        _delta = DeltaModel(input_file=p)

        for _ in range(0, 10):
            _delta.update()

        # slice is: test_DeltaModel.eta[:5, 4]
        _exp = np.array([1.7, 0.394864, -0.95006764, -1.0, -1.0])
        assert np.all(_delta.eta[:5, 4] == pytest.approx(_exp))

    def test_long_multi_validation(self, tmp_path: Path) -> None:
        # IndexError on corner.

        file_name = "user_parameters.yaml"
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, "seed", 42)
        utilities.write_parameter_to_file(f, "Length", 600.0)
        utilities.write_parameter_to_file(f, "Width", 600.0)
        utilities.write_parameter_to_file(f, "dx", 5)
        utilities.write_parameter_to_file(f, "Np_water", 10)
        utilities.write_parameter_to_file(f, "Np_sed", 10)
        utilities.write_parameter_to_file(f, "f_bedload", 0.05)
        f.close()
        delta = DeltaModel(input_file=p)

        for _ in range(0, 3):
            delta.update()

        # slice is: test_DeltaModel.eta[:5, 62]
        _exp1 = np.array([-4.976912, -4.979, -7.7932253, -4.9805, -2.7937498])
        assert np.all(delta.eta[:5, 62] == pytest.approx(_exp1))

        for _ in range(0, 10):
            delta.update()

        _exp2 = np.array([-4.9614887, -3.4891236, -12.195051, -4.6706524, -2.7937498])
        assert np.all(delta.eta[:5, 62] == pytest.approx(_exp2))
        delta.finalize()


class TestModelIsReproducible:
    def test_same_result_two_models(self, tmp_path: Path) -> None:
        """Test consistency of two models initialized from same yaml."""
        p1 = utilities.yaml_from_dict(
            tmp_path,
            "input_1.yaml",
            {
                "out_dir": tmp_path / "out_dir_1",
                "seed": 10,
                "save_sandfrac_grids": True,
            },
        )
        p2 = utilities.yaml_from_dict(
            tmp_path,
            "input_2.yaml",
            {
                "out_dir": tmp_path / "out_dir_2",
                "seed": 10,
                "save_sandfrac_grids": True,
            },
        )

        # create and update first model
        ModelA = DeltaModel(input_file=p1)
        ModelA.update()
        ModelA.output_netcdf.close()
        # create and update second model
        ModelB = DeltaModel(input_file=p2)
        ModelB.update()
        ModelB.output_netcdf.close()

        # fields should be the same
        assert ModelA.time == ModelB.time
        assert ModelA._time_iter == ModelB._time_iter
        assert ModelA._save_iter == ModelB._save_iter
        assert ModelA._save_time_since_data == ModelB._save_time_since_data
        assert np.all(ModelA.uw == ModelB.uw)
        assert np.all(ModelA.ux == ModelB.ux)
        assert np.all(ModelA.uy == ModelB.uy)
        assert np.all(ModelA.depth == ModelB.depth)
        assert np.all(ModelA.stage == ModelB.stage)
        assert np.all(ModelA.sand_frac == ModelB.sand_frac)
        assert np.all(ModelA.active_layer == ModelB.active_layer)

    def test_same_result_two_models_diff_save_dt(self, tmp_path: Path) -> None:
        """Test consistency of two models initialized from same yaml."""
        p1 = utilities.yaml_from_dict(
            tmp_path,
            "input_1.yaml",
            {
                "out_dir": tmp_path / "out_dir_1",
                "seed": 10,
                "save_sandfrac_grids": True,
                "save_dt": 1,
                "Length": 10.0,
                "Width": 10.0,
                "dx": 1.0,
                "L0_meters": 1.0,
            },
        )
        p2 = utilities.yaml_from_dict(
            tmp_path,
            "input_2.yaml",
            {
                "out_dir": tmp_path / "out_dir_2",
                "seed": 10,
                "save_sandfrac_grids": True,
                "save_dt": 2,
                "Length": 10.0,
                "Width": 10.0,
                "dx": 1.0,
                "L0_meters": 1.0,
            },
        )

        # create and update first model
        ModelA = DeltaModel(input_file=p1)
        for _ in range(10):
            ModelA.update()
        ModelA.output_netcdf.close()
        # create and update second model
        ModelB = DeltaModel(input_file=p2)
        for _ in range(10):
            ModelB.update()
        ModelB.output_netcdf.close()

        # fields should be the same
        assert ModelA.time == ModelB.time
        assert ModelA._time_iter == ModelB._time_iter
        assert ModelA._save_iter == ModelB._save_iter
        assert ModelA._save_time_since_data == ModelB._save_time_since_data
        assert np.all(ModelA.uw == ModelB.uw)
        assert np.all(ModelA.ux == ModelB.ux)
        assert np.all(ModelA.uy == ModelB.uy)
        assert np.all(ModelA.depth == ModelB.depth)
        assert np.all(ModelA.stage == ModelB.stage)
        assert np.all(ModelA.sand_frac == ModelB.sand_frac)
        assert np.all(ModelA.active_layer == ModelB.active_layer)


class CustomParamModel(DeltaModel):
    """Subclass for custom yaml parameters."""

    def __init__(self, input_file=None, defer_output: bool = False, **kwargs) -> None:
        super().__init__(input_file, **kwargs)

    def hook_import_files(self) -> None:
        """Hook to define custom yaml parameters."""
        self.subclass_parameters["new_str"] = {
            "type": ["str"],
            "default": "DefaultString",
        }
        self.subclass_parameters["new_val"] = {"type": ["int", "float"], "default": 0}


class TestCustomParams:
    def test_custom_defaults(self, tmp_path: Path) -> None:
        """Default subclass yaml parameters."""
        file_name = "user_parameters.yaml"
        p = utilities.yaml_from_dict(tmp_path, file_name)
        # initialize model
        _delta = CustomParamModel(input_file=p)
        # assert that hook has been used and default params exist
        assert _delta.subclass_parameters["new_str"]["default"] == "DefaultString"
        assert _delta.subclass_parameters["new_val"]["default"] == 0
        assert hasattr(_delta, "subclass_parameters")
        assert _delta.new_str == "DefaultString"
        assert _delta.new_val == 0

    def test_yaml_custom_params(self, tmp_path: Path) -> None:
        """Specify custom params in yaml."""
        file_name = "user_parameters.yaml"
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, "out_dir", tmp_path / "out_dir")
        utilities.write_parameter_to_file(f, "new_str", "Customized")
        utilities.write_parameter_to_file(f, "new_val", 10.5)
        f.close()
        # initialize model
        _delta = CustomParamModel(input_file=p)
        # assert that hook has been used and default params exist
        assert _delta.subclass_parameters["new_str"]["default"] == "DefaultString"
        assert _delta.subclass_parameters["new_val"]["default"] == 0
        assert hasattr(_delta, "subclass_parameters")
        assert _delta.new_str == "Customized"
        assert _delta.new_val == 10.5

    def test_kwargs_custom_params(self, tmp_path: Path) -> None:
        """Try to specify one custom parameter via kwargs."""
        file_name = "user_parameters.yaml"
        p = utilities.yaml_from_dict(tmp_path, file_name)
        # initialize model
        _delta = CustomParamModel(input_file=p, new_val=-5.25)
        # assert that hook has been used and default params exist
        assert _delta.subclass_parameters["new_str"]["default"] == "DefaultString"
        assert _delta.subclass_parameters["new_val"]["default"] == 0
        assert hasattr(_delta, "subclass_parameters")
        assert _delta.new_str == "DefaultString"
        assert _delta.new_val == -5.25

    def test_invalid_custom_params(self, tmp_path: Path) -> None:
        """Specify invalid custom param type in yaml."""
        file_name = "user_parameters.yaml"
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, "out_dir", tmp_path / "out_dir")
        utilities.write_parameter_to_file(f, "new_val", "invalid_str")
        f.close()
        # initialize model
        with pytest.raises(TypeError):
            _ = CustomParamModel(input_file=p)


from pyDeltaRCM import preprocessor
from netCDF4 import Dataset
import os


class TestConsistentOutputsSameSeed:
    def test_same_models_diff_save_dt_singlesave(self, tmp_path: Path) -> None:
        """Test models that have same parameters but different save_dt.

        Several parts of the outputs here are expected to NOT be the same,
        because the second model is only saved one time (intiial time).
        """
        file_name = "user_parameters.yaml"
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, "out_dir", tmp_path / "out_dir")
        utilities.write_parameter_to_file(f, "Length", 10.0)
        utilities.write_parameter_to_file(f, "Width", 20.0)
        utilities.write_parameter_to_file(f, "seed", 1)
        utilities.write_parameter_to_file(f, "verbose", 1)
        utilities.write_parameter_to_file(f, "dx", 1.0)
        utilities.write_parameter_to_file(f, "L0_meters", 1.0)
        utilities.write_parameter_to_file(f, "save_eta_grids", True)
        utilities.write_matrix_to_file(f, ["save_dt"], [[0, 50000]])
        f.close()

        # use preprocessor to run models
        pp = preprocessor.Preprocessor(input_file=p, timesteps=2)
        pp.run_jobs()

        # look at outputs
        ModelA = Dataset(
            os.path.join(str(pp._file_list[0])[:-12], "pyDeltaRCM_output.nc"),
            "r",
            format="NETCDF4",
        )
        ModelB = Dataset(
            os.path.join(str(pp._file_list[1])[:-12], "pyDeltaRCM_output.nc"),
            "r",
            format="NETCDF4",
        )

        # check attributes of the netCDFs (NOTE THESE ARE NOT THE ACTUAL MODELS!)
        # final eta grid has same LxW shape
        assert ModelA["eta"][-1, :, :].shape == ModelB["eta"][-1, :, :].shape
        assert ModelA.variables.keys() == ModelB.variables.keys()
        # check a few pieces of metadata
        assert ModelA["meta"]["L0"][:] == ModelB["meta"]["L0"][:]
        assert ModelA["meta"].variables.keys() == ModelB["meta"].variables.keys()
        # final time should NOT be the same (because only initial time is saved in ModelB)
        assert ModelA["time"][-1].data > ModelB["time"][-1].data
        # final eta grids should NOT be the same (because only initial eta is saved in ModelB)
        assert np.any(ModelA["eta"][-1, :, :].data != ModelB["eta"][-1, :, :].data)
        # this is seen in the difference in shape of the eta variable
        assert ModelA["eta"].shape != ModelB["eta"].shape
        # first eta grids should be the same (because both initial eta is the same)
        assert np.all(ModelA["eta"][0, :, :].data == ModelB["eta"][0, :, :].data)
        # close netCDF output files
        ModelA.close()
        ModelB.close()

    def test_same_models_diff_save_dt_saveend(self, tmp_path: Path) -> None:
        """Test models that have same parameters but different save_dt.

        All outputs are expected to be the same because they have the same
        number of saves.
        """
        file_name = "user_parameters.yaml"
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, "out_dir", tmp_path / "out_dir")
        utilities.write_parameter_to_file(f, "Length", 10.0)
        utilities.write_parameter_to_file(f, "Width", 20.0)
        utilities.write_parameter_to_file(f, "seed", 1)
        utilities.write_parameter_to_file(f, "verbose", 2)
        utilities.write_parameter_to_file(f, "dx", 1.0)
        utilities.write_parameter_to_file(f, "L0_meters", 1.0)
        utilities.write_parameter_to_file(f, "save_eta_grids", True)
        utilities.write_matrix_to_file(f, ["save_dt"], [[0, 2]])
        f.close()

        # use preprocessor to run models
        pp = preprocessor.Preprocessor(input_file=p, timesteps=5)
        pp.run_jobs()

        # look at outputs
        ModelA = Dataset(
            os.path.join(str(pp._file_list[0])[:-12], "pyDeltaRCM_output.nc"),
            "r",
            format="NETCDF4",
        )
        ModelB = Dataset(
            os.path.join(str(pp._file_list[1])[:-12], "pyDeltaRCM_output.nc"),
            "r",
            format="NETCDF4",
        )

        # check attributes of the netCDFs
        # final eta grid has same LxW shape
        assert ModelA["eta"][-1, :, :].shape == ModelB["eta"][-1, :, :].shape
        assert ModelA["eta"].shape == ModelB["eta"].shape
        assert ModelA.variables.keys() == ModelB.variables.keys()
        # check a few pieces of metadata
        assert ModelA["meta"]["L0"][:] == ModelB["meta"]["L0"][:]
        assert ModelA["meta"].variables.keys() == ModelB["meta"].variables.keys()
        # final time should be the same
        assert ModelA["time"][-1].data == ModelB["time"][-1].data
        # final eta grids should be the same (because both initial eta is save in ModelB)
        assert np.all(ModelA["eta"][-1, :, :].data == ModelB["eta"][-1, :, :].data)
        # first eta grids should be the same (because both initial eta is the same)
        assert np.all(ModelA["eta"][0, :, :].data == ModelB["eta"][0, :, :].data)
        # close netCDF output files
        ModelA.close()
        ModelB.close()

    @pytest.mark.skipif(
        platform.system() != "Linux", reason="Parallel support only on Linux OS."
    )
    def test_same_models_in_serial_or_parallel(self, tmp_path: Path) -> None:
        """Test models that have same parameters but different save_dt."""
        file_name = "user_parameters_ser.yaml"
        pser, f = utilities.create_temporary_file(tmp_path / "ser", file_name)
        utilities.write_parameter_to_file(f, "out_dir", tmp_path / "ser" / "out_dir")
        utilities.write_parameter_to_file(f, "Length", 10.0)
        utilities.write_parameter_to_file(f, "Width", 20.0)
        utilities.write_parameter_to_file(f, "seed", 1)
        utilities.write_parameter_to_file(f, "verbose", 2)
        utilities.write_parameter_to_file(f, "dx", 1.0)
        utilities.write_parameter_to_file(f, "L0_meters", 1.0)
        utilities.write_parameter_to_file(f, "save_eta_grids", True)
        utilities.write_parameter_to_file(f, "parallel", False)
        utilities.write_matrix_to_file(f, ["save_dt"], [[0, 2]])
        f.close()

        file_name = "user_parameters_par.yaml"
        ppar, f = utilities.create_temporary_file(tmp_path / "par", file_name)
        utilities.write_parameter_to_file(f, "out_dir", tmp_path / "par" / "out_dir")
        utilities.write_parameter_to_file(f, "Length", 10.0)
        utilities.write_parameter_to_file(f, "Width", 20.0)
        utilities.write_parameter_to_file(f, "seed", 1)
        utilities.write_parameter_to_file(f, "verbose", 2)
        utilities.write_parameter_to_file(f, "dx", 1.0)
        utilities.write_parameter_to_file(f, "L0_meters", 1.0)
        utilities.write_parameter_to_file(f, "save_eta_grids", True)
        utilities.write_parameter_to_file(f, "parallel", True)
        utilities.write_matrix_to_file(f, ["save_dt"], [[0, 2]])
        f.close()

        # use preprocessor to run models
        ppser = preprocessor.Preprocessor(input_file=pser, timesteps=5)
        ppser.run_jobs()
        pppar = preprocessor.Preprocessor(input_file=ppar, timesteps=5)
        pppar.run_jobs()

        # look at outputs
        ModelA_ser = Dataset(
            os.path.join(str(ppser._file_list[0])[:-12], "pyDeltaRCM_output.nc"),
            "r",
            format="NETCDF4",
        )
        ModelB_ser = Dataset(
            os.path.join(str(ppser._file_list[1])[:-12], "pyDeltaRCM_output.nc"),
            "r",
            format="NETCDF4",
        )
        ModelA_par = Dataset(
            os.path.join(str(pppar._file_list[0])[:-12], "pyDeltaRCM_output.nc"),
            "r",
            format="NETCDF4",
        )
        ModelB_par = Dataset(
            os.path.join(str(pppar._file_list[1])[:-12], "pyDeltaRCM_output.nc"),
            "r",
            format="NETCDF4",
        )

        # check attributes of the netCDFs
        # final eta grid has same LxW shape
        assert ModelA_ser["eta"][-1, :, :].shape == ModelA_par["eta"][-1, :, :].shape
        assert ModelA_ser["eta"][-1, :, :].shape == ModelB_par["eta"][-1, :, :].shape

        # final time should be the same
        assert ModelA_ser["time"][-1].data == ModelA_par["time"][-1].data
        assert ModelB_par["time"][-1].data == ModelA_par["time"][-1].data
        assert ModelB_par["time"][-1].data == ModelA_ser["time"][-1].data
        # final eta grids should be the same (because both initial eta is save in ModelB)
        assert np.all(
            ModelA_ser["eta"][-1, :, :].data == ModelA_par["eta"][-1, :, :].data
        )
        assert np.all(
            ModelA_ser["eta"][-1, :, :].data == ModelB_par["eta"][-1, :, :].data
        )
        # close netCDF output files
        ModelA_par.close()
        ModelB_par.close()
        ModelA_ser.close()
        ModelB_ser.close()
