import pytest
import os
import platform
import locale
import subprocess

from .. import utilities
import pyDeltaRCM as _pyimportedalias


class TestCommandLineInterfaceDirectly:

    # test the entry points
    def test_entry_point_installed_call(self, tmp_path):
        """
        test calling the command line feature with a config file.
        """
        file_name = 'user_parameters.yaml'
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, 'Length', 10.0)
        utilities.write_parameter_to_file(f, 'Width', 10.0)
        utilities.write_parameter_to_file(f, 'seed', 0)
        utilities.write_parameter_to_file(f, 'dx', 1.0)
        utilities.write_parameter_to_file(f, 'L0_meters', 1.0)
        utilities.write_parameter_to_file(f, 'N0_meters', 2.0)
        utilities.write_parameter_to_file(f, 'Np_water', 10)
        utilities.write_parameter_to_file(f, 'Np_sed', 10)
        utilities.write_parameter_to_file(f, 'timesteps', 2)
        utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'test')
        utilities.write_parameter_to_file(f, 'save_dt', 300)
        utilities.write_parameter_to_file(f, 'save_eta_figs', True)
        utilities.write_parameter_to_file(f, 'save_eta_grids', True)
        f.close()
        subprocess.check_output(['pyDeltaRCM',
                                 '--config', str(p)])
        exp_path_nc = os.path.join(tmp_path / 'test', 'pyDeltaRCM_output.nc')
        exp_path_png0 = os.path.join(tmp_path / 'test', 'eta_00000.png')
        exp_path_png1 = os.path.join(tmp_path / 'test', 'eta_00001.png')
        assert os.path.isfile(exp_path_nc)
        assert os.path.isfile(exp_path_png0)
        assert os.path.isfile(exp_path_png1)

    def test_entry_point_python_main_call(self, tmp_path):
        """
        test calling the python hook command line feature with a config file.
        """
        file_name = 'user_parameters.yaml'
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, 'Length', 10.0)
        utilities.write_parameter_to_file(f, 'Width', 10.0)
        utilities.write_parameter_to_file(f, 'seed', 0)
        utilities.write_parameter_to_file(f, 'dx', 1.0)
        utilities.write_parameter_to_file(f, 'L0_meters', 1.0)
        utilities.write_parameter_to_file(f, 'N0_meters', 2.0)
        utilities.write_parameter_to_file(f, 'Np_water', 10)
        utilities.write_parameter_to_file(f, 'Np_sed', 10)
        utilities.write_parameter_to_file(f, 'timesteps', 1)
        utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'test')
        utilities.write_parameter_to_file(f, 'save_dt', 300)
        utilities.write_parameter_to_file(f, 'save_eta_figs', True)
        utilities.write_parameter_to_file(f, 'save_eta_grids', True)
        f.close()
        subprocess.check_output(['python', '-m', 'pyDeltaRCM',
                                 '--config', str(p)])
        exp_path_nc = os.path.join(tmp_path / 'test', 'pyDeltaRCM_output.nc')
        exp_path_png = os.path.join(tmp_path / 'test', 'eta_00000.png')
        exp_path_png1 = os.path.join(tmp_path / 'test', 'eta_00001.png')
        exp_path_png2 = os.path.join(tmp_path / 'test', 'eta_00002.png')
        assert os.path.isfile(exp_path_nc)
        assert os.path.isfile(exp_path_png)
        assert os.path.isfile(exp_path_png1)
        assert not os.path.isfile(exp_path_png2)

    def test_entry_point_python_main_call_dryrun(self, tmp_path):
        file_name = 'user_parameters.yaml'
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, 'Length', 10.0)
        utilities.write_parameter_to_file(f, 'Width', 10.0)
        utilities.write_parameter_to_file(f, 'seed', 0)
        utilities.write_parameter_to_file(f, 'dx', 1.0)
        utilities.write_parameter_to_file(f, 'L0_meters', 1.0)
        utilities.write_parameter_to_file(f, 'N0_meters', 2.0)
        utilities.write_parameter_to_file(f, 'Np_water', 10)
        utilities.write_parameter_to_file(f, 'Np_sed', 10)
        utilities.write_parameter_to_file(f, 'timesteps', 1)
        utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'test')
        f.close()
        subprocess.check_output(['python', '-m', 'pyDeltaRCM',
                                 '--config', str(p),
                                 '--dryrun'])
        exp_path_nc = os.path.join(tmp_path / 'test', 'pyDeltaRCM_output.nc')
        exp_path_png = os.path.join(tmp_path / 'test', 'eta_00000.png')
        assert not os.path.isfile(exp_path_nc)   # does not exist because --dryrun
        assert not os.path.isfile(exp_path_png)  # does not exist because --dryrun

    def test_entry_point_python_main_call_timesteps(self, tmp_path):
        file_name = 'user_parameters.yaml'
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, 'Length', 10.0)
        utilities.write_parameter_to_file(f, 'Width', 10.0)
        utilities.write_parameter_to_file(f, 'seed', 0)
        utilities.write_parameter_to_file(f, 'dx', 1.0)
        utilities.write_parameter_to_file(f, 'L0_meters', 1.0)
        utilities.write_parameter_to_file(f, 'N0_meters', 2.0)
        utilities.write_parameter_to_file(f, 'Np_water', 10)
        utilities.write_parameter_to_file(f, 'Np_sed', 10)
        utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'test')
        utilities.write_parameter_to_file(f, 'save_dt', 300)
        utilities.write_parameter_to_file(f, 'save_eta_figs', True)
        utilities.write_parameter_to_file(f, 'save_eta_grids', True)
        f.close()
        subprocess.check_output(['python', '-m', 'pyDeltaRCM',
                                 '--config', str(p),
                                 '--timesteps', '2'])
        exp_path_nc = os.path.join(tmp_path / 'test', 'pyDeltaRCM_output.nc')
        exp_path_png = os.path.join(tmp_path / 'test', 'eta_00000.png')
        assert os.path.isfile(exp_path_nc)
        assert os.path.isfile(exp_path_png)

    def test_error_if_no_timesteps(self, tmp_path):
        file_name = 'user_parameters.yaml'
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'test')
        f.close()
        with pytest.raises(subprocess.CalledProcessError):
            subprocess.check_output(['python', '-m', 'pyDeltaRCM',
                                     '--config', str(p)])

    def test_entry_point_timesteps(self, tmp_path):
        """
        test calling the command line feature with a config file.
        """
        file_name = 'user_parameters.yaml'
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, 'Length', 10.0)
        utilities.write_parameter_to_file(f, 'Width', 10.0)
        utilities.write_parameter_to_file(f, 'seed', 0)
        utilities.write_parameter_to_file(f, 'dx', 1.0)
        utilities.write_parameter_to_file(f, 'L0_meters', 1.0)
        utilities.write_parameter_to_file(f, 'N0_meters', 2.0)
        utilities.write_parameter_to_file(f, 'Np_water', 10)
        utilities.write_parameter_to_file(f, 'Np_sed', 10)
        utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'test')
        utilities.write_parameter_to_file(f, 'save_dt', 300)
        utilities.write_parameter_to_file(f, 'save_eta_figs', True)
        utilities.write_parameter_to_file(f, 'save_eta_grids', True)
        f.close()
        subprocess.check_output(['pyDeltaRCM',
                                 '--config', str(p), '--timesteps', '2'])
        exp_path_nc = os.path.join(tmp_path / 'test', 'pyDeltaRCM_output.nc')
        exp_path_png0 = os.path.join(tmp_path / 'test', 'eta_00000.png')
        exp_path_png1 = os.path.join(tmp_path / 'test', 'eta_00001.png')
        assert os.path.isfile(exp_path_nc)
        assert os.path.isfile(exp_path_png0)
        assert os.path.isfile(exp_path_png1)

    def test_entry_point_time(self, tmp_path):
        """
        test calling the command line feature with a config file.
        """
        file_name = 'user_parameters.yaml'
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, 'Length', 10.0)
        utilities.write_parameter_to_file(f, 'Width', 10.0)
        utilities.write_parameter_to_file(f, 'seed', 0)
        utilities.write_parameter_to_file(f, 'dx', 1.0)
        utilities.write_parameter_to_file(f, 'L0_meters', 1.0)
        utilities.write_parameter_to_file(f, 'N0_meters', 2.0)
        utilities.write_parameter_to_file(f, 'Np_water', 10)
        utilities.write_parameter_to_file(f, 'Np_sed', 10)
        utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'test')
        utilities.write_parameter_to_file(f, 'save_dt', 300)
        utilities.write_parameter_to_file(f, 'save_eta_figs', True)
        utilities.write_parameter_to_file(f, 'save_eta_grids', True)
        f.close()
        subprocess.check_output(['pyDeltaRCM',
                                 '--config', str(p), '--time', '1000'])
        exp_path_nc = os.path.join(tmp_path / 'test', 'pyDeltaRCM_output.nc')
        exp_path_png0 = os.path.join(tmp_path / 'test', 'eta_00000.png')
        exp_path_png1 = os.path.join(tmp_path / 'test', 'eta_00001.png')
        assert os.path.isfile(exp_path_nc)
        assert os.path.isfile(exp_path_png0)
        assert os.path.isfile(exp_path_png1)

    def test_version_call(self):
        """
        test calling the command line feature to query the version.
        """
        encoding = locale.getpreferredencoding()
        printed1 = subprocess.run(['pyDeltaRCM', '--version'],
                                  stdout=subprocess.PIPE, encoding=encoding)
        _exp_str1 = 'pyDeltaRCM ' + _pyimportedalias.__version__ + '\n'
        assert printed1.stdout == _exp_str1
        printed2 = subprocess.run(
            ['python', '-m', 'pyDeltaRCM', '--version'],
            stdout=subprocess.PIPE, encoding=encoding)
        _exp_str2 = 'pyDeltaRCM ' + _pyimportedalias.__version__ + '\n'
        assert printed2.stdout == _exp_str2
