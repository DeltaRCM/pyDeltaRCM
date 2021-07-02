
import pytest

import platform
import os

import unittest.mock as mock

import pyDeltaRCM
from pyDeltaRCM.model import DeltaModel
# from pyDeltaRCM import shared_tools
from pyDeltaRCM import preprocessor

from . import utilities
# from .utilities import test_DeltaModel


class TestPreprocessorSingleJobSetups:

    # test high level python api
    def test_py_hlvl_wo_args(self):
        with pytest.raises(ValueError):
            _ = preprocessor.Preprocessor()

    def test_py_hlvl_runjobs_simple_param(self, tmp_path):
        # a single parameter
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'h0': 7.5})
        pp = preprocessor.Preprocessor(p)

        assert len(pp.file_list) == 1
        assert pp._is_completed is False

        # a parameter that takes null as the default
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'hb': 7.5})
        pp = preprocessor.Preprocessor(p)

        assert len(pp.file_list) == 1
        assert pp._is_completed is False

        # a parameter that takes null with null
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'hb': None})
        pp = preprocessor.Preprocessor(p)

        assert len(pp.file_list) == 1
        assert pp._is_completed is False

    def test_py_hlvl_tsteps_yml_runjobs_sngle(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'timesteps': 50})
        pp = preprocessor.Preprocessor(p)

        assert len(pp.file_list) == 1
        assert pp._is_completed is False

        assert pp.config_dict['timesteps'] == 50
        assert not ('time' in pp.config_dict.keys())
        assert not ('time_years' in pp.config_dict.keys())

    def test_py_hlvl_time_yml_runjobs_sngle(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'time': 1000})
        pp = preprocessor.Preprocessor(p)

        assert len(pp.file_list) == 1
        assert pp._is_completed is False

        assert pp.config_dict['time'] == 1000
        assert not ('timesteps' in pp.config_dict.keys())

    def test_py_hlvl_time_If_yml_runjobs_sngle(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'time': 10000,
                                      'If': 0.1})
        pp = preprocessor.Preprocessor(p)

        assert len(pp.file_list) == 1
        assert pp._is_completed is False

        assert pp.config_dict['time'] == 10000
        assert pp.config_dict['If'] == 0.1
        assert not ('timesteps' in pp.config_dict.keys())

    def test_py_hlvl_timeyears_yml_runjobs_sngle(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'time_years': 3.16880878140e-05})
        pp = preprocessor.Preprocessor(p)

        assert len(pp.file_list) == 1
        assert pp._is_completed is False

        assert pp.config_dict['time_years'] == pytest.approx(3.1688087e-05)
        assert not ('If' in pp.config_dict.keys())
        assert not ('time' in pp.config_dict.keys())
        assert not ('timesteps' in pp.config_dict.keys())

    def test_py_hlvl_timeyears_If_yml_runjobs_sngle(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'time_years': 0.00031688087814,
                                      'If': 0.1})
        pp = preprocessor.Preprocessor(p)

        assert len(pp.file_list) == 1
        assert pp._is_completed is False

        assert pp.config_dict['time_years'] == pytest.approx(.00031688087814)
        assert pp.config_dict['If'] == 0.1
        assert not ('time' in pp.config_dict.keys())
        assert not ('timesteps' in pp.config_dict.keys())

    def test_py_hlvl_timesteps_and_yaml_argument(self, tmp_path):
        """
        Test that timesteps can come from cli, but others from yaml
        """
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'save_eta_figs': True})
        pp = preprocessor.Preprocessor(input_file=p, timesteps=2)

        assert type(pp.file_list) is list
        assert len(pp.file_list) == 1
        assert pp._is_completed is False

        assert pp.config_dict['timesteps'] == 2
        assert (pp.config_dict['save_eta_figs'] is True)
        assert not ('time' in pp.config_dict.keys())

    def test_py_hlvl_timesteps_yaml_and_cli(self, tmp_path):
        """
        Test preference for command line argument
        """
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'save_eta_figs': True,
                                      'timesteps': 20})
        pp = preprocessor.Preprocessor(input_file=p, timesteps=13)

        assert type(pp.file_list) is list
        assert len(pp.file_list) == 1
        assert pp._is_completed is False

        assert pp.config_dict['timesteps'] == 13
        assert (pp.config_dict['save_eta_figs'] is True)
        assert not ('time' in pp.config_dict.keys())

    def test_py_hlvl_timesteps_and_time_yaml_and_cli(self, tmp_path):
        """
        Test that preference is given to command line arguments

        note that time and timesteps both persist, precedence for these args
        is determined at runtime of run_jobs()
        """
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml',
                                     {'save_eta_figs': True,
                                      'timesteps': 20,
                                      'time': 1000})
        pp = preprocessor.Preprocessor(input_file=p, timesteps=13, time=13000)

        assert type(pp.file_list) is list
        assert len(pp.file_list) == 1
        assert pp._is_completed is False

        assert pp.config_dict['timesteps'] == 13
        assert pp.config_dict['time'] == 13000
        assert (pp.config_dict['save_eta_figs'] is True)
        assert ('time' in pp.config_dict.keys())
        assert ('timesteps' in pp.config_dict.keys())


class TestPreprocessorMatrixJobsSetups:

    def test_py_hlvl_mtrx_1list_timesteps_arg(self, tmp_path):
        file_name = 'user_parameters.yaml'
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'test')
        utilities.write_matrix_to_file(f,
                                       ['f_bedload'],
                                       [[0.2, 0.6]])
        f.close()
        pp = preprocessor.Preprocessor(input_file=p, timesteps=3)

        assert pp._has_matrix is True
        assert type(pp.file_list) is list
        assert len(pp.file_list) == 2
        assert pp._is_completed is False

        f_bedload_list = pp._matrix_table[:, 1]

        assert sum([j == 0.2 for j in f_bedload_list]) == 1
        assert sum([j == 0.6 for j in f_bedload_list]) == 1
        assert pp.config_dict['timesteps'] == 3

    def test_py_hlvl_mtrx_1list_tsteps_cfg(self, tmp_path):
        file_name = 'user_parameters.yaml'
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'test')
        utilities.write_parameter_to_file(f, 'timesteps', 3)
        utilities.write_matrix_to_file(f,
                                       ['f_bedload'],
                                       [[0.2, 0.6]])
        f.close()
        pp = preprocessor.Preprocessor(input_file=p)

        assert pp._has_matrix is True
        assert type(pp.file_list) is list
        assert len(pp.file_list) == 2
        assert pp._is_completed is False

        f_bedload_list = pp._matrix_table[:, 1]

        assert sum([j == 0.2 for j in f_bedload_list]) == 1
        assert sum([j == 0.6 for j in f_bedload_list]) == 1
        assert pp.config_dict['timesteps'] == 3

    def test_py_hlvl_mtrx_smthng_null(self, tmp_path):
        file_name = 'user_parameters.yaml'
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'test')
        utilities.write_parameter_to_file(f, 'timesteps', 3)
        utilities.write_matrix_to_file(f,
                                       ['hb'],
                                       [[None, 2, 7]])
        f.close()
        pp = preprocessor.Preprocessor(input_file=p)

        assert pp._has_matrix is True
        assert type(pp.file_list) is list
        assert len(pp.file_list) == 3
        assert pp._is_completed is False

        hb_list = pp._matrix_table[:, 1]

        assert sum([j == 'None' for j in hb_list]) == 1
        assert sum([j == 2 for j in hb_list]) == 1
        assert sum([j == 7 for j in hb_list]) == 1

    def test_py_hlvl_mtrx_2list(self, tmp_path):
        file_name = 'user_parameters.yaml'
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'test')
        utilities.write_matrix_to_file(f,
                                       ['f_bedload', 'u0'],
                                       [[0.2, 0.5, 0.6], [1.0, 1.5, 2.0]])
        f.close()
        pp = preprocessor.Preprocessor(input_file=p)

        assert pp._has_matrix is True
        assert type(pp.file_list) is list
        assert len(pp.file_list) == 9
        assert pp._is_completed is False

        f_bedload_list = pp._matrix_table[:, 1]
        u0_list = pp._matrix_table[:, 2]

        assert sum([j == 0.2 for j in f_bedload_list]) == 3
        assert sum([j == 0.5 for j in f_bedload_list]) == 3
        assert sum([j == 0.6 for j in f_bedload_list]) == 3
        assert sum([j == 1.0 for j in u0_list]) == 3
        assert sum([j == 1.5 for j in u0_list]) == 3
        assert sum([j == 2.0 for j in u0_list]) == 3

        comb_list = pp._matrix_table[:, 1:].tolist()
        assert [0.2, 2.0] in comb_list
        assert [0.5, 1.0] in comb_list
        assert not ([0.5, 0.2] in comb_list)

    def test_py_hlvl_mtrx_scientificnotation(self, tmp_path):
        """Test that preprocessor can read and write scinotation"""
        file_name = 'user_parameters.yaml'
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'test')
        utilities.write_matrix_to_file(f,
                                       ['f_bedload', 'SLR'],
                                       [[0.2, 0.5, 0.6], [0.00004, 1e-6]])
        f.close()

        # will convert everything to scientific notation
        pp = preprocessor.Preprocessor(input_file=p)

        SLR_list = pp._matrix_table[:, 2]
        assert sum([j == 4e-5 for j in SLR_list]) == 3
        assert sum([j == 0.000001 for j in SLR_list]) == 3

    def test_py_hlvL_mtrx_needs_out_dir(self, tmp_path):
        file_name = 'user_parameters.yaml'
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        # missing out_dir in the config will throw an error
        # => utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'test')
        utilities.write_matrix_to_file(f,
                                       ['f_bedload'],
                                       [[0.2, 0.5, 0.6]])
        f.close()
        with pytest.raises(ValueError,
                           match=r'You must specify "out_dir" in YAML .*'):
            _ = preprocessor.Preprocessor(input_file=p)

    def test_py_hlvl_mtrx_outdir_exists_error(self, tmp_path):
        file_name = 'user_parameters.yaml'
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'test')
        utilities.write_matrix_to_file(f,
                                       ['f_bedload'],
                                       [[0.2, 0.5, 0.6]])
        f.close()
        pp = preprocessor.Preprocessor(input_file=p)

        # misc assertions
        assert pp._has_matrix is True

        # verify the directory exists
        assert pp._jobs_root == str(tmp_path / 'test')
        assert (tmp_path / 'test').is_dir()

        # try to create another preprocessor at same directory
        with pytest.raises(FileExistsError,
                           match=r'Job output directory .*'):
            _ = preprocessor.Preprocessor(input_file=p)

    def test_py_hlvl_mtrx_bad_len1(self, tmp_path):
        file_name = 'user_parameters.yaml'
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'test')
        # bad configuration is a list of length 1
        utilities.write_matrix_to_file(f,
                                       ['f_bedload', 'u0'],
                                       [[0.2], [0.5, 0.6, 1.25]])
        f.close()
        with pytest.warns(UserWarning,
                          match=r'Length of matrix key "f_bedload" was 1,'):
            pp = preprocessor.Preprocessor(input_file=p)

        # check job length
        assert pp._has_matrix is True
        assert len(pp.file_list) == 3

    def test_py_hlvl_mtrx_bad_listinlist(self, tmp_path):
        file_name = 'user_parameters.yaml'
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'test')
        # bad configuration will lead to error
        utilities.write_matrix_to_file(f,
                                       ['f_bedload', 'u0'],
                                       [[0.2, [0.5, 0.6]], [0.5, 1.25]])
        f.close()
        with pytest.raises(ValueError,
                           match=r'Depth of matrix expansion must not be > 1'):
            _ = preprocessor.Preprocessor(input_file=p)

    def test_py_hlvl_mtrx_bad_samekey(self, tmp_path):
        file_name = 'user_parameters.yaml'
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, 'f_bedload', 0.3)
        utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'test')
        # bad configuration will lead to error
        utilities.write_matrix_to_file(f,
                                       ['f_bedload', 'u0'],
                                       [[0.2, 0.5, 0.6], [0.5, 1.25]])
        f.close()
        with pytest.raises(ValueError,
                           match=r'You cannot specify the same key '
                                  'in the matrix .*'):  # noqa: E127
            _ = preprocessor.Preprocessor(input_file=p)

    def test_py_hlvl_mtrx_bad_colon(self, tmp_path):
        file_name = 'user_parameters.yaml'
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'test')
        # bad configuration will lead to error
        utilities.write_matrix_to_file(f,
                                       ['f_bedload', 'u0:'],
                                       [[0.2, 0.5], [0.5, 1.25]])
        f.close()
        with pytest.raises(ValueError,
                           match=r'Colon operator found '
                                  'in matrix expansion key.'):  # noqa: E127
            _ = preprocessor.Preprocessor(input_file=p)

    def test_py_hlvl_mtrx_no_out_dir_in_mtrx(self, tmp_path):
        file_name = 'user_parameters.yaml'
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'test')
        utilities.write_matrix_to_file(f,
                                       ['out_dir', 'f_bedload'],
                                       [['dir1', 'dir2'], [0.2, 0.5, 0.6]])
        f.close()
        with pytest.raises(ValueError,
                           match=r'You cannot specify "out_dir" as .*'):
            _ = preprocessor.Preprocessor(input_file=p, timesteps=3)

    def test_python_highlevelapi_matrix_verbosity(self, tmp_path, capsys):
        file_name = 'user_parameters.yaml'
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, 'verbose', 1)
        utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'test')
        utilities.write_matrix_to_file(f,
                                       ['f_bedload', 'u0'],
                                       [[0.2, 0.5, 0.6], [1.5, 2.0]])
        f.close()
        _ = preprocessor.Preprocessor(input_file=p, timesteps=3)

        captd = capsys.readouterr()
        assert 'Timestep: 0.0' not in captd.out
        assert 'Writing YAML file for job 0' in captd.out
        assert 'Writing YAML file for job 1' in captd.out
        assert 'Writing YAML file for job 2' in captd.out
        assert 'Writing YAML file for job 3' in captd.out
        assert 'Writing YAML file for job 4' in captd.out
        assert 'Writing YAML file for job 5' in captd.out
        assert 'Matrix expansion:' in captd.out
        assert '  dims 2' in captd.out
        assert '  jobs 6' in captd.out


class TestPreprocessorSetJobsSetups:

    def test_py_hlvl_set_two_sets(self, tmp_path):
        file_name = 'user_parameters.yaml'
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'test')
        utilities.write_set_to_file(f, [{'f_bedload': 0.2, 'u0': 1},
                                        {'f_bedload': 0.4, 'u0': 2}])
        f.close()
        pp = preprocessor.Preprocessor(input_file=p, timesteps=3)

        assert pp._has_set is True
        assert pp._has_matrix is False
        assert type(pp.file_list) is list
        assert len(pp.file_list) == 2
        assert pp._is_completed is False

        f_bedload_list = pp._matrix_table[:, 1]

        assert sum([j == 0.2 for j in f_bedload_list]) == 1
        assert sum([j == 0.4 for j in f_bedload_list]) == 1
        assert pp.config_dict['timesteps'] == 3

    def test_py_hlvl_set_one_sets(self, tmp_path):
        file_name = 'user_parameters.yaml'
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'test')
        utilities.write_set_to_file(f, [{'f_bedload': 0.77, 'u0': 1}])
        f.close()
        pp = preprocessor.Preprocessor(input_file=p, timesteps=3)

        assert pp._has_set is True
        assert pp._has_matrix is False
        assert type(pp.file_list) is list
        assert len(pp.file_list) == 1
        assert pp._is_completed is False

        f_bedload_list = pp._matrix_table[:, 1]

        assert sum([j == 0.77 for j in f_bedload_list]) == 1
        assert pp.config_dict['timesteps'] == 3

    def test_py_hlvl_set_two_mismatched_sets(self, tmp_path):
        file_name = 'user_parameters.yaml'
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'test')
        utilities.write_set_to_file(f, [{'f_bedload': 0.2, 'u0': 1},
                                        {'h0': 0.4, 'u0': 2}])
        f.close()
        with pytest.raises(ValueError,
                           match=r'All keys in all sets *.'):  # noqa: E127
            _ = preprocessor.Preprocessor(input_file=p)

    def test_py_hlvl_set_bad_not_list(self, tmp_path):
        file_name = 'user_parameters.yaml'
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'test')
        # utilities.write_set_to_file(f, [{'f_bedload': 0.77, 'u0': 1}])
        utilities.write_parameter_to_file(f, 'set', 'astring!')
        f.close()
        with pytest.raises(TypeError,
                           match=r'Set list must be *.'):
            _ = preprocessor.Preprocessor(input_file=p)

    def test_py_hlvl_set_bad_colon(self, tmp_path):
        file_name = 'user_parameters.yaml'
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'test')
        utilities.write_set_to_file(f, [{'f_bedload:': 0.77, 'u0': 1}])
        f.close()
        with pytest.raises(ValueError,
                           match=r'Colon operator found '
                                  'in matrix expansion key.'):  # noqa: E127
            _ = preprocessor.Preprocessor(input_file=p)

    def test_output_table_correctly(self, tmp_path):
        file_name = 'user_parameters.yaml'
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'test')
        utilities.write_set_to_file(
            f, [{'f_bedload': 0.77, 'u0': 1, 'C0_percent': 0.033}])
        f.close()
        pp = preprocessor.Preprocessor(input_file=p, timesteps=3)

        # now check what the file looks like
        with open(os.path.join(
                tmp_path, 'test', 'jobs_parameters.txt'), 'r') as tbl:
            Lines = tbl.readlines()

        assert Lines[0] == 'job_id, f_bedload, u0, C0_percent\n'


class TestPreprocessorEnsembleJobsSetups:

    def test_py_hlvl_ensemble(self, tmp_path):
        file_name = 'user_parameters.yaml'
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, 'ensemble', 2)
        utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'test')
        f.close()
        pp = preprocessor.Preprocessor(input_file=p)
        # assertions for job creation
        assert pp._has_matrix is True
        assert pp._has_ensemble is True
        assert type(pp.file_list) is list
        assert len(pp.file_list) == 2

    def test_py_hlvl_ensemble_1(self, tmp_path):
        file_name = 'user_parameters.yaml'
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, 'ensemble', 1)
        utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'test')
        f.close()
        with pytest.warns(UserWarning,
                          match=r'Ensemble was set to 1. *.'):
            pp = preprocessor.Preprocessor(input_file=p)

        # check that keys were reset and config set correctly
        assert pp._has_ensemble is False
        assert pp._has_matrix is False
        assert len(pp.file_list) == 1

    def test_py_hlvl_ensemble_badtype(self, tmp_path):
        file_name = 'user_parameters.yaml'
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, 'ensemble', 2.0)
        utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'test')
        f.close()
        with pytest.raises(TypeError,
                           match=r'Invalid ensemble type, '
                                  'must be an integer.'):  # noqa: E127
            _ = preprocessor.Preprocessor(input_file=p)

    def test_py_hlvl_ensemble_double_seeds(self, tmp_path):
        file_name = 'user_parameters.yaml'
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, 'ensemble', 2)
        utilities.write_parameter_to_file(f, 'seed', 1)
        utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'test')
        f.close()
        with pytest.raises(ValueError,
                           match=r'You cannot specify the same key in the '
                                  'matrix configuration and '  # noqa: E127
                                  'fixed configuration. Key "seed" was '
                                  'specified in both.'):
            _ = preprocessor.Preprocessor(input_file=p)


class TestPreprocessorMatrixAndEnsembleJobsSetups:

    def test_py_hlvl_ensemble_with_matrix(self, tmp_path):
        file_name = 'user_parameters.yaml'
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, 'ensemble', 2)
        utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'test')
        utilities.write_matrix_to_file(f,
                                       ['f_bedload'],
                                       [[0.2, 0.5, 0.6]])
        f.close()
        pp = preprocessor.Preprocessor(input_file=p)
        # assertions for job creation
        assert pp._has_matrix is True
        assert type(pp.file_list) is list
        assert len(pp.file_list) == 6

    def test_py_hlvl_ensemble_matrix_seeds(self, tmp_path):
        file_name = 'user_parameters.yaml'
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, 'ensemble', 2)
        utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'test')
        utilities.write_matrix_to_file(f,
                                       ['seed'],
                                       [[1, 2]])
        f.close()
        with pytest.raises(ValueError,
                           match=r'Random seeds cannot be specified in '
                                  'the matrix, if an "ensemble" number '  # noqa: E127, E501
                                  'is specified as well.'):
            _ = preprocessor.Preprocessor(input_file=p, timesteps=3)


class TestPreprocessorSetAndEnsembleJobsSetups:

    def test_1_set_ensemble_3(self, tmp_path):
        file_name = 'user_parameters.yaml'
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'test')
        utilities.write_parameter_to_file(f, 'ensemble', 3)
        utilities.write_set_to_file(f, [{'f_bedload': 0.77, 'u0': 1}])
        f.close()
        pp = preprocessor.Preprocessor(input_file=p, timesteps=3)

        assert pp._has_set is True
        assert pp._has_matrix is False
        assert pp._has_ensemble is True
        assert type(pp.file_list) is list
        assert len(pp.file_list) == 3
        assert pp._is_completed is False

        f_bedload_list = pp._matrix_table[:, 1]

        assert sum([j == 0.77 for j in f_bedload_list]) == 3
        assert pp.config_dict['timesteps'] == 3


class TestPreprocessorSetAndMatrixJobsSetups:

    def test_not_possible(self, tmp_path):
        file_name = 'user_parameters.yaml'
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'test')
        utilities.write_set_to_file(f, [{'f_bedload': 0.77, 'h0': 1}])
        utilities.write_matrix_to_file(f,
                                       ['u0'],
                                       [[0.2, 0.5, 0.6]])
        f.close()

        with pytest.raises(ValueError,
                           match=r'Cannot use "matrix" and "set" .*'):
            _ = preprocessor.Preprocessor(input_file=p, timesteps=3)


class TestPreprocessorParallelJobsSetups:
    """
    Note that parallel job setup will work on any operating system. This is
    helpful for someone who wants to use the preprocessor to tinker with job
    configurations locally, but will run jobs on a remote Linux cluster/HPC.
    """

    def test_py_hlvl_parallel_works_onejob(self, tmp_path):
        file_name = 'user_parameters.yaml'
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'test')
        utilities.write_parameter_to_file(f, 'parallel', True)
        f.close()
        pp = preprocessor.Preprocessor(input_file=p)
        # assertions for job creation
        assert type(pp.file_list) is list
        assert len(pp.file_list) == 1
        assert pp._has_matrix is False

        assert pp.config_dict['parallel'] is True

    def test_py_hlvl_parallel_boolean_yaml(self, tmp_path):
        file_name = 'user_parameters.yaml'
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, 'ensemble', 2)
        utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'test')
        utilities.write_parameter_to_file(f, 'parallel', True)
        f.close()
        pp = preprocessor.Preprocessor(input_file=p)
        # assertions for job creation
        assert type(pp.file_list) is list
        assert len(pp.file_list) == 2
        assert pp._has_matrix is True

        assert pp.config_dict['parallel'] is True

    def test_py_hlvl_parallel_boolean_cli(self, tmp_path):
        file_name = 'user_parameters.yaml'
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, 'ensemble', 2)
        utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'test')
        f.close()
        pp = preprocessor.Preprocessor(input_file=p, parallel=True)
        # assertions for job creation
        assert type(pp.file_list) is list
        assert len(pp.file_list) == 2
        assert pp._has_matrix is True

        assert pp.config_dict['parallel'] is True

    def test_py_hlvl_parallel_integer_yaml(self, tmp_path):
        file_name = 'user_parameters.yaml'
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, 'ensemble', 2)
        utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'test')
        utilities.write_parameter_to_file(f, 'parallel', 2)
        f.close()
        pp = preprocessor.Preprocessor(input_file=p)
        # assertions for job creation
        assert type(pp.file_list) is list
        assert len(pp.file_list) == 2
        assert pp._has_matrix is True

        assert pp.config_dict['parallel'] == 2

    def test_py_hlvl_parallel_integer_cli(self, tmp_path):
        file_name = 'user_parameters.yaml'
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, 'ensemble', 2)
        utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'test')
        f.close()
        pp = preprocessor.Preprocessor(input_file=p, parallel=2)
        # assertions for job creation
        assert type(pp.file_list) is list
        assert len(pp.file_list) == 2
        assert pp._has_matrix is True
        assert pp._is_completed is False

        assert pp.config_dict['parallel'] == 2


class TestPreprocessorRunJobs:

    def test_dryrun(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml')
        pp = preprocessor.Preprocessor(p)

        pp._dryrun = True

        # run the method
        pp.run_jobs()

        assert pp._is_completed is False

    def test_run_single_serial_job(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml')
        pp = preprocessor.Preprocessor(p)

        # patch jobs
        with mock.patch('pyDeltaRCM.preprocessor._SerialJob') as ptch:

            # run the method
            pp.run_jobs()

        assert ptch.call_count == 1
        assert len(pp.job_list) == 1
        assert pp._is_completed is True

    def test_run_two_serial_jobs(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml')
        pp = preprocessor.Preprocessor(p)

        pp._file_list = [*pp.file_list] * 2
        pp._config_list = [*pp.config_list] * 2

        # patch jobs
        with mock.patch('pyDeltaRCM.preprocessor._SerialJob') as ptch:

            # run the method
            pp.run_jobs()

        assert ptch.call_count == 2
        assert len(pp.job_list) == 2
        assert pp._is_completed is True

    @pytest.mark.skipif(
        platform.system() != 'Linux',
        reason='Parallel support only on Linux OS.')
    def test_run_five_parallel_jobs_bool(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml')
        pp = preprocessor.Preprocessor(p, parallel=True)

        # expand the list to include multiples
        pp._file_list = [*pp.file_list] * 5
        pp._config_list = [*pp.config_list] * 5

        # patch jobs and semaphore to check number of processes called
        with mock.patch('pyDeltaRCM.preprocessor._ParallelJob') as ptch, \
             mock.patch('multiprocessing.Semaphore') as sem:

            # run the method
            pp.run_jobs()

        # assertions
        assert ptch.call_count == 5
        # can only assert sem was called:
        #   number called with will depend on number of cpu cores)
        sem.assert_called()
        assert len(pp.job_list) == 5
        assert pp._is_completed is True

    @pytest.mark.skipif(
        platform.system() != 'Linux',
        reason='Parallel support only on Linux OS.')
    def test_run_five_parallel_jobs_int_two(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml')
        pp = preprocessor.Preprocessor(p, parallel=2)

        # expand the list to include multiples
        pp._file_list = [*pp.file_list] * 5
        pp._config_list = [*pp.config_list] * 5

        # patch jobs and semaphore to check number of processes called
        with mock.patch('pyDeltaRCM.preprocessor._ParallelJob') as ptch, \
             mock.patch('multiprocessing.Semaphore') as sem:

            # run the method
            pp.run_jobs()

        # assertions
        assert ptch.call_count == 5
        sem.assert_called_with(2)
        assert len(pp.job_list) == 5
        assert pp._is_completed is True

    @pytest.mark.skipif(
        platform.system() != 'Linux',
        reason='Parallel support only on Linux OS.')
    def test_run_five_parallel_jobs_bad_types(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml')
        pp_float = preprocessor.Preprocessor(p, parallel=3.33)
        pp_str = preprocessor.Preprocessor(p, parallel='string!')

        # patch jobs and semaphore to check number of processes called
        with mock.patch('pyDeltaRCM.preprocessor._ParallelJob'), \
             mock.patch('multiprocessing.Semaphore'):

            with pytest.raises(ValueError, match=r'Parallel flag *.'):
                # run the method
                pp_float.run_jobs()

        with mock.patch('pyDeltaRCM.preprocessor._ParallelJob'), \
             mock.patch('multiprocessing.Semaphore'):

            with pytest.raises(ValueError, match=r'Parallel flag *.'):
                # run the method
                pp_str.run_jobs()

    @pytest.mark.skipif(
        platform.system() == 'Linux',
        reason='Parallel support only on Linux OS.')
    def test_run_parallel_notimplemented_nonlinux(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml')
        pp = preprocessor.Preprocessor(p, parallel=True)

        # patch jobs and semaphore to check number of processes called
        with pytest.raises(NotImplementedError):

            # run the method
            pp.run_jobs()


@mock.patch.multiple(pyDeltaRCM.preprocessor._BaseJob,
                     __abstractmethods__=set())
class TestBaseJob:
    """
    To test the BaseJob, we patch the base job with a filled abstract method
    `.run()`.

    .. note:: This patch is handled at the class level above!!
    """

    def test_wo_timesteps(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml')

        with pytest.raises(ValueError,
                           match=r'You must specify a run duration *.'):
            _ = preprocessor._BaseJob(
                i=0, input_file=p,
                config_dict={})

    def test_timeargs_precedence_tstepsovertime(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml')

        basej = preprocessor._BaseJob(
            i=0, input_file=p,
            config_dict={'time': 10000,
                         'timesteps': 2})

        assert isinstance(basej.deltamodel, DeltaModel)
        assert basej._job_end_time == basej.deltamodel._dt * 2
        assert basej._job_end_time != 10000

    def test_timeargs_precedence_tstepsovertimeyears(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml')

        basej = preprocessor._BaseJob(
            i=0, input_file=p,
            config_dict={'time_years': 10000,
                         'timesteps': 2})

        assert isinstance(basej.deltamodel, DeltaModel)
        assert basej._job_end_time == basej.deltamodel._dt * 2
        assert basej._job_end_time != 10000
        assert basej._job_end_time != 10000 * 365.25

    def test_timeargs_precedence_timeovertimeyears(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml')

        basej = preprocessor._BaseJob(
            i=0, input_file=p,
            config_dict={'time_years': 10000,
                         'time': 900})

        assert isinstance(basej.deltamodel, DeltaModel)
        assert basej._job_end_time == 900
        assert basej._job_end_time != 10000
        assert basej._job_end_time != 10000 * 365.25


class TestSerialJob:

    def test_run(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml')

        sj = preprocessor._SerialJob(
            i=0, input_file=p,
            config_dict={'timesteps': 10})

        # modify the save interval to be twice dt
        sj.deltamodel._save_dt = 2 * sj.deltamodel._dt
        sj.deltamodel._checkpoint_dt = 2 * sj.deltamodel._dt

        # mock top-level methods, verify call was made to each
        sj.deltamodel.solve_water_and_sediment_timestep = mock.MagicMock()
        sj.deltamodel.apply_subsidence = mock.MagicMock()
        sj.deltamodel.finalize_timestep = mock.MagicMock()
        sj.deltamodel.log_model_time = mock.MagicMock()
        sj.deltamodel.logger = mock.MagicMock()
        sj.deltamodel.output_data = mock.MagicMock()
        sj.deltamodel.output_checkpoint = mock.MagicMock()
        sj.deltamodel.finalize = mock.MagicMock()

        # run the method (call update 10 times)
        sj.run()

        # assertions
        assert sj.deltamodel._time_iter == 10
        assert sj.deltamodel.output_data.call_count == 10
        assert sj.deltamodel.solve_water_and_sediment_timestep.call_count == 10
        assert sj.deltamodel.apply_subsidence.call_count == 10
        assert sj.deltamodel.finalize_timestep.call_count == 10
        assert sj.deltamodel.log_model_time.call_count == 10
        assert sj.deltamodel.output_checkpoint.call_count == 10
        assert sj.deltamodel.finalize.call_count == 1

        # check the log outputs for successes
        _calls = [mock.call('job: 0, stage: 1, code: 0'),
                  mock.call('job: 0, stage: 2, code: 0')]
        sj.deltamodel.logger.info.assert_has_calls(_calls)

    def test_run_error_update(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml')

        sj = preprocessor._SerialJob(
            i=99, input_file=p,
            config_dict={'timesteps': 10})

        # modify the save interval to be twice dt
        sj.deltamodel._save_dt = 2 * sj.deltamodel._dt
        sj.deltamodel._checkpoint_dt = 2 * sj.deltamodel._dt

        # mock top-level methods, verify call was made to each
        sj.deltamodel.solve_water_and_sediment_timestep = mock.MagicMock()
        sj.deltamodel.apply_subsidence = mock.MagicMock()
        sj.deltamodel.finalize_timestep = mock.MagicMock()
        sj.deltamodel.log_model_time = mock.MagicMock()
        sj.deltamodel.logger = mock.MagicMock()
        sj.deltamodel.output_data = mock.MagicMock()
        sj.deltamodel.output_checkpoint = mock.MagicMock(
            side_effect=RuntimeError('error!'))
        sj.deltamodel.finalize = mock.MagicMock()

        # run the method (error on `output_checkpoint`)
        sj.run()

        # assertions
        assert sj.deltamodel._time_iter == 1
        assert sj.deltamodel.solve_water_and_sediment_timestep.call_count == 1
        assert sj.deltamodel.apply_subsidence.call_count == 1
        assert sj.deltamodel.finalize_timestep.call_count == 1
        assert sj.deltamodel.log_model_time.call_count == 1
        assert sj.deltamodel.output_data.call_count == 1
        assert sj.deltamodel.output_checkpoint.call_count == 1
        assert sj.deltamodel.finalize.call_count == 1

        # check the log outputs for success/failure
        _info_calls = [mock.call('job: 99, stage: 2, code: 0')]
        _error_calls = [mock.call('job: 99, stage: 1, code: 1, msg: error!')]
        sj.deltamodel.logger.info.assert_has_calls(_info_calls)
        sj.deltamodel.logger.error.assert_has_calls(_error_calls)
        sj.deltamodel.logger.exception.assert_called_once()

    def test_run_error_finalize(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml')

        sj = preprocessor._SerialJob(
            i=0, input_file=p,
            config_dict={'timesteps': 10})

        # modify the save interval to be twice dt
        sj.deltamodel._save_dt = 2 * sj.deltamodel._dt
        sj.deltamodel._checkpoint_dt = 2 * sj.deltamodel._dt

        # mock top-level methods, verify call was made to each
        sj.deltamodel.solve_water_and_sediment_timestep = mock.MagicMock()
        sj.deltamodel.apply_subsidence = mock.MagicMock()
        sj.deltamodel.finalize_timestep = mock.MagicMock()
        sj.deltamodel.log_model_time = mock.MagicMock()
        sj.deltamodel.logger = mock.MagicMock()
        sj.deltamodel.output_data = mock.MagicMock()
        sj.deltamodel.output_checkpoint = mock.MagicMock()
        sj.deltamodel.finalize = mock.MagicMock(
            side_effect=RuntimeError('error!'))

        # run the method (error on `output_checkpoint`)
        sj.run()

        # assertions
        assert sj.deltamodel._time_iter == 10
        assert sj.deltamodel.solve_water_and_sediment_timestep.call_count == 10
        assert sj.deltamodel.apply_subsidence.call_count == 10
        assert sj.deltamodel.finalize_timestep.call_count == 10
        assert sj.deltamodel.log_model_time.call_count == 10
        assert sj.deltamodel.output_data.call_count == 10
        assert sj.deltamodel.output_checkpoint.call_count == 10
        assert sj.deltamodel.finalize.call_count == 1

        # check the log outputs for success/failure
        _info_calls = [mock.call('job: 0, stage: 1, code: 0')]
        _error_calls = [mock.call('job: 0, stage: 2, code: 1, msg: error!')]
        sj.deltamodel.logger.info.assert_has_calls(_info_calls)
        sj.deltamodel.logger.error.assert_has_calls(_error_calls)
        sj.deltamodel.logger.exception.assert_called_once()


class TestParallelJob:

    def test_run(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml')

        Q = mock.MagicMock()
        S = mock.MagicMock()

        pj = preprocessor._ParallelJob(
            i=0, queue=Q, sema=S, input_file=p,
            config_dict={'timesteps': 10})

        # modify the save interval to be twice dt
        pj.deltamodel._save_dt = 2 * pj.deltamodel._dt
        pj.deltamodel._checkpoint_dt = 2 * pj.deltamodel._dt

        # mock top-level methods, verify call was made to each
        pj.deltamodel.solve_water_and_sediment_timestep = mock.MagicMock()
        pj.deltamodel.apply_subsidence = mock.MagicMock()
        pj.deltamodel.finalize_timestep = mock.MagicMock()
        pj.deltamodel.log_model_time = mock.MagicMock()
        pj.deltamodel.logger = mock.MagicMock()
        pj.deltamodel.output_data = mock.MagicMock()
        pj.deltamodel.output_checkpoint = mock.MagicMock()
        pj.deltamodel.finalize = mock.MagicMock()

        # mock initialization and their hooks
        pj.deltamodel.init_output_file = mock.MagicMock()
        pj.deltamodel.hook_init_output_file = mock.MagicMock()
        pj.deltamodel.hook_output_data = mock.MagicMock()
        pj.deltamodel.hook_output_checkpoint = mock.MagicMock()

        # run the method (call update 10 times)
        pj.run()

        # assertions
        assert pj.deltamodel._time_iter == 10
        assert pj.deltamodel.solve_water_and_sediment_timestep.call_count == 10
        assert pj.deltamodel.apply_subsidence.call_count == 10
        assert pj.deltamodel.finalize_timestep.call_count == 10
        assert pj.deltamodel.log_model_time.call_count == 10
        assert pj.deltamodel.output_data.call_count == 11  # once in init
        assert pj.deltamodel.output_checkpoint.call_count == 11  # once in init
        assert pj.deltamodel.finalize.call_count == 1
        # hooks/initialization
        assert pj.deltamodel.init_output_file.call_count == 1
        assert pj.deltamodel.hook_init_output_file.call_count == 1
        assert pj.deltamodel.hook_output_data.call_count == 11
        assert pj.deltamodel.hook_output_checkpoint.call_count == 11

        # check the log outputs for successes
        _calls = [mock.call({'job': 0, 'stage': 0, 'code': 0}),
                  mock.call({'job': 0, 'stage': 1, 'code': 0}),
                  mock.call({'job': 0, 'stage': 2, 'code': 0})]
        Q.put.assert_has_calls(_calls)

    def test_run_error_update(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml')

        Q = mock.MagicMock()
        S = mock.MagicMock()

        pj = preprocessor._ParallelJob(
            i=99, queue=Q, sema=S, input_file=p,
            config_dict={'timesteps': 10})

        # modify the save interval to be twice dt
        pj.deltamodel._save_dt = 2 * pj.deltamodel._dt
        pj.deltamodel._checkpoint_dt = 2 * pj.deltamodel._dt

        # mock top-level methods, verify call was made to each
        pj.deltamodel.solve_water_and_sediment_timestep = mock.MagicMock()
        pj.deltamodel.apply_subsidence = mock.MagicMock()
        pj.deltamodel.finalize_timestep = mock.MagicMock()
        pj.deltamodel.log_model_time = mock.MagicMock(
            side_effect=RuntimeError('error!'))
        pj.deltamodel.logger = mock.MagicMock()
        pj.deltamodel.output_data = mock.MagicMock()
        pj.deltamodel.output_checkpoint = mock.MagicMock()
        pj.deltamodel.finalize = mock.MagicMock()

        # run the method (error on `output_checkpoint`)
        pj.run()

        # assertions
        assert pj.deltamodel._time_iter == 1
        assert pj.deltamodel.solve_water_and_sediment_timestep.call_count == 1
        assert pj.deltamodel.apply_subsidence.call_count == 1
        assert pj.deltamodel.finalize_timestep.call_count == 1
        assert pj.deltamodel.log_model_time.call_count == 1
        assert pj.deltamodel.output_data.call_count == 1
        assert pj.deltamodel.output_checkpoint.call_count == 1
        assert pj.deltamodel.finalize.call_count == 1

        # check the log outputs for success/failure
        _info_calls = [mock.call({'job': 99, 'stage': 0, 'code': 0}),
                       mock.call({'job': 99, 'stage': 2, 'code': 0})]
        _error_calls = [mock.call({'job': 99, 'stage': 1,
                                   'code': 1, 'msg': 'error!'})]
        Q.put.assert_has_calls(_info_calls, any_order=True)
        Q.put.assert_has_calls(_error_calls)
        pj.deltamodel.logger.error.assert_has_calls(
            [mock.call('error!')])
        pj.deltamodel.logger.exception.assert_called_once()

    def test_run_error_finalize(self, tmp_path):
        p = utilities.yaml_from_dict(tmp_path, 'input.yaml')

        Q = mock.MagicMock()
        S = mock.MagicMock()

        pj = preprocessor._ParallelJob(
            i=0, queue=Q, sema=S, input_file=p,
            config_dict={'timesteps': 10})

        # modify the save interval to be twice dt
        pj.deltamodel._save_dt = 2 * pj.deltamodel._dt
        pj.deltamodel._checkpoint_dt = 2 * pj.deltamodel._dt

        # mock top-level methods, verify call was made to each
        pj.deltamodel.solve_water_and_sediment_timestep = mock.MagicMock()
        pj.deltamodel.apply_subsidence = mock.MagicMock()
        pj.deltamodel.finalize_timestep = mock.MagicMock()
        pj.deltamodel.log_model_time = mock.MagicMock()
        pj.deltamodel.logger = mock.MagicMock()
        pj.deltamodel.output_data = mock.MagicMock()
        pj.deltamodel.output_checkpoint = mock.MagicMock()
        pj.deltamodel.finalize = mock.MagicMock(
            side_effect=RuntimeError('error!'))

        # run the method (error on `output_checkpoint`)
        pj.run()

        # assertions
        assert pj.deltamodel._time_iter == 10
        assert pj.deltamodel.solve_water_and_sediment_timestep.call_count == 10
        assert pj.deltamodel.apply_subsidence.call_count == 10
        assert pj.deltamodel.finalize_timestep.call_count == 10
        assert pj.deltamodel.log_model_time.call_count == 10
        assert pj.deltamodel.output_data.call_count == 11  # once in init
        assert pj.deltamodel.output_checkpoint.call_count == 11  # once in init
        assert pj.deltamodel.finalize.call_count == 1

        # check the log outputs for success/failure
        _info_calls = [mock.call({'job': 0, 'stage': 0, 'code': 0}),
                       mock.call({'job': 0, 'stage': 1, 'code': 0})]
        _error_calls = [mock.call({'job': 0, 'stage': 2,
                                   'code': 1, 'msg': 'error!'})]
        Q.put.assert_has_calls(_info_calls, any_order=True)
        Q.put.assert_has_calls(_error_calls)
        pj.deltamodel.logger.error.assert_has_calls(
            [mock.call('error!')])
        pj.deltamodel.logger.exception.assert_called_once()


class TestWriteYamlConfigToFile:

    def test_write_single_int(self, tmp_path):
        # set up what to write
        file_path = tmp_path / 'output.yml'
        yaml_dict = {'variable': 1}

        # write the file
        preprocessor._write_yaml_config_to_file(yaml_dict, file_path)

        with open(file_path) as f:
            _returned = ' '.join(f.readlines())

        assert _returned == 'variable: 1\n'

    def test_write_single_float(self, tmp_path):
        # set up what to write
        file_path = tmp_path / 'output.yml'
        yaml_dict = {'variable': 1.5}

        # write the file
        preprocessor._write_yaml_config_to_file(yaml_dict, file_path)

        with open(file_path) as f:
            _returned = ' '.join(f.readlines())

        assert _returned == 'variable: 1.5\n'

    def test_write_single_string(self, tmp_path):
        # set up what to write
        file_path = tmp_path / 'output.yml'
        yaml_dict = {'variable': 'a string'}

        # write the file
        preprocessor._write_yaml_config_to_file(yaml_dict, file_path)

        with open(file_path) as f:
            _returned = ' '.join(f.readlines())

        assert _returned == 'variable: a string\n'

    def test_write_single_bool(self, tmp_path):
        # set up what to write
        file_path = tmp_path / 'output.yml'
        yaml_dict = {'variable': False}

        # write the file
        preprocessor._write_yaml_config_to_file(yaml_dict, file_path)

        with open(file_path) as f:
            _returned = ' '.join(f.readlines())

        assert _returned == 'variable: False\n'

    def test_write_single_null(self, tmp_path):
        # set up what to write
        file_path = tmp_path / 'output.yml'
        yaml_dict = {'variable': None}

        # write the file
        preprocessor._write_yaml_config_to_file(yaml_dict, file_path)

        with open(file_path) as f:
            _returned = ' '.join(f.readlines())

        assert _returned == 'variable: null\n'

    def test_write_multiple_one_each(self, tmp_path):
        # set up what to write
        file_path = tmp_path / 'output.yml'
        yaml_dict = {'variable1': 1,
                     'variable2': 2.2,
                     'variable3': 'a string',
                     'variable4': False,
                     'variable5': None}

        # write the file
        preprocessor._write_yaml_config_to_file(yaml_dict, file_path)

        with open(file_path) as f:
            _returned = ' '.join(f.readlines())
        print(_returned)

        assert _returned == ('variable1: 1\n variable2: 2.2\n '
                             'variable3: a string\n variable4: False\n '
                             'variable5: null\n')


class TestPreprocessorImported:

    def test_Preprocessor_toplevelimport(self):
        import pyDeltaRCM

        assert 'Preprocessor' in dir(pyDeltaRCM)
        assert pyDeltaRCM.Preprocessor is pyDeltaRCM.preprocessor.Preprocessor


class TestScaleRelativeSeaLeveLRiseRate():

    def test_scale_If_1(self):
        scaled = preprocessor.scale_relative_sea_level_rise_rate(5, If=1)
        assert scaled == 5 / 1000 / 365.25 / 86400

    def test_scale_If_0p1(self):
        scaled = preprocessor.scale_relative_sea_level_rise_rate(5, If=0.1)
        assert scaled == 5 / 1000 / 365.25 / 86400 * 10
