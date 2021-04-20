import os
import sys
import glob

import unittest.mock as mock

from pyDeltaRCM.model import DeltaModel

from . import utilities


class TestLogModelTime:

    def test_records_arbitrary_time_values(self, tmp_path, capsys):
        """
        This test should create the log, and then print nothing at all.
        """
        file_name = 'user_parameters.yaml'
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'out_dir')
        f.close()
        _delta = DeltaModel(input_file=p)

        _delta.logger = mock.MagicMock()

        # change values
        _delta._time = 3.14159
        _delta._time_iter = 42

        _delta.log_model_time()

        # will record whatever value is in _delta
        _calls = [mock.call('Time: 3.1; timestep: 42')]
        _delta.logger.info.assert_has_calls(_calls)

    def test_verbose_printing_0(self, tmp_path, capsys):
        """
        This test should create the log, and then print nothing at all.
        """
        file_name = 'user_parameters.yaml'
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'out_dir')
        utilities.write_parameter_to_file(f, 'verbose', 0)
        f.close()
        _delta = DeltaModel(input_file=p)

        _delta.logger = mock.MagicMock()

        _delta.log_model_time()

        # time is logged regardless of verbosity
        _calls = [mock.call('Time: 0.0; timestep: 0')]
        _delta.logger.info.assert_has_calls(_calls)

        # stdout is empty because verbose 0
        captd = capsys.readouterr()
        assert captd.out == ''

    def test_verbose_printing_1(self, tmp_path, capsys):
        """
        This test should create the log, and then print nothing at all.
        """
        file_name = 'user_parameters.yaml'
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'out_dir')
        utilities.write_parameter_to_file(f, 'verbose', 1)
        f.close()
        _delta = DeltaModel(input_file=p)

        _delta.logger = mock.MagicMock()

        _delta.log_model_time()

        # time is logged regardless of verbosity
        _calls = [mock.call('Time: 0.0; timestep: 0')]
        _delta.logger.info.assert_has_calls(_calls)

        # stdout has times because verbose 1
        captd = capsys.readouterr()
        assert 'Time: 0.0' in captd.out  # if verbose >= 1

    def test_verbose_printing_2(self, tmp_path, capsys):
        """
        This test should create the log, and then print nothing at all.
        """
        file_name = 'user_parameters.yaml'
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'out_dir')
        utilities.write_parameter_to_file(f, 'verbose', 2)
        f.close()
        _delta = DeltaModel(input_file=p)

        _delta.logger = mock.MagicMock()

        _delta.log_model_time()

        # time is logged regardless of verbosity
        _calls = [mock.call('Time: 0.0; timestep: 0')]
        _delta.logger.info.assert_has_calls(_calls)

        # stdout is empty because verbose 2
        captd = capsys.readouterr()
        assert 'Time: 0.0' in captd.out  # if verbose >= 1


class TestLoggerIntegratedDuringInitialization:
    """Test that logger records during instantiation.

    These tests could be set up with a patch on the logger before
    instantiation of the DeltaModel...
    """

    def test_logger_has_initialization_lines(self, tmp_path):
        file_name = 'user_parameters.yaml'
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'out_dir')
        utilities.write_parameter_to_file(f, 'verbose', 1)
        utilities.write_parameter_to_file(f, 'seed', 10)
        f.close()
        _delta = DeltaModel(input_file=p)

        _logs = glob.glob(os.path.join(_delta.prefix, '*.log'))
        assert len(_logs) == 1  # log file exists
        with open(_logs[0], 'r') as _logfile:
            _lines = _logfile.readlines()
            _lines = ' '.join(_lines)  # collapse to a single string
            assert 'Setting model constants' in _lines
            assert 'Random seed is: 10' in _lines
            assert 'Creating model domain' in _lines
            assert 'Initializing output NetCDF4 file' in _lines
            assert 'Model initialization complete' in _lines

            if sys.platform.startswith('linux'):
                assert 'Platform: Linux-' in _lines
            elif sys.platform == 'darwin':
                guess1 = 'Platform: Darwin-' in _lines
                guess2 = 'Platform: macOS-' in _lines
                assert (guess1 | guess2)
            elif sys.platform.startswith('win'):
                assert 'Platform: Windows-' in _lines
            else:
                raise RuntimeError(
                    'Platform type not recognized.')
        assert not os.path.isfile(os.path.join(
            tmp_path, 'out_dir', 'discharge_0.0.png'))
        assert not os.path.isfile(os.path.join(
            tmp_path, 'out_dir', 'eta_0.0.png'))

    def test_logger_random_seed_always_recorded(self, tmp_path):
        file_name = 'user_parameters.yaml'
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'out_dir')
        utilities.write_parameter_to_file(f, 'verbose', 0)
        # do not set the seed explicitly, let it be set by the model
        f.close()
        delta = DeltaModel(input_file=p)
        _logs = glob.glob(os.path.join(delta.prefix, '*.log'))
        assert len(_logs) == 1  # log file exists
        with open(_logs[0], 'r') as _logfile:
            _lines = _logfile.readlines()
            _joinedlines = ' '.join(_lines)  # collapse to a single string
            assert 'Random seed is: ' in _joinedlines

            # determine the index of the line
            _idx = ['Random seed is: ' in _l for _l in _lines]
            assert sum(_idx) == 1  # one and only one True in list
            _idx = _idx.index(True)

            # try to covert to int, otherwise fail
            _seed = _lines[_idx].split(':')[-1]  # pull the seed value
            try:
                _intseed = int(_seed)
            except ValueError:
                raise ValueError('Could not convert the seed to int')

            assert _intseed >= 0
