import sys
import os
import glob

import pytest
from pyDeltaRCM.model import DeltaModel

# utilities for file writing


def create_temporary_file(tmp_path, file_name):
    d = tmp_path / 'configs'
    try:
        d.mkdir()
    except Exception:
        pass
    p = d / file_name
    f = open(p, "a")
    return p, f


def write_parameter_to_file(f, varname, varvalue):
    f.write(varname + ': ' + str(varvalue) + '\n')


def write_matrix_to_file(f, keys, lists):
    # assert len(keys) == len(lists)
    f.write('matrix' + ': ' + '\n')
    for i in range(len(keys)):
        f.write('  ' + keys[i] + ': ' + '\n')
        for j in range(len(lists[i])):
            f.write('    ' + '- ' + str(lists[i][j]) + '\n')


def yaml_from_dict(tmp_path, file_name, _dict):
    p, f = create_temporary_file(tmp_path, file_name)
    for k in _dict.keys():
        write_parameter_to_file(f, k, _dict[k])
    f.close()
    return p


@pytest.fixture(scope='function')
def test_DeltaModel(tmp_path):
    file_name = 'user_parameters.yaml'
    p, f = create_temporary_file(tmp_path, file_name)
    write_parameter_to_file(f, 'out_dir', tmp_path / 'out_dir')
    write_parameter_to_file(f, 'Length', 10.0)
    write_parameter_to_file(f, 'Width', 10.0)
    write_parameter_to_file(f, 'seed', 0)
    write_parameter_to_file(f, 'dx', 1.0)
    write_parameter_to_file(f, 'L0_meters', 1.0)
    write_parameter_to_file(f, 'S0', 0.0002)
    write_parameter_to_file(f, 'itermax', 1)
    write_parameter_to_file(f, 'Np_water', 10)
    write_parameter_to_file(f, 'u0', 1.0)
    write_parameter_to_file(f, 'N0_meters', 2.0)
    write_parameter_to_file(f, 'h0', 1.0)
    write_parameter_to_file(f, 'H_SL', 0.0)
    write_parameter_to_file(f, 'SLR', 0.001)
    write_parameter_to_file(f, 'Np_sed', 10)
    write_parameter_to_file(f, 'f_bedload', 0.5)
    write_parameter_to_file(f, 'C0_percent', 0.1)
    write_parameter_to_file(f, 'toggle_subsidence', False)
    write_parameter_to_file(f, 'sigma_max', 0.0)
    write_parameter_to_file(f, 'start_subsidence', 50.)
    write_parameter_to_file(f, 'save_eta_figs', False)
    write_parameter_to_file(f, 'save_stage_figs', False)
    write_parameter_to_file(f, 'save_depth_figs', False)
    write_parameter_to_file(f, 'save_discharge_figs', False)
    write_parameter_to_file(f, 'save_velocity_figs', False)
    write_parameter_to_file(f, 'save_eta_grids', False)
    write_parameter_to_file(f, 'save_stage_grids', False)
    write_parameter_to_file(f, 'save_depth_grids', False)
    write_parameter_to_file(f, 'save_discharge_grids', False)
    write_parameter_to_file(f, 'save_velocity_grids', False)
    write_parameter_to_file(f, 'save_dt', 500)
    write_parameter_to_file(f, 'save_strata', True)
    f.close()
    _delta = DeltaModel(input_file=p)
    return _delta


def read_endtime_from_log(log_folder):
    _logs = glob.glob(os.path.join(log_folder, '*.log'))
    assert len(_logs) == 1  # log file exists
    with open(_logs[0], 'r') as _logfile:
        _lines = _logfile.readlines()
        _t = 0
        for i, _line in enumerate(_lines):
            if 'Time: ' in _line:
                _t = _line.split(' ')[6]
                _t = _t.strip(' ;')
        return float(_t)
