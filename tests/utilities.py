import sys
import os
import glob

import numpy as np

import pytest
from pyDeltaRCM.model import DeltaModel
from pyDeltaRCM import shared_tools

# utilities for file writing


def create_temporary_file(tmp_path, file_name):
    d = tmp_path / 'configs'
    d.mkdir(parents=True, exist_ok=True)
    p = d / file_name
    f = open(p, "w")
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


def write_set_to_file(f, set_list):
    f.write('set' + ': ' + '\n')
    for i, _set in enumerate(set_list):
        f.write('  - {')
        for j, (k, v) in enumerate(_set.items()):
            f.write(k + ': ' + str(v) + ', ')
        f.write('}' + '\n')


def yaml_from_dict(tmp_path, file_name, _dict=None):
    p, f = create_temporary_file(tmp_path, file_name)
    if (_dict is None):
        _dict = {'out_dir': tmp_path / 'out_dir'}
    elif ('out_dir' not in _dict.keys()):
        _dict['out_dir'] = tmp_path / 'out_dir'

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
    write_parameter_to_file(f, 'subsidence_rate', 0.0)
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
    f.close()
    _delta = DeltaModel(input_file=p)
    return _delta


class FastIteratingDeltaModel:
    """A Fast iterating DeltaModel

    This class is useful in patching the DeltaModel for timing tests. The
    patched DeltaModel uses the random number generation internally, so it
    will verify functionality in any checkpointing scenarios, and overwriting
    only the `solve_water_and_sediment_timestep` method removes most of the jitting compilation
    time and much of the actual computation time.
    """

    def solve_water_and_sediment_timestep(self):
        """PATCH"""

        def _get_random_field(shp):
            """Get a field or randoms using the shared function.

            It is critical to use the `shared_tools.get_random_uniform` for
            reproducibility.
            """
            field = np.zeros(shp, dtype=np.float32)
            for i in range(shp[0]):
                for j in range(shp[1]):
                    field[i, j] = shared_tools.get_random_uniform(1)
            return field

        shp = self.eta.shape
        self.eta += _get_random_field(shp)
        self.uw += _get_random_field(shp)
        self.ux += _get_random_field(shp)
        self.uy += _get_random_field(shp)
        self.depth += _get_random_field(shp)
        self.stage += _get_random_field(shp)


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
