import os
import argparse

import pyDeltaRCM as deltaModule
from pyDeltaRCM.deltaRCM_driver import pyDeltaRCM


def run_model(use_test_yaml=False):

    parser = argparse.ArgumentParser(description='Options for running pyDeltaRCM from command line')

    parser.add_argument('--config', help='Path to a config file that you would like to use.')
    parser.add_argument('--version', action='version', version='pyDeltaRCM ' + deltaModule.__version__, help='Prints the version of pyDeltaRCM.')

    args = parser.parse_args()

    cmdlineargs = vars(args)

    if cmdlineargs['config'] is not None:
        delta = pyDeltaRCM(input_file=cmdlineargs['config'])
    # elif use_test_yaml:
    #     test_yaml = os.path.join(os.path.dirname(__file__), '..', 'tests', 'test_output.yaml')
    #     delta = pyDeltaRCM(input_file=os.path.abspath(test_yaml))
    else:
        delta = pyDeltaRCM()

    for time in range(0, delta.timesteps):

        print(delta._time)
        delta.update()

        delta.finalize()


if __name__ == '__main__':

    run_model()
