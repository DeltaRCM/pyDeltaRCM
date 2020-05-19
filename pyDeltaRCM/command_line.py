import os
import argparse

import pyDeltaRCM
from . import deltaRCM_driver


def run_model():

    parser = argparse.ArgumentParser(description='Options for running pyDeltaRCM from command line')

    parser.add_argument('--config', help='Path to a config file that you would like to use.')
    parser.add_argument('--version', action='version', version='pyDeltaRCM ' + pyDeltaRCM.__version__, help='Prints the version of pyDeltaRCM.')

    args = parser.parse_args()

    cmdlineargs = vars(args)

    if cmdlineargs['config'] is not None:
        delta = deltaRCM_driver.pyDeltaRCM(input_file=cmdlineargs['config'])
    else:
        delta = deltaRCM_driver.pyDeltaRCM()

    for time in range(0, delta.timesteps):

        delta.update()

    delta.finalize()


if __name__ == '__main__':

    run_model()
