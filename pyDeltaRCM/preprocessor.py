import os
import argparse
import abc

import yaml

from .shared_tools import _get_version
from .model import DeltaModel


_ver = ' '.join(('pyDeltaRCM', _get_version()))


class BasePreprocessor(abc.ABC):
    """Base preprocessor class.

    Defines a prelimiary yaml file parsing, the model instatiation,
    and timestep loop routine.

    Subclasses handle the high-level command line API 
    and the python API. 
    """

    def extract_yaml_config(self):
        """Extract yaml into dictionary if provided.

        Here, we check whether the file is valid yaml,
        and place it into a dictionary.
        """

        # open the file, an error will be thrown if invalid yaml?
        user_file = open(self.input_file, mode='r')
        self.user_dict = yaml.load(user_file, Loader=yaml.FullLoader)
        user_file.close()

        if 'matrix' in self.user_dict.keys():
            raise NotImplementedError(
                'Matrix expansion not yet implemented...')
            # 1. compute expansion, create indiv yaml files
            # 2. loop expanded to create jobs from yaml files
            self._has_matrix = True
        else:
            self._has_matrix = False

    def expand_yaml_matrix(self):
        """Expand YAML matrix, if given.

        Look for a field named ``matrix`` in the yaml dict.

        Compute the matrix expansion.

        """

        pass

    def extract_timesteps(self):
        if hasattr(self, 'cli_timesteps'):
            # overrides everything else
            self.timesteps = self.cli_timesteps

        if not hasattr(self, 'timesteps'):
            if 'timesteps' in self.user_dict.keys():
                self.timesteps = self.user_dict['timesteps']
            else:
                raise ValueError('You must specify timesteps in either the '
                                 'YAML configuration file or via the --timesteps '
                                 'CLI flag, in order to use the high-level API.')

    class _Job(object):

        def __init__(self, input_file, yaml_timesteps, cli_timesteps):
            self.deltamodel = DeltaModel(input_file=input_file)

            if yaml_timesteps:
                _timesteps = yaml_timesteps
            elif cli_timesteps:
                _timesteps = cli_timesteps
            else:
                raise ValueError('You must specify timesteps in either the '
                                 'YAML configuration file or via the --timesteps '
                                 'CLI flag, in order to use the high-level API.')
            self.timesteps = _timesteps

        def run_model(self):
            """Loop the model.

            Iterate the timestep ``update`` routine for the specified number of
            iterations.
            """

            for _t in range(self.timesteps):
                self.deltamodel.update()

        def finalize_model(self):
            self.deltamodel.finalize()


class PreprocessorCLI(BasePreprocessor):

    def __init__(self):

        super().__init__()

        self.process_arguments()

        if self.args['config']:
            self.input_file = self.args['config']
            self.extract_yaml_config()
        else:
            self.input_file = None
            self.user_dict = {}
            self._has_matrix = False

        if self.args['timesteps']:
            self.cli_timesteps = int(self.args['timesteps'])
        else:
            self.cli_timesteps = None

        if 'timesteps' in self.user_dict.keys():
            self.yaml_timesteps = self.user_dict['timesteps']
        else:
            self.yaml_timesteps = None

        self.job_list = []
        if self._has_matrix:
            self.expand_yaml_matrix()
        else:
            self.job_list.append(self._Job(self.input_file,
                                           yaml_timesteps=self.yaml_timesteps,
                                           cli_timesteps=self.cli_timesteps))

        self.extract_timesteps()

    def process_arguments(self):
        parser = argparse.ArgumentParser(
            description='Arguments for running pyDeltaRCM from command line')

        parser.add_argument('--config',
                            help='Path to a config file that you would like to use.')
        parser.add_argument('--timesteps',
                            help='Number of timesteps to run model defined '
                                 'in config file. Optional, and if provided, '
                                 'will override any value in the config file.')
        parser.add_argument('--dryrun', action='store_true',
                            help='Boolean indicating whether to execute '
                                 ' timestepping or only set up the run.')
        parser.add_argument('--version', action='version',
                            version=_ver, help='Print pyDeltaRCM version.')

        args = parser.parse_args()

        self.args = vars(args)


class Preprocessor(BasePreprocessor):
    """Python high level api.

    Call the preprocessor with a yaml file to handle multi-job yaml configs,
    as well as timestepping from script.
    """

    def __init__(self):
        raise NotImplementedError
        super().__init__(self)
        pass


def preprocessor_wrapper():
    """Wrapper for CLI interface.
    """
    pp = PreprocessorCLI()

    # check no mulitjobs, no implemented
    if len(pp.job_list) > 1:
        raise NotImplementedError
        # 1. set up parallel pool if multiple jobs
        # 2. run jobs in list
    ######

    # run the job(s)
    for job in pp.job_list:
        job.run_model()
        job.finalize_model()


if __name__ == '__main__':

    preprocessor_wrapper()
