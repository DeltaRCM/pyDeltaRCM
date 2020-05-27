import os
import argparse
import abc

import yaml

from .shared_tools import _get_version
from .model import DeltaModel

_ver = _get_version()


class BasePreprocessor(abc.ABC):
    """Base preprocessor class.

    Defines a prelimiary yaml file parsing, the model instatiation,
    and timestep loop routine.

    Subclasses handle the high-level command line API 
    and the python API. 
    """

    def __init__(self):
        pass

    def preliminary_yaml_parsing(self):
        """A very preliminiary yaml parsing step.

        Here, we check whether 

            1) the file is valid yaml
            2) for a field named ``matrix`` in the yaml file
            3) for a field named ``timesteps`` in the yaml file, which is used
               for time looping if it is given. If ``timesteps`` is not given, 
               an error is raised, and the user is directed to either fix the 
               issue or use the low-level API.

        """
        
        # open the file, an error will be thrown if invalid yaml?
        user_file = open(self.input_file, mode='r')
        user_dict = yaml.load(user_file, Loader=yaml.FullLoader)
        user_file.close()

        if 'matrix' in user_dict.keys():
            raise NotImplementedError('Matrix expansion not yet implemented...')

        if 'timesteps' in user_dict.keys():
            self.timesteps = user_dict['timesteps']
        else:
            raise ValueError('You must specify timesteps to use the high-level API.')      

    def instatiate_model(self):
        self.deltamodel = DeltaModel(input_file=self.input_file)

    def run_model(self):
        """Loop the model.

        Iterate the timestep ``update`` routine for the specified number of
        iterations.
        """

        for _t in range(0, self.timesteps):
            self.deltamodel.update()

        self.deltamodel.finalize()


class CLI_API(BasePreprocessor):

    def __init__(self):

        super().__init__()

        self.process_arguments()

        if self.args['config']:
            self.input_file = self.args['config']
            self.preliminary_yaml_parsing()
        else:
            self.input_file = None

        self.instatiate_model()

        self.run_model()

    def process_arguments(self):
        parser = argparse.ArgumentParser(
            description='Options for running pyDeltaRCM from command line')

        parser.add_argument(
            '--config', help='Path to a config file that you would like to use.')
        parser.add_argument('--version', action='version',
                            version=_ver, help='Print pyDeltaRCM version.')

        args = parser.parse_args()

        self.args = vars(args)


class Python_API(BasePreprocessor):

    def __init__(self):
        raise NotImplementedError
        super().__init__(self)
        pass


if __name__ == '__main__':

    CLI_API()
