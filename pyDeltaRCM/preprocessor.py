import os
import argparse
import abc

import itertools
from pathlib import Path

import yaml
import numpy as np

from . import shared_tools
from .model import DeltaModel


_ver = ' '.join(('pyDeltaRCM', shared_tools._get_version()))


class BasePreprocessor(abc.ABC):
    """Base preprocessor class.

    Defines a prelimiary yaml reading, then handles the YAML "meta" tag
    parsing, model instatiation, and job running.

    Subclasses create the high-level command line API 
    and the high-level python API. 

    .. note::

        You probably do not need to interact with this class direclty. Check
        out the Python API.

    """

    def extract_yaml_config(self):
        """Preliminary YAML parsing. 

        Extract ``.yml`` file (``self.input_file``) into a dictionary, if
        provided. This dictionary provides a few keys used throughout the
        code.

        Here, we check whether the file is valid yaml,
        and place it into a dictionary.

        Additionally, set the ``self._has_matrix`` flag, which is used in the
        :meth:`expand_yaml_matrix`.

        """
        # open the file, an error will be thrown if invalid yaml?
        user_file = open(self.input_file, mode='r')
        self.user_dict = yaml.load(user_file, Loader=yaml.FullLoader)
        user_file.close()

        if 'matrix' in self.user_dict.keys():
            self._has_matrix = True
        else:
            self._has_matrix = False

        if 'verbose' in self.user_dict.keys():
            self.verbose = self.user_dict['verbose']
        else:
            self.verbose = 0

    def write_yaml_config(self, i, ith_config, ith_dir, ith_id):
        """Write full config to file in output folder.

        Write the entire yaml configuation for the configured job out to a
        file in the job output foler.
        """
        if self.verbose > 0:
            print('Writing YAML file for job ' + str(int(i)))

        d = Path(ith_dir)
        d.mkdir()
        ith_p = d / (str(ith_id) + '.yml')
        write_yaml_config_to_file(ith_config, ith_p)
        return ith_p

    def expand_yaml_matrix(self):
        """Expand YAML matrix, if given.

        Compute the matrix expansion.

        """
        if self._has_matrix:
            # extract and remove 'matrix' from config
            _matrix = self.user_dict.pop('matrix')

            # check validity of matrix specs
            if not isinstance(_matrix, dict):
                raise ValueError(
                    'Invalid matrix spceification, was not evaluated to "dict".')
            if 'out_dir' not in self.user_dict.keys():
                raise ValueError(
                    'You must specify "out_dir" in YAML to use matrix expansion.')
            if 'out_dir' in _matrix.keys():
                raise ValueError(
                    'You cannot specify "out_dir" as a matrix expansion key.')
            for k in _matrix.keys():  # check validity of keys, depth == 1
                if len(_matrix[k]) == 1:
                    raise ValueError(
                        'Length of matrix key "%s" was 1, '
                        'relocate to fixed configuration.' % str(k))
                for v in _matrix[k]:
                    print(v)
                    if isinstance(v, list):
                        raise ValueError(
                            'Depth of matrix expansion must not be > 1')
                if ':' in k:
                    raise ValueError(
                        'Colon operator found in matrix expansion key.')
                if k in self.user_dict.keys():
                    raise ValueError(
                        'You cannot specify the same key in the matrix '
                        'configuration and fixed configuration. '
                        'Key "%s" was specified in both.' % str(k))

            # compute the expansion
            var_list = [k for k in _matrix.keys()]
            lil = [_matrix[v] for k, v in enumerate(var_list)]
            dims = len(lil)
            pts = [len(l) for l in lil]
            jobs = np.prod(pts)
            _combs = list(itertools.product(*lil))  # actual matrix expansion
            _fixed_config = self.user_dict.copy()  # fixed config dict to expand on

            if self.verbose > 0:
                print(('Matrix expansion:\n' +
                       '  dims {_dims}\n' +
                       '  jobs {_jobs}').format(_dims=dims, _jobs=jobs))

            # create directory at root
            jobs_root = self.user_dict['out_dir']  # checked above for exist
            p = Path(jobs_root)
            try:
                p.mkdir()
            except FileExistsError:
                raise FileExistsError(
                    'Job output directory (%s) already exists.' % str(p))

            # loop, create job yamls
            self.file_list = []
            for i in range(jobs):
                _ith_config = _fixed_config.copy()
                ith_id = 'job_' + str(i).zfill(3)
                ith_dir = os.path.join(jobs_root, ith_id)
                _ith_config['out_dir'] = ith_dir
                for j, val in enumerate(_combs[i]):
                    _ith_config[var_list[j]] = val
                ith_p = self.write_yaml_config(i, _ith_config, ith_dir, ith_id)
                self.file_list.append(ith_p)

    def extract_timesteps(self):
        """Pull timestep from arg and YAML.

        Extract the `timesteps` parameter from the arguments line and YAML
        configuration. The arguments line may come from either the command
        line as ``--timesteps=N`` or in the python API as ``timesteps=N``,
        where ``N`` is the number of timesteps.

        """
        if hasattr(self, 'arg_timesteps'):
            # overrides everything else
            self.timesteps = self.arg_timesteps

        if not hasattr(self, 'timesteps'):
            if 'timesteps' in self.user_dict.keys():
                self.timesteps = self.user_dict['timesteps']
            else:
                raise ValueError('You must specify timesteps in either the '
                                 'YAML configuration file or via the --timesteps '
                                 'CLI flag, in order to use the high-level API.')

    def construct_job_list(self):
        """Construct the job list.

        The job list is constructed by expanding the ``.yml`` matrix, and
        forming ensemble runs as needed.

        """
        self.job_list = []
        if self._has_matrix:
            self.expand_yaml_matrix()
            # loop the expanded jobs
            for i in range(len(self.file_list)):
                try:
                    self.job_list.append(self._Job(self.file_list[i],
                                                   yaml_timesteps=self.yaml_timesteps,
                                                   arg_timesteps=self.arg_timesteps))
                except TypeError as e:
                    raise TypeError(
                        'During job instantiation, one of the model '
                        'configurations received an incorrect type. This '
                        'is likely because one of the "matrix" keys was '
                        'misconfigured in your input YAML file. '
                        'Check that the type of each value matches the '
                        'expected type for that key. The original error '
                        'is reproduced below:\n\n' +
                        str(e))
        else:
            # there's only one job so append directly.
            self.job_list.append(self._Job(self.input_file,
                                           yaml_timesteps=self.yaml_timesteps,
                                           arg_timesteps=self.arg_timesteps))

    def run_jobs(self):
        """Run the set of jobs.

        .. note::
            TODO: implement the parallel pool.

        """
        if len(self.job_list) > 1:
            # set up parallel pool if multiple jobs
            pass

        # run the job(s)
        for i, job in enumerate(self.job_list):

            if self.verbose > 0:
                print("Starting job %s" % str(i))

            job.run_job()
            job.finalize_model()

    class _Job(object):

        def __init__(self, input_file, yaml_timesteps, arg_timesteps):
            """Initialize a job.

            All arguments are required, but either the ``yaml_timesteps`` or
            ``arg_timesteps`` can be `None`.

            The ``input_file`` argument is passed to the DeltaModel for
            instantiation.

            The ``timesteps`` value is selected from either of the timestep
            input arguments, *but the YAML input will override an argument
            values.
            """
            self.deltamodel = DeltaModel(input_file=input_file)

            # determine the timestep from the two arguments given. Either the
            # yaml file or the arguments should include the timesteps
            # argument.
            if yaml_timesteps:
                _timesteps = yaml_timesteps
            elif arg_timesteps:
                _timesteps = arg_timesteps
            else:
                raise ValueError('You must specify timesteps in either the '
                                 'YAML configuration file or via the timesteps '
                                 'argument, in order to use the high-level API.')
            self.timesteps = _timesteps

            self._is_completed = False

        def run_job(self):
            """Loop the model.

            Iterate the timestep ``update`` routine for the specified number of
            iterations.
            """
            for _ in range(self.timesteps):
                self.deltamodel.update()

        def finalize_model(self):
            """Finalize the job."""
            self.deltamodel.finalize()
            self._is_completed = True


class PreprocessorCLI(BasePreprocessor):
    """Command line preprocessor.

    This is the main CLI class that is called from the command line. The class
    defines a method to process the arguments from the command line (using the
    `argparse` package).

    .. note:: 

        You probably do not need to interact with this class directly.
        Instead, you can use the command line API as it is defined HERE XX or
        the python API :class:`~pyDeltaRCM.preprocessor.Preprocessor`. 

        When the class is called from the command line the instantiated object's
        method :meth:`run_jobs` is called by the
        :obj:`~pyDeltaRCM.preprocessor.preprocessor_wrapper` function that the
        CLI (entry point) calls directly.

    """

    def __init__(self):
        """Initialize the CLI preprocessor.

        The initialization includes the entire configuration of the job list
        (parsing, timesteps, etc.). The jobs are *not* run automatically
        during instantiation of the class.
        """
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
            self.arg_timesteps = int(self.args['timesteps'])
        else:
            self.arg_timesteps = None

        if 'timesteps' in self.user_dict.keys():
            self.yaml_timesteps = self.user_dict['timesteps']
        else:
            self.yaml_timesteps = None

        self.extract_timesteps()

        self.construct_job_list()

    def process_arguments(self):
        """Process the command line arguments.

        .. note:: 

            The command line args are not directly passed to this function in
            any way.

        """
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

    This is the python high-level API class that is callable from a python
    script. For complete documentation on the API configurations, see
    XXXXXXXXXXXXXX. 

    The class gives a way to configure and run multiple jobs from a python script.

    Examples
    --------

    To configure a set of jobs (or a single job), instantiate the python
    preprocessor and then manually run the configured jobs:

    .. code::

        >>> pp = preprocessor.Preprocessor(input_file=p, timesteps=2)
        >>> pp.run_jobs()

    """

    def __init__(self, input_file=None, timesteps=None):
        """Initialize the python preprocessor.

        The initialization includes the entire configuration of the job list
        (parsing, timesteps, etc.). The jobs are *not* run automatically
        during instantiation of the class.

        You must specify timesteps in either the YAML configuration file or
        via the `timesteps` parameter.

        Parameters
        ----------
        input_file : :obj:`str`, optional
            Path to an input YAML configuration file. Must include the
            `timesteps` parameter if you do not specify the `timesteps` as a
            keyword argument.

        timesteps : :obj:`int`, optional
            Number of timesteps to run each of the jobs. Must be specified if
            you do not specify the `timesteps` parameter in the input YAML
            file.

        """
        super().__init__()

        if not input_file and not timesteps:
            raise ValueError('Cannot use Preprocessor with no arguments.')

        if input_file:
            self.input_file = input_file
            self.extract_yaml_config()
        else:
            self.input_file = None
            self.user_dict = {}
            self._has_matrix = False

        if timesteps:
            self.arg_timesteps = int(timesteps)
        else:
            self.arg_timesteps = None

        if 'timesteps' in self.user_dict.keys():
            self.yaml_timesteps = self.user_dict['timesteps']
        else:
            self.yaml_timesteps = None

        self.extract_timesteps()

        self.construct_job_list()


def preprocessor_wrapper():
    """Wrapper for CLI interface.

    The entry_points setup of a command line interface requires a function, so
    we use this simple wrapper to instantiate and run the jobs.

    Works by creating an instance of the
    :obj:`~pyDeltaRCM.preprocessor.PreprocessorCLI` and calls the
    :meth:`~pyDeltaRCM.preprocessor.PreprocessorCLI.run_jobs` to execute all
    jobs.
    """
    pp = PreprocessorCLI()
    pp.run_jobs()


def write_yaml_config_to_file(_config, _path):
    """Write a config to file in output folder.

    Write the entire yaml configuation for the configured job out to a
    file in the job output foler.
    """
    def _write_parameter_to_file(f, varname, varvalue):
        """Write each line, formatted."""
        f.write(varname + ': ' + str(varvalue) + '\n')

    f = open(_path, "a")
    for k in _config.keys():
        _write_parameter_to_file(f, k, _config[k])
    f.close()


if __name__ == '__main__':

    preprocessor_wrapper()
