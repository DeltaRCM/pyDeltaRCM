import os
import argparse
import abc
import time

import itertools
from pathlib import Path

import multiprocessing

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
    def __init__(self):
        """Initialize the base preprocessor.
        """
        self._is_completed = False

    def preliminary_yaml_parsing(self):
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
        self.yaml_dict = yaml.load(user_file, Loader=yaml.FullLoader)
        user_file.close()

        if 'ensemble' in self.yaml_dict.keys():
            self._has_ensemble = True
            self.expand_yaml_ensemble()

        if 'matrix' in self.yaml_dict.keys():
            self._has_matrix = True
        else:
            self._has_matrix = False

        if 'verbose' in self.yaml_dict.keys():
            self.verbose = self.yaml_dict['verbose']
        else:
            self.verbose = 0

        return self.yaml_dict

    def _create_matrix(self):
        """Create a matrix if not already in the yaml.

        Note that this is only needed for ensemble expansion.
        """
        _matrix = dict()
        self.yaml_dict['matrix'] = _matrix

    def _seed_matrix(self, n_ensembles):
        """Generate random integers to be used as seeds for ensemble runs.

        Parameters
        ----------
        n_ensembles : `int`
            Number of ensembles which is the number of seeds to generate.

        """
        _matrix = self.yaml_dict.pop('matrix')
        if not isinstance(_matrix, dict):
            raise ValueError(
                'Invalid matrix specification, was not evaluated to "dict".')

        if 'seed' in _matrix.keys():
            raise ValueError('Random seeds cannot be specified in the matrix, '
                             'if an "ensemble" number is specified as well.')

        # generate list of random seeds
        seed_list = []
        for i in range(n_ensembles):
            seed_list.append(np.random.randint((2**32) - 1, dtype='u8'))

        # add list of random seeds to the matrix
        _matrix['seed'] = seed_list
        self.yaml_dict['matrix'] = _matrix

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

    def expand_yaml_ensemble(self):
        """Create ensemble random seeds and put into matrix.

        Seed the yaml configuration based on the number of ensembles specified.
        If matrix exists add seeds there, if not, create matrix and add seeds.
        """
        if self._has_ensemble:
            # ensemble must be integer - check if valid
            _ensemble = self.yaml_dict.pop('ensemble')
            if not isinstance(_ensemble, int):
                raise TypeError('Invalid ensemble type, must be an integer.')

            # if matrix does not exist, then it must be created
            if 'matrix' not in self.yaml_dict.keys():
                self._create_matrix()

            # then the seed values must be added to the matrix
            self._seed_matrix(_ensemble)

    def expand_yaml_matrix(self):
        """Expand YAML matrix, if given.

        Compute the matrix expansion of parameters listed in `matrix` key.

        """
        if self._has_matrix:
            # extract and remove 'matrix' from config
            _matrix = self.yaml_dict.pop('matrix')

            # check validity of matrix specs
            if not isinstance(_matrix, dict):
                raise ValueError(
                    'Invalid matrix specification, was not evaluated to "dict".')
            if 'out_dir' not in self.yaml_dict.keys():
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
                    if isinstance(v, list):
                        raise ValueError(
                            'Depth of matrix expansion must not be > 1')
                if ':' in k:
                    raise ValueError(
                        'Colon operator found in matrix expansion key.')
                if k in self.yaml_dict.keys():
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
            _fixed_config = self.yaml_dict.copy()  # fixed config dict to expand on

            if self.verbose > 0:
                print(('Matrix expansion:\n' +
                       '  dims {_dims}\n' +
                       '  jobs {_jobs}').format(_dims=dims, _jobs=jobs))

            # create directory at root
            self.jobs_root = self.yaml_dict['out_dir']  # checked above for exist
            p = Path(self.jobs_root)
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
                ith_dir = os.path.join(self.jobs_root, ith_id)
                _ith_config['out_dir'] = ith_dir
                for j, val in enumerate(_combs[i]):
                    _ith_config[var_list[j]] = val
                ith_p = self.write_yaml_config(i, _ith_config, ith_dir, ith_id)
                self.file_list.append(ith_p)

    def construct_job_file_list(self):
        """Construct the job list.

        The job list is constructed by expanding the ``.yml`` matrix, and
        forming ensemble runs as needed.
        """
        if self._has_matrix:
            self.expand_yaml_matrix()  # creates self.file_list
        else:
            self.file_list = [self.input_file]

    def run_jobs(self):
        """Run the set of jobs.

        This will run jobs in parallel if `--parallel` is specified in the
        command line or YAML.

        """
        if self._dryrun:
            return

        # NOTE: we always use the multiprocessing infrastructure, regardless
        # of number of jobs, or whether running in serial or parallel.
        _parallel_flag = _optional_input(
            'parallel', cli_dict=self.cli_dict, yaml_dict=self.yaml_dict,
            default=False)
        if isinstance(_parallel_flag, bool):
            if (_parallel_flag is True):
                # (number cores avail - 1), or 1 and never 0
                num_parallel_processes = (multiprocessing.cpu_count() - 1) or 1
            else:
                # serial jobs
                num_parallel_processes = 1
        elif (isinstance(_parallel_flag, int)):
            num_parallel_processes = _parallel_flag
        else:
            num_parallel_processes = 1

        _msg = 'Running %g parallel jobs' % num_parallel_processes
        if self.verbose >= 1:
            print(_msg)

        # use Semaphore to limit number of concurrent Job
        s = multiprocessing.Semaphore(num_parallel_processes)

        # initialize empty list to maintain reference to all Job instances
        num_total_processes = len(self.file_list)
        self.job_list = list()

        # start a Queue for all Jobs
        q = multiprocessing.Queue()

        # loop and create and start all jobs
        for i in range(0, num_total_processes):
            s.acquire()  # aquire resource from Semaphore
            p = _Job(i=i, queue=q, sema=s, input_file=self.file_list[i],
                     cli_dict=self.cli_dict, yaml_dict=self.yaml_dict)
            self.job_list.append(p)
            p.start()

        # join processes to prevent ending the jobs before moving forward
        for i in self.job_list:
            i.join()

        # read from the queue and report (asynchronous...buggy...)
        time.sleep(1)
        while not q.empty():
            gotq = q.get()
            if gotq['code'] == 1:
                print("Job {job} ended in error:\n {msg}".format_map(gotq))
            else:
                print("Job {job} returned code {code} "
                      "for stage {stage}.".format_map(gotq))

        self._is_completed = True


class _Job(multiprocessing.Process):
    """Class for individual jobs to run.

    This class handles setting options for its own run time duration.

    You probably don't need to interact with this class directly.
    """

    def __init__(self, i, queue, sema, input_file, cli_dict, yaml_dict):
        """Initialize a job.

        The `input_file` argument is passed to the DeltaModel for
        instantiation.

        The various model run duration parameters are passed from the
        `cli_dict` and `yaml_dict` arguments, and are processed into a
        single value for the run time. Precedence is given to values
        specified in the command line interface.
        """
        super(_Job, self).__init__()
        self.i = i
        self.queue = queue
        self.sema = sema  # semaphore for limited multiprocessing
        self.input_file = input_file

        self.deltamodel = DeltaModel(input_file=input_file, defer_output=True)
        _curr_time = self.deltamodel._time

        self.timesteps = ('timesteps', cli_dict, yaml_dict)
        self.time = ('time', cli_dict, yaml_dict)
        self.time_years = ('time_years', cli_dict, yaml_dict)
        self.If = ('If', cli_dict, yaml_dict)

        # determine job end time, *in model time*
        if not (self.timesteps is None):
            self._job_end_time = _curr_time + \
                ((self.timesteps * self.deltamodel._dt))
        elif not (self.time is None):
            self._job_end_time = _curr_time + ((self.time) * self.If)
        elif not (self.time_years is None):
            self._job_end_time = _curr_time + \
                ((self.time_years) * self.If * 86400 * 365.25)
        else:
            raise ValueError(
                'You must specify a run duration configuration in either '
                'the input YAML file or via input arguments.')

    def run(self):
        """Loop the model.

        Iterate the timestep ``update`` routine for the specified number of
        iterations.
        """
        self.queue.put({'job': self.i, 'stage': 0, 'code': 0})
        try:
            try:
                self.deltamodel.init_output_file()
                while self.deltamodel._time < self._job_end_time:
                    self.deltamodel.update()
            except (RuntimeError, ValueError) as e:
                self.queue.put({'job': self.i, 'stage': 1, 'code': 1, 'msg': e})
            else:
                self.queue.put({'job': self.i, 'stage': 1, 'code': 0})

            try:
                self.deltamodel.finalize()
            except (RuntimeError, ValueError) as e:
                self.queue.put({'job': self.i, 'stage': 2, 'code': 1, 'msg': e})
            else:
                self.queue.put({'job': self.i, 'stage': 2, 'code': 0})
        finally:
            self.sema.release()

    @property
    def timesteps(self):
        return self._timesteps

    @timesteps.setter
    def timesteps(self, arg_tuple):
        _timesteps = _optional_input(
            arg_tuple[0], cli_dict=arg_tuple[1], yaml_dict=arg_tuple[2],
            type_func=int)
        self._timesteps = _timesteps

    @property
    def time(self):
        return self._time

    @time.setter
    def time(self, arg_tuple):
        _time = _optional_input(
            arg_tuple[0], cli_dict=arg_tuple[1], yaml_dict=arg_tuple[2],
            type_func=float)
        self._time = _time

    @property
    def time_years(self):
        return self._time_years

    @time_years.setter
    def time_years(self, arg_tuple):
        _time_years = _optional_input(
            arg_tuple[0], cli_dict=arg_tuple[1], yaml_dict=arg_tuple[2],
            type_func=float)
        self._time_years = _time_years

    @property
    def If(self):
        return self._If

    @If.setter
    def If(self, arg_tuple):
        _If = _optional_input(
            arg_tuple[0], cli_dict=arg_tuple[1], yaml_dict=arg_tuple[2],
            default=1, type_func=float)
        self._If = _If


def _optional_input(argument, cli_dict=None, yaml_dict=None,
                    default=None, type_func=lambda x: x):
    """
    Processor function to be used on the optional choices, of the _Job
    for determining the hierachy of options and setting a single
    value.

    Hierarchy is to use CLI/preprocessor options first, then yaml.

    Both inputs should be `dict`!

    If default is not `None`, this value is used if no parameter
    specified in the cli or yaml.

    If type_func is given, this function is applied to the parsed value,
    before returning.
    """
    if argument in cli_dict.keys() and not (cli_dict[argument] is None):
        return type_func(cli_dict[argument])
    elif argument in yaml_dict.keys():
        return type_func(yaml_dict[argument])
    elif not (default is None):
        return default
    else:
        return None


class PreprocessorCLI(BasePreprocessor):
    """Command line preprocessor.

    This is the main CLI class that is called from the command line. The class
    defines a method to process the arguments from the command line (using the
    `argparse` package).

    .. note::

        You probably do not need to interact with this class directly.
        Instead, you can use the command line API as it is defined HERE XX or
        the python API :class:`~pyDeltaRCM.preprocessor.Preprocessor`.

        When the class is called from the command line the instantiated
        object's method :meth:`run_jobs` is called by the
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

        if self.cli_dict['config']:
            self.input_file = self.cli_dict['config']
            self.preliminary_yaml_parsing()
        else:
            self.verbose = 0  # no verbosity by defauly
            self.input_file = None
            self.yaml_dict = {}
            self._has_matrix = False

        if self.cli_dict['dryrun']:
            self._dryrun = True
        else:
            self._dryrun = False

        self.construct_job_file_list()

    def process_arguments(self):
        """Process the command line arguments.

        .. note::

            The command line args are not directly passed to this function in
            any way.

        """
        parser = argparse.ArgumentParser(
            description='Arguments for running pyDeltaRCM from command line')

        parser.add_argument(
            '--config', help='Path to a config file that you would like '
            'to use.')
        parser.add_argument(
            '--timesteps', help='Number of timesteps to run model. Optional, '
            'and if provided, will override any value in the config file. '
            'Is overrided by "time" options.')
        parser.add_argument(
            '--time', help='Time to run the model until, in seconds.'
            'Time is scaled due to intermittency, which is set to 1 by '
            'default (i.e., no scaling). See options "If" or "time_years" '
            'for other specification options. Optional, '
            'and if provided, will override any value in the config file.')
        parser.add_argument(
            '--time_years', help='Time to run the model until, in years.'
            'Time is scaled due to intermittency, which is set to 1 by '
            'default (i.e., no scaling). See options "If" or "time" '
            'for other specification options. Optional, '
            'and if provided, will override any value in the config file.')
        parser.add_argument(
            '--If', help='Assumed intermittency factor for converting model '
            'time to real-world time. This is a value in (0, 1] representing '
            'the fraction of days per year that the delta is flooding. '
            'Optional, default value is 1.')
        parser.add_argument(
            '--dryrun', action='store_true',
            help='Boolean indicating whether to execute timestepping or only '
            'set up the run.')
        parser.add_argument(
            '--parallel', type=int, nargs='?', const=True,
            help='Run jobs in parallel, if possible. If given without any'
            'argument, the number of parallel cores used depends '
            'on system specs as (ncores - 1). Specify an integer value to '
            'specify the number of cores to be used. '
            'Optional, default value is False.')
        #    Note: the default value for the parallel option is assigned
        #        during arg parsing of the yaml file.
        parser.add_argument(
            '--version', action='version',
            version=_ver, help='Print pyDeltaRCM version.')

        args = parser.parse_args()

        self.cli_dict = vars(args)


class Preprocessor(BasePreprocessor):
    """Python high level api.

    This is the python high-level API class that is callable from a python
    script. For complete documentation on the API configurations, see
    XXXXXXXXXXXXXX.

    The class gives a way to configure and run multiple jobs from a python
    script.

    Examples
    --------

    To configure a set of jobs (or a single job), instantiate the python
    preprocessor and then manually run the configured jobs:

    .. code::

        >>> pp = preprocessor.Preprocessor(input_file=p, timesteps=2)
        >>> pp.run_jobs()

    """

    def __init__(self, input_file=None, **kwargs):
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
        self._dryrun = False

        if not input_file and len(kwargs.keys()) == 0:
            raise ValueError('Cannot use Preprocessor with no arguments.')

        self.cli_dict = kwargs

        if input_file:
            self.input_file = input_file
            self.preliminary_yaml_parsing()
        else:
            self.input_file = None
            self.yaml_dict = {}
            self._has_matrix = False

        self.construct_job_file_list()


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

    .. note::

        This fuinction is utilized by the BMI implementation of pyDeltaRCM as
        well.
    """
    def _write_parameter_to_file(f, varname, varvalue):
        """Write each line, formatted."""
        f.write(varname + ': ' + str(varvalue) + '\n')

    f = open(_path, "a")
    for k in _config.keys():
        _write_parameter_to_file(f, k, _config[k])
    f.close()


def scale_relative_sea_level_rise_rate(mmyr, If=1):
    """Scale a relative sea level rise rate to model time.

    This function scales any relative sea level rise rate (RSLR) (e.g., sea
    level rise, subsidence) to a rate appropriate for the model time. This is
    helpful, because most discussion of RSLR uses units of mm/yr, but the
    model (and model configuration) require units of m/s. Additionally, the
    model framework needs to assume an "intermittency factor" to convert from
    real-world time to model time.

    Relative sea level rise (subsidence and/or sea level rise) are scaled from
    real world dimensions of mm/yr to model input as:

    .. math::

        \widehat{RSLR} = (RSLR / 1000) \cdot \dfrac{1}{I_f \cdot 365.25 \cdot 86400}

    This conversion makes it such that when one real-world year has elapsed
    (:math:`I_f \cdot 365.25 \cdot 86400` seconds in model time), the relative
    sea level has changed by the number of millimeters specified in the input
    :obj:`mmyr`.

    .. note::

        Users should use this function to determine the value to specify in
        an input YAML configuration file; no scaling is performed
        internally.

    Parameters
    ----------
    mmyr : :obj:`float`
        Millimeters per year, relative sea level rise rate.

    If : :obj:`float`, optional
        Intermittency factor, fraction of time represented by morphodynamic
        activity. Should be in interval (0, 1). Defaults to 1 if not provided,
        i.e., no scaling is performed.

    Returns
    -------
    scaled : :obj:`float`
        Scaled relative sea level rise rate, in meters per second.
    """
    return (mmyr / 1000) * (1 / (shared_tools._scale_factor(
        If, units='years')))


if __name__ == '__main__':

    preprocessor_wrapper()
