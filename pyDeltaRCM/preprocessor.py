import os
import sys
import argparse
import abc
import time
import platform

import itertools
from pathlib import Path
import warnings

import multiprocessing

import yaml
import numpy as np

from . import shared_tools
from .model import DeltaModel as BaseDeltaModel


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
        self._input_file = None  # initialize empty

        self._has_ensemble = False
        self._has_set = False
        self._has_matrix = False

        self._file_list = []    # list of yaml files for jobs
        self._config_list = []  # list of dict configs for jobs
        self._job_list = []     # list of _Job objects

        self._dryrun = False

        self._is_completed = False

    @property
    def file_list(self):
        """File list.

        A list of `Path` to input YAML files for jobs constructed by the
        preprocessor.
        """
        return self._file_list

    @property
    def config_list(self):
        """Configuration list.

        A list of `dict` containing the input configurations for jobs
        constructed by the preprocessor.
        """
        return self._config_list

    @property
    def job_list(self):
        """Job list.

        A list of jobs, type :obj:`_SerialJob` or :obj:`_ParallelJob`, from
        the jobs executed by the preprocessor.

        .. note::

            This will be an empty list before :obj:`run_jobs` has been called.
        """
        return self._job_list

    def open_input_file_to_dict(self, input_file):
        """Open an input file and convert it to a dictionary.

        This method is used by subclassing Preprocessors to complete the first
        import of the yaml configuration.

        Parameters
        ----------
        input_file
            Path to the input file as string, or Pathlib `Path`.

        Returns
        -------
        yaml_dict
            Input file converted to a dictionary.
        """
        # handle complex preprocessor cases where already is a dict
        if isinstance(input_file, dict):
            yaml_dict = input_file

        # simple cases will be string/path
        elif (isinstance(input_file, str) or
              isinstance(input_file, Path)):
            yaml_dict = self._open_yaml(input_file)

        # something wrong with input
        else:
            raise ValueError('Invalid input file argument.')

        return yaml_dict

    def construct_file_list(self):
        """Construct the file list.

        The job list is constructed by expanding the various multi-job
        specifications. For example, `matrix`, `set`, and
        ensemble yaml files are created in this stage.
        """
        # complete a preliminary parsing of the yaml for high-level controls
        self._prelim_config_parsing()

        # if there is an ensemble specification, expand it first
        if self._has_ensemble:
            self._expand_ensemble()

        if self._has_ensemble or self._has_set or self._has_matrix:
            self._prepare_multijob_output()

        # if there is a matrix or set specification
        if self._has_matrix:
            self._expand_matrix()  # creates self.file_list
        elif self._has_set:
            self._expand_set()
        # otherwise convert to a simple list on input file
        else:
            self._file_list = [self._input_file]
            self._config_list = [self.config_dict]

        # write the job configs to file, if needed
        if self._has_ensemble or self._has_set or self._has_matrix:
            self._write_job_configs()

    def _prelim_config_parsing(self):
        """Preliminary configuration parsing.

        Extract ``.yml`` file (``self.input_file``) into a dictionary, if
        provided. This dictionary provides a few keys used throughout the
        code.

        Here, we check whether the file is valid yaml,
        and place it into a dictionary.

        Additionally, set the ``self._has_matrix`` flag, which is used in the
        :meth:`_expand_matrix`.

        """
        if 'set' in self.config_dict.keys():
            # set found
            self._has_set = True
            # can't specify matrix with set
            if ('matrix' in self.config_dict.keys()):
                raise ValueError(
                    'Cannot use "matrix" and "set" expansion together.')

        if 'ensemble' in self.config_dict.keys():
            # ensemble found
            self._has_ensemble = True

        if 'matrix' in self.config_dict.keys():
            # matrix found
            self._has_matrix = True

        if 'verbose' in self.config_dict.keys():
            self.verbose = self.config_dict['verbose']
        else:
            self.verbose = 0

        if 'dryrun' in self.config_dict.keys():
            self._dryrun = self.config_dict['dryrun']

    def _open_yaml(self, input_file):
        """Safely open, read, and close a yaml file.

        Parameters
        ----------
        input file
            string or path to file

        Returns
        -------
        yaml_dict
            yaml file, as a Python dict
        """
        if (input_file is None):
            return {}  # return an empty dict
        else:
            # get the special loader from the shared tools
            loader = shared_tools.custom_yaml_loader()

            # open the file with the loader
            user_file = open(input_file, mode='r')
            yaml_dict = yaml.load(user_file, Loader=loader)
            user_file.close()
            return yaml_dict

    def _prepare_multijob_output(self):
        if 'out_dir' not in self.config_dict.keys():
            raise ValueError(
                'You must specify "out_dir" in YAML to use any '
                'multi-job expansion tools.')

        self._jobs_root = self.config_dict['out_dir']

        # create directory at root
        p = Path(self._jobs_root)
        if p.is_dir():
            if 'resume_checkpoint' in self.config_dict and \
              self.config_dict['resume_checkpoint'] is True:
                pass
            else:
                raise FileExistsError(
                    'Job output directory (%s) already exists.' % str(p))
        else:
            p.mkdir()

    def _expand_ensemble(self):
        """Create ensemble random seeds and put into matrix.

        Ensemble expansion is implemented as a special class of matrix/set
        expansion where:

            * if set specification is used, each set is duplicated the
              specified number of times, and a seed is added to each set
            * otherwise, the matrix key is set for the `seed` field of the
              model

        This approach ensures that the runs will have different
        outcomes, while allowing all other parameters to remain fixed, and
        supporting additional keys in the matrix/set.

        In the non-set expansion implementation, if the matrix does not yet
        exist (e.g., only ensemble specified), then the matrix is created.
        """
        # extract the ensemble key
        _ensemble = self.config_dict.pop('ensemble')

        # ensemble must be integer - check if valid
        if not isinstance(_ensemble, int):
            raise TypeError('Invalid ensemble type, must be an integer.')

        if _ensemble == 1:
            # there's nothing to do here
            warnings.warn(UserWarning(
                    'Ensemble was set to 1. '
                    'Remove this key from configuration.'))
            self._has_ensemble = False
            return

        if self._has_set:

            # pop the set to work with it, then replace at end
            old_set = self.config_dict.pop('set')
            n_ensembles = _ensemble  # from parsing above

            # here we augment each set by duplicating and add seed
            new_set = []  # new set list
            for ith_old_set in old_set:

                # loop through the number of ensembles
                for _ in range(n_ensembles):
                    # make a copy of the ith
                    jth_new_set = ith_old_set.copy()

                    # make a new seed for jth set
                    jth_seed = np.random.randint((2**32) - 1, dtype='u8')

                    # fill seed spec
                    jth_new_set['seed'] = jth_seed

                    # append to the new set list
                    new_set.append(jth_new_set)

            self.config_dict['set'] = new_set

        else:
            # if matrix does not exist, then it must be created
            if 'matrix' not in self.config_dict.keys():
                _matrix = {}
            else:
                _matrix = self.config_dict.pop('matrix')

            # check type of matrix
            #   is this needed here? Do earlier? Let it error naturally?
            if not isinstance(_matrix, dict):
                raise ValueError(
                    'Invalid matrix specification, '
                    'was not evaluated to "dict".')

            # check for invalid specs
            if 'seed' in _matrix.keys():
                raise ValueError(
                    'Random seeds cannot be specified in the matrix, '
                    'if an "ensemble" number is specified as well.')

            # generate list of random seeds
            seed_list = []
            n_ensembles = _ensemble
            for i in range(n_ensembles):
                seed_list.append(np.random.randint((2**32) - 1, dtype='u8'))

            # add list of random seeds to the matrix
            _matrix['seed'] = seed_list

            # write it back to the total configuration
            self.config_dict['matrix'] = _matrix

            # change the variable to expand matrix on future step
            self._has_matrix = True

    def _expand_set(self):

        # determine the set
        _set = self.config_dict.pop('set')

        # check that all sets are dictionaries
        if not isinstance(_set, list):
            raise TypeError(
                'Set list must be type `list` but was {}.'.format(type(_set)))
        for i, d in enumerate(_set):  # check validity of keys, depth == 1
            if not isinstance(d, dict):
                raise TypeError(
                    'Set must specify as a list of dictionaries')
            for k in d.keys():  # check validity of keys, depth == 1
                if ':' in k:
                    raise ValueError(
                        'Colon operator found in matrix expansion key.')

        # check that all sets have the same entries
        set0_set = set(_set[0].keys())
        for s in range(1, len(_set)):
            if not (set0_set == set(_set[s].keys())):
                raise ValueError('All keys in all sets must be identical.')

        # extract dimensionality of set
        jobs = len(_set)
        dims = len(_set[0])

        if self.verbose > 0:
            print(('Set expansion:\n' +
                   '  dims {_dims}\n' +
                   '  jobs {_jobs}').format(_dims=dims, _jobs=jobs))

        # preallocate the matrix expansion job yamls and output table
        self._matrix_table = np.empty(  # output table
            (jobs, dims+1), dtype='O')

        _fixed_config = self.config_dict.copy()  # fixed config dict

        # loop through each and create a config and add to _config_list
        for i in range(jobs):

            # being with the fixed config
            ith_config = _fixed_config.copy()

            # find job id and create output file
            ith_id = 'job_' + str(i).zfill(3)
            ith_dir = os.path.join(self._jobs_root, ith_id)

            # write the job number into output table
            self._matrix_table[i, 0] = ith_id

            # get config for this job
            ith_config['out_dir'] = ith_dir

            # loop through each var of this job
            for j, (key, val) in enumerate(_set[i].items()):

                # write info into fixed config dict
                ith_config[key] = val

                # write info into output table
                self._matrix_table[i, j+1] = val

            # add the configuration to a list to write out below
            self._config_list.append(ith_config)

        # store the matrix expansion
        #   this is useful for references by custom
        #   Python preprocessing / postprocessing
        matrix_table_file = os.path.join(
            self._jobs_root, 'jobs_parameters.txt')
        self._matrix_table_header = ', '.join(
            ['job_id', *[i for i in _set[0].keys()]])
        np.savetxt(matrix_table_file, self._matrix_table,
                   fmt='%s', delimiter=',', comments='',
                   header=self._matrix_table_header)

    def _expand_matrix(self):
        """Expand YAML matrix.

        Compute the matrix expansion of parameters listed in `matrix` key.
        """
        # extract and remove 'matrix' from config
        _matrix = self.config_dict.pop('matrix')

        # check validity of matrix specs
        if not isinstance(_matrix, dict):
            raise ValueError(
                'Invalid matrix specification, was not evaluated to "dict".')
        if 'out_dir' in _matrix.keys():
            raise ValueError(
                'You cannot specify "out_dir" as a matrix expansion key.')
        for k in _matrix.keys():  # check validity of keys, depth == 1
            if len(_matrix[k]) == 1:
                warnings.warn(UserWarning(
                    'Length of matrix key "%s" was 1, '
                    'remove this key from matrix configuration.' % str(k)))
            for v in _matrix[k]:
                if isinstance(v, list):
                    raise ValueError(
                        'Depth of matrix expansion must not be > 1')
            if ':' in k:
                raise ValueError(
                    'Colon operator found in matrix expansion key.')
            if k in self.config_dict.keys():
                raise ValueError(
                    'You cannot specify the same key in the matrix '
                    'configuration and fixed configuration. '
                    'Key "%s" was specified in both.' % str(k))

        # compute the expansion
        var_list = [k for k in _matrix.keys()]
        lil = [_matrix[v] for k, v in enumerate(var_list)]
        dims = len(lil)
        pts = [len(lst) for lst in lil]
        jobs = np.prod(pts)
        _combs = list(itertools.product(*lil))  # actual matrix expansion
        _fixed_config = self.config_dict.copy()  # fixed config dict

        if self.verbose > 0:
            print(('Matrix expansion:\n' +
                   '  dims {_dims}\n' +
                   '  jobs {_jobs}').format(_dims=dims, _jobs=jobs))

        # preallocate the matrix expansion job yamls and output table
        self._matrix_table = np.empty(  # output table
            (jobs, dims+1), dtype='O')

        # loop through each and create a config and add to _config_list
        for i in range(jobs):

            # begin with the fixed config
            ith_config = _fixed_config.copy()

            # find job id and create output file
            ith_id = 'job_' + str(i).zfill(3)
            ith_dir = os.path.join(self._jobs_root, ith_id)

            # write the job number into output table
            self._matrix_table[i, 0] = ith_id

            # get config for this job
            ith_config['out_dir'] = ith_dir

            # loop through each var of this job
            for j, val in enumerate(_combs[i]):

                # write info into fixed config dict
                ith_config[var_list[j]] = val

                # write info into output table
                self._matrix_table[i, j+1] = val

            # add the configuration to a list to write out below
            self._config_list.append(ith_config)

        # store the matrix expansion
        #   this is useful for references by custom
        #   Python preprocessing / postprocessing
        matrix_table_file = os.path.join(
            self._jobs_root, 'jobs_parameters.txt')
        self._matrix_table_header = ', '.join(['job_id', *var_list])
        np.savetxt(matrix_table_file, self._matrix_table,
                   fmt='%s', delimiter=',', comments='',
                   header=self._matrix_table_header)

    def _write_job_configs(self):

        if len(self._config_list) == 0:
            raise ValueError('Config list empty!')

        # loop through each job to write out info
        for c, config in enumerate(self.config_list):

            # write out the job specific yaml file
            # ith_p = self._write_yaml_config(c, config)
            if self.verbose > 0:
                print('Writing YAML file for job ' + str(int(c)))

            # ith_config = config['config']
            ith_dir = Path(config['out_dir'])       # job output folder
            ith_id = ith_dir.parts[-1]              # job id

            # create the output directory if needed
            if ith_dir.is_dir():
                if 'resume_checkpoint' in self.config_dict and \
                  self.config_dict['resume_checkpoint'] is True:
                    pass
                else:
                    raise FileExistsError(
                        'Job output directory (%s) already exists.' % str(ith_dir))
            else:
                ith_dir.mkdir()

            # write the file into the output directory
            ith_p = ith_dir / (str(ith_id) + '.yml')
            _write_yaml_config_to_file(config, ith_p)

            # append to the file list
            self._file_list.append(ith_p)

    def run_jobs(self, DeltaModel=None):
        """Run the set of jobs.

        This method can be seen as the actual execution stage of the
        preprocessor. If `--parallel` is specified in the command line or YAML
        file, the jobs will run in parallel.
        """
        if self._dryrun:
            return

        # process the special DeltaModel argument
        #  if not None, it is a class to be used by the Jobs
        if (DeltaModel is None):
            DeltaModel = BaseDeltaModel

        # initialize empty list to maintain reference to all Job instances
        num_total_processes = len(self.file_list)

        # NOTE: multiprocessing infrastructure is only available on linux.
        #       We only use parallel approach if the --parallel flag is given
        #       and we are running on linux.
        if 'parallel' in self.config_dict.keys():
            # and not (self.config_dict['parallel'] is None)):
            _parallel_flag = self.config_dict['parallel']
        else:
            _parallel_flag = False

        # if the parallel flag is given use the parallel infrastructure
        #   NOTE: the following evaluates to true for boolean `True` or
        #         any `int` > 0
        if _parallel_flag:
            # validate that os is Linux, otherwise error
            _os = platform.system()
            if _os != 'Linux':
                raise NotImplementedError(
                    'Parallel simulations only implemented on Linux.')

            # determine the number of processors to use
            if (_parallel_flag is True):
                # (number cores avail - 1), or 1 and never 0
                num_parallel_processes = (multiprocessing.cpu_count() - 1) or 1
            elif (isinstance(_parallel_flag, int)):
                num_parallel_processes = _parallel_flag
            else:
                raise ValueError('Parallel flag must be boolean or integer, '
                                 'but was {}, {}.'.format(type(_parallel_flag),
                                                          str(_parallel_flag)))
            # number of parallel processes is never greater than number of jobs
            num_parallel_processes = np.minimum(
                num_parallel_processes, num_total_processes)

            _msg = 'Running %g parallel jobs' % num_parallel_processes
            if self.verbose >= 1:
                print(_msg)

            # use Semaphore to limit number of concurrent Job
            s = multiprocessing.Semaphore(num_parallel_processes)

            # start a Queue for all Jobs
            q = multiprocessing.Queue()

            # loop and create and start all jobs
            for i in range(0, num_total_processes):
                s.acquire()  # aquire resource from Semaphore

                # open yaml file for specific job
                job_yaml = self._open_yaml(self.file_list[i])

                # apply job yaml to config dict
                #   enables complex job setups, where configs are edited
                #   between writing and .run_jobs()
                job_config = self.config_list[i].copy()
                job_config.update(job_yaml)

                # instantiate the job
                p = _ParallelJob(i=i, queue=q, sema=s,
                                 input_file=self.file_list[i],
                                 config_dict=job_config,
                                 DeltaModel=DeltaModel)
                self._job_list.append(p)
                p.start()

            # join processes to prevent ending the jobs before moving forward
            for i in self._job_list:
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

        # if the parallel flag is False (default)
        elif (_parallel_flag is False):

            # loop and create all jobs
            for i in range(0, num_total_processes):

                # open yaml file for specific job
                job_yaml = self._open_yaml(self.file_list[i])

                # apply job yaml to config dict
                #   enables complex job setups, where configs are edited
                #   between writing and .run_jobs()
                job_config = self.config_list[i].copy()
                job_config.update(job_yaml)

                p = _SerialJob(i=i,
                               input_file=self.file_list[i],
                               config_dict=job_config,
                               DeltaModel=DeltaModel)
                self._job_list.append(p)

            # run the job(s)
            for i, job in enumerate(self._job_list):
                if self.verbose > 0:
                    print("Starting job %s" % str(i))
                job.run()

        # if the parallel flag is a junk value
        else:
            raise ValueError

        self._is_completed = True


class _BaseJob(abc.ABC):
    """Base class for individual jobs to run via the preprocessor.

    The base class handles setting options for the run time duration, based on
    the inputs to the command line and the yaml file.

    .. important:: the :meth:`run` method must be implemented by subclasses.

    .. note:: You probably don't need to interact with this class directly.
    """

    def __init__(self, i, input_file, config_dict,
                 DeltaModel=None, defer_output=False):
        """Initialize a job.

        The `input_file` argument is passed to the DeltaModel for
        instantiation.

        The various model run duration parameters are passed from the
        `cli_dict` and `yaml_dict` arguments, and are processed into a
        single value for the run time. Precedence is given to values
        specified in the command line interface.
        """
        self.i = i
        self.input_file = input_file

        # if 'DeltaModel' in config_dict.keys():
        #     _DM = config_dict['DeltaModel']
        # else:
        #     _DM = DeltaModel
        if (DeltaModel is None):
            DeltaModel = BaseDeltaModel

        self.deltamodel = DeltaModel(input_file=input_file,
                                     defer_output=defer_output)
        _curr_time = self.deltamodel._time

        # process If
        if 'If' in config_dict.keys():
            self._If = float(config_dict['If'])
        else:
            self._If = 1.0

        # determine job end time, *in model time*
        if ('timesteps' in config_dict.keys()):
            # fill informational fields
            self._time_type = 'timesteps'
            self._time_config = config_dict['timesteps']

            # compute the end time
            self._job_end_time = _curr_time + \
                ((config_dict['timesteps'] * self.deltamodel._dt))

        elif ('time' in config_dict.keys()):
            # fill informational fields
            self._time_type = 'time'
            self._time_config = config_dict['time']

            # compute the end time
            self._job_end_time = _curr_time + \
                (config_dict['time'] * self._If)

        elif ('time_years' in config_dict.keys()):
            # fill informational fields
            self._time_type = 'time_years'
            self._time_config = config_dict['time_years']

            # compute the end time
            self._job_end_time = _curr_time + \
                (config_dict['time_years'] * self._If * 86400 * 365.25)
        else:
            raise ValueError(
                'You must specify a run duration configuration in either '
                'the input YAML file or via input arguments.')

    @abc.abstractmethod
    def run(self):
        """Implementation to run the jobs as needed.

        .. note::

            Consider using a series of ``try-except-else-finally` statements to
            ensure that this method handles all outcomes of the simulations.
        """
        ...


class _SerialJob(_BaseJob):
    """Serial job run by the preprocessor.

    This class is a subclass of :obj:`_BaseJob` and implements the :meth:`run`
    method for jobs that should occur in serial.

    .. note:: You probably don't need to interact with this class directly.
    """

    def __init__(self, i, input_file, config_dict, DeltaModel=None,):
        """Initialize a serial job.

        The `input_file` argument is passed to the DeltaModel for
        instantiation.

        The various model run duration parameters are passed from the
        `config_dict` argument, and are processed into a single value for the
        run time.
        """
        super().__init__(i, input_file, config_dict,
                         DeltaModel=DeltaModel, defer_output=False)

    def run(self):
        """Loop the model.

        Iterate the timestep ``update`` routine for the specified number of
        iterations.
        """
        # try to initialize and run the model
        try:
            # run the simualtion
            while self.deltamodel._time < self._job_end_time:
                self.deltamodel.update()

        # if the model run fails
        except (RuntimeError, ValueError) as e:
            _msg = ', '.join(['job: ' + str(self.i), 'stage: ' + '1',
                             'code: ' + '1', 'msg: ' + str(e)])
            self.deltamodel.logger.error(_msg)
            self.deltamodel.logger.exception(e)

        # if the model run succeeds
        else:
            _msg = ', '.join(['job: ' + str(self.i), 'stage: ' + '1',
                             'code: ' + '0'])
            self.deltamodel.logger.info(_msg)

        # try to finalize the model
        try:
            self.deltamodel.finalize()

        # if the model finalization fails
        except (RuntimeError, ValueError) as e:
            _msg = ', '.join(['job: ' + str(self.i), 'stage: ' + '2',
                             'code: ' + '1', 'msg: ' + str(e)])
            self.deltamodel.logger.error(_msg)
            self.deltamodel.logger.exception(e)

        # if the finalization succeeds
        else:
            _msg = ', '.join(['job: ' + str(self.i), 'stage: ' + '2',
                             'code: ' + '0'])
            self.deltamodel.logger.info(_msg)


class _ParallelJob(_BaseJob, multiprocessing.Process):
    """Parallel job run by the preprocessor.

    This class is a subclass of :obj:`_BaseJob` and implements the :meth:`run`
    method for jobs that should occur in parallel.

    .. note:: You probably don't need to interact with this class directly.
    """

    def __init__(self, i, queue, sema,
                 input_file, config_dict, DeltaModel=None):
        """Initialize a parallel job.

        The `input_file` argument is passed to the DeltaModel for
        instantiation.

        The various model run duration parameters are passed from the
        `config_dict` argument, and are processed into a single value for the
        run time.
        """
        # inherit with explicit method resolution order
        _BaseJob.__init__(self, i, input_file, config_dict,
                          DeltaModel=DeltaModel, defer_output=True)
        multiprocessing.Process.__init__(self)

        self.queue = queue
        self.sema = sema  # semaphore for limited multiprocessing

    def run(self):
        """Run the model, with infrastructure for parallel.

        Iterate the timestep ``update`` routine for the specified number of
        iterations.
        """
        self.queue.put({'job': self.i, 'stage': 0, 'code': 0})

        # overall wrapped in try to ensure sema is safely released
        try:
            # try to initialize and run the model
            try:
                # initialize the output files (defer_output=True above)
                # unless resuming from checkpoint, then load the checkpoint
                if self.deltamodel.resume_checkpoint:
                    # here we set defer output to false when loading the
                    #   checkpoint on this thread
                    self.deltamodel.load_checkpoint(defer_output=False)
                else:
                    # infrastructure deferred, need to trigger manually
                    self.deltamodel.hook_init_output_file()
                    self.deltamodel.init_output_file()

                    self.deltamodel.hook_output_data()
                    self.deltamodel.output_data()

                    self.deltamodel.hook_output_checkpoint()
                    self.deltamodel.output_checkpoint()

                # run the simualtion
                while self.deltamodel._time < self._job_end_time:
                    self.deltamodel.update()

            # if the model run fails
            except (RuntimeError, ValueError) as e:
                self.queue.put({'job': self.i, 'stage': 1,
                                'code': 1, 'msg': str(e)})
                self.deltamodel.logger.error(str(e))
                self.deltamodel.logger.exception(e)

            # if the model run succeeds
            else:
                self.queue.put({'job': self.i, 'stage': 1,
                                'code': 0})

            # try to finalize the model
            try:
                self.deltamodel.finalize()

            # if the model finalization fails
            except (RuntimeError, ValueError) as e:
                self.queue.put({'job': self.i, 'stage': 2,
                                'code': 1, 'msg': str(e)})
                self.deltamodel.logger.error(str(e))
                self.deltamodel.logger.exception(e)

            # if the finalization succeeds
            else:
                self.queue.put({'job': self.i, 'stage': 2,
                                'code': 0})

        # ALWAYS release the semaphore, so other runs can continue
        finally:
            self.sema.release()


class PreprocessorCLI(BasePreprocessor):
    """Command line preprocessor.

    This is the main CLI class that is called from the command line. The class
    defines a method to process the arguments from the command line (using the
    `argparse` package).

    .. note::

        You probably do not need to interact with this class directly.
        Instead, you can use the command line API as it is described in the
        :doc:`User Guide </guides/user_guide>` or the python API
        :class:`~pyDeltaRCM.preprocessor.Preprocessor`.

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

        # process command line to a dictionary
        cli_dict = self.process_arguments()

        # process the input file to a dictionary (or empty if none)
        if cli_dict['config'] is not None:
            input_file = cli_dict['config']
            yaml_dict = self.open_input_file_to_dict(input_file)
            self._input_file = input_file  # fill field
        else:
            yaml_dict = {}

        # combine the dicts into a single config
        self.config_dict = {}
        self.config_dict.update(yaml_dict)
        self.config_dict.update(cli_dict)

        # construct file list (expansions)
        self.construct_file_list()

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
        parser.add_argument(
            '--version', action='version',
            version=_ver, help='Print pyDeltaRCM version.')

        # parse and convert to dict
        args = parser.parse_args()
        args_dict = vars(args)

        # convert arguments to valid types for preprocessor
        if not (args_dict['timesteps'] is None):
            args_dict['timesteps'] = int(args_dict['timesteps'])
        else:
            args_dict.pop('timesteps')
        if not (args_dict['time'] is None):
            args_dict['time'] = float(args_dict['time'])
        else:
            args_dict.pop('time')
        if not (args_dict['time_years'] is None):
            args_dict['time_years'] = float(args_dict['time_years'])
        else:
            args_dict.pop('time_years')

        # set (or remove) defaults as needed
        if (args_dict['parallel'] is None):
            # if not given remove the key from cli spec
            args_dict.pop('parallel')
        if (args_dict['If'] is None):
            # if not given remove the key from cli spec
            args_dict.pop('If')

        return args_dict


class Preprocessor(BasePreprocessor):
    """Python high level api.

    This is the python high-level API class that is callable from a python
    script. For complete documentation on the API configurations, see the
    :doc:`User Guide </guides/user_guide>`.

    The class gives a way to configure and run multiple jobs from a python
    script.

    Examples
    --------

    To configure a set of jobs (or a single job), instantiate the python
    preprocessor and then manually run the configured jobs:

    .. code::

        >>> pp = preprocessor.Preprocessor(input_file=p, timesteps=500)
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

        if not input_file and len(kwargs.keys()) == 0:
            raise ValueError('Cannot use Preprocessor with no arguments.')

        # process "command line" to a dict (already is)
        cli_dict = kwargs

        # process the input yaml file to a dict
        if input_file is None:
            yaml_dict = {}
        else:
            yaml_dict = self.open_input_file_to_dict(input_file)
            self._input_file = input_file  # fill field

        # combine the dicts into a single config
        self.config_dict = {}
        self.config_dict.update(yaml_dict)
        self.config_dict.update(cli_dict)

        # construct file list (expansions)
        self.construct_file_list()


def preprocessor_wrapper():
    """Wrapper for command line interface.

    The `entry_points` setup of a command line interface requires a function,
    so we use this simple wrapper to instantiate and run the jobs.

    This function creates an instance of the
    :obj:`~pyDeltaRCM.preprocessor.PreprocessorCLI` and calls the
    :meth:`~pyDeltaRCM.preprocessor.PreprocessorCLI.run_jobs` to execute all
    jobs configured in the preprocessor. In code:

    .. code:: python

        pp = PreprocessorCLI()
        pp.run_jobs()
    """
    pp = PreprocessorCLI()
    pp.run_jobs()


def _write_yaml_config_to_file(_config, _path):
    """Write a config to file in output folder.

    Write the entire yaml configuation for the configured job out to a
    file in the job output foler.

    .. note::

        This function is utilized by the BMI implementation of pyDeltaRCM as
        well. Please do not move.
    """
    def _write_parameter_to_file(f, varname, varvalue):
        """Write each line, formatted."""
        if varvalue is None:
            f.write(varname + ': ' + 'null' + '\n')
        else:
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

        \\widehat{RSLR} = (RSLR / 1000) \\cdot \\dfrac{1}{I_f \\cdot 365.25 \\cdot 86400}

    This conversion makes it such that when one real-world year has elapsed
    (:math:`I_f \\cdot 365.25 \\cdot 86400` seconds in model time), the relative
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


# make the connection for running the preprocessor directly
if __name__ == '__main__':

    preprocessor_wrapper()
