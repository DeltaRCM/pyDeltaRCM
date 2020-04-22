**********
User Guide
**********

Guide to users!


Configuring the input yaml file
===============================

The configuration for a pyDeltaRCM run is set up by a parameter set, described in the ``yaml`` markup format.
To configure a run, you should create a file called, for example, ``run_parameters.yml``. 
Inside this file you can specify parameters for your run, with each parameter on a new line. For example, if ``run_parameters.yml`` contained the line: 

.. code-block:: yaml

    S0: 0.005

then a :obj:`~pyDeltaRCM.deltaRCM_driver.pyDeltaRCM` model instance initialized with this file specified as ``input_file`` will have a slope of 0.005.
Multiple parameters can be specified line by line.

Default values are substituted for any parameter not explicitly given in the ``input_file`` ``.yaml`` file.
Default values are defined as:

.. literalinclude:: ../../../pyDeltaRCM/default.yml
   :language: yaml
   :linenos: 
