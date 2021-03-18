.. api:

=============
API reference
=============

The :obj:`DeltaModel` class, defined in `pyDeltaRCM.model`, is the main class of pyDeltaRCM, which provides the object that is manipulated to evolve the numerical delta model.
This class uses "mix-in" classes, which are defined in separate files (Python `modules`), that break out logically based on the various stages of model use, and components of the model iteration sequence.
Most model functionality is organized into the various mix-in classes, that are then inherited by the `DeltaModel`. 
Additionally, several routines of the model are organized into module-level functions, which are "jitted" via the `numba <https://numba.pydata.org/>`_  code optimizer library for Python.

This index lists the `pyDeltaRCM` organization, hopefully providing enough information to begin to determine where various components of the model are implemented.
The index includes model classes, methods, and attributes, as well as additionally utility classes and functions.

.. toctree::
   :maxdepth: 2

   model/index
   preprocessor/index
   iteration_tools/index
   init_tools/index
   water_tools/index
   sed_tools/index
   shared_tools/index
   hook_tools/index
   debug_tools/index


References
----------

* :ref:`modindex`
* :ref:`search`


Search the Index
==================

* :ref:`genindex`
