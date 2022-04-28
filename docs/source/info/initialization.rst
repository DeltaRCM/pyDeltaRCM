**************
Initialization
**************

.. currentmodule:: pyDeltaRCM.init_tools

There are many built-in parameters for setting up pyDeltaRCM runs.


Domain configuration
====================

Domain size
-----------

The domain size is controlled by the `Length` and `Width` YAML parameters. 

.. plot:: init_tools/domain_size_compare.py

.. hint:: 

       The size of the computational domain is determined by the size of the smallest size of the domain. The value for `Width` should almost always be 2 times `Length` to minimize unecessary calcualtions outside the valid computational domain.

Basin depth and inlet depth
---------------------------

.. note:: example showing the effects of :obj:`hb` and :obj:`h0`.


Inlet width and length
----------------------

.. note:: example showing the effects of :obj:`L0_meters` and :obj:`N0_meters`.

