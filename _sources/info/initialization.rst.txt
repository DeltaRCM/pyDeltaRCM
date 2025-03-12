**************
Initialization
**************

.. currentmodule:: pyDeltaRCM.init_tools

There are many built-in parameters for setting up pyDeltaRCM runs.


Domain configuration
====================

The domain configuration is controlled by a series of input configuration parameters. 
Conceptually, we can consider the domain of pyDeltaRCM to be made up of an *inlet* along the top edge of the model domain and bounded by a strip of land, and the *basin*.

.. plot:: init_tools/domain_parameters.py

By default, the basin and inlet are both flat and have the same depth. However these characteristics can be controlled by a variety of input parameters explained below. 
Moreover, any complex initial geometry can be configured by the use of hooks; see an example here :doc:`../examples/slight_slope`.

Domain size
-----------

The domain size is controlled by the `Length` and `Width` YAML parameters. 

.. plot:: init_tools/domain_size_compare.py

.. hint:: 

       The size of the computational domain is determined by the size of the smallest size of the domain. The value for `Width` should almost always be 2 times `Length` to minimize unecessary calcualtions outside the valid computational domain.

Importantly, the number of cells in the domain is then a function of the domain size in meters and the grid spacing :obj:`dx`. 
Because the number of grid cells is inverse nonlinearly related to the grid spacing (:math:`numcells = 1/dx^2`), the model runtime is sensitive to the choice of `dx` relative to `Length` and `Width`.


Basin depth and inlet depth
---------------------------

Inlet depth is controlled by :obj:`h0`.
If no argument is given in the configuration for the parameter :obj:`hb`, then the basin depth is determined to be the same as the inlet depth. 
However, these depths can be controlled independently.

.. plot:: init_tools/domain_basin_inlet_depth.py


Inlet width and length
----------------------

The inlet length and width are controlled by an inlet length parameter :obj:`L0_meters` and an inlet width parameter :obj:`N0_meters`. 
In implementation, the input length and width are rounded to the nearest grid location (a function of :obj:`dx`).

.. plot:: init_tools/domain_inlet_geometry.py



Input water and sediment
========================

The input flow velocity is determined by :obj:`u0`.

The input water discharge is thus fully constrained by the inlet geometry and the inlet flow velocity as:

.. math::
       Q_{w0} = h_0 \times N_0 \times u_0

.. hint::
       Keep in mind that :math:`N_0` is specified in meters as :obj:`N0_meters` in the input configuration. Ditto for the inlet length (:obj:`L0_meters`).


Input sediment discharge is then determined by a **percentage concentration** of sediment in the flow :obj:`C0_percent`.

.. math::
       Q_{s0} = Q_{w0} \times (C_{0percent}/100)

The volume of each parcel of water to be routed is then determined by the total water discharge divided by the number of parcels :obj:`Np_water`.
The volume of each parcel of sediment to be routed is then determined by the total sediment discharge divided by the number of parcels :obj:`Np_sed`.

The number of sand and mud parcels are then determined as `Np_sed` times :obj:`f_bedload` and `Np_sed` times (1-:obj:`f_bedload`), respectively.
