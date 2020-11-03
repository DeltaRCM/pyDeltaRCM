=======================
Working with subsidence
=======================

What is subsidence anyway? Subsidence is basically the downward vertical
movement of the ground. There are many direct and indirect causes of subsidence,
check out the `Wikipedia page <https://en.wikipedia.org/wiki/Subsidence>`_
to get an overview.

Turning on Subsidence in pyDeltaRCM
===================================

To configure a pyDeltaRCM model with subsidence, the yaml parameter,
``toggle_subsidence`` must be set to ``True``.

Configuring Lateral Extents of Subsidence
=========================================

There are two yaml parameters associated with the lateral extents of the
subsiding region in pyDeltaRCM: ``theta1`` and ``theta2``. ``theta1`` is
specified in radians and represents the left boundary of the subsiding region.
The angle in radians is expressed relative to a datum aligned with the inlet
channel. ``theta2`` determines the right boundary of the subsiding region.

If, for example we wanted the left half of the domain to subside, we would
write our yaml file with the following parameters:

.. code:: yaml

    toggle_subsidence: True
    theta1: -1.5707963267948966
    theta2: 0

This specifies ``theta1`` as negative 90 degrees, and ``theta2`` as 0 degrees.
Doing so, will generate a domain that has the a subsiding region as shown in
yellow in the figure below.

.. plot:: userguide/subsidence_region.py

Configuring When Subsidence Occurs
==================================

To control when subsidence begins over the course of the model run, there is
another yaml parameter, ``start_subsidence``. The value assigned to this
parameter controls the time (in seconds) at which the subsidence will be turned
"on" in the model.

Configuring The Rate of Subsidence
==================================

.. todo:: Add documentation here.


Advanced Subsidence Configurations
==================================

.. todo:: Bit about subclassing and custom configurations
