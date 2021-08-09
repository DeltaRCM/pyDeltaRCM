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

Controlling Subsidence Behavior
===============================

Two yaml parameters are provided to give users some basic control over
subsidence behavior. The first is ``start_subsidence``, which defines *when*
subsidence begins in the model run. This parameter is set in terms of seconds,
and is set to begin on the step *following* the time step that brings the
model to ``time >= start_subsidence``. The second subsidence parameter is the
``subsidence_rate`` yaml parameter. This parameter defines the rate at which
the basin will subside in meters per second. The default subsiding region is
the entire delta basin with the exception of the inlet cells and the land cells
along boundary.

If, for example we wanted the basin to begin to subside after 2000 seconds
with a rate of 2e-10 m/s, we would write our yaml file with the following
parameters:

.. code:: yaml

    toggle_subsidence: True
    start_subsidence: 2000
    subsidence_rate: 2e-10

Advanced Subsidence Configurations
==================================

Subsidence behavior can be easily modified by :ref:`creating a subclass <customize_the_model>` and overwriting the :obj:`~pyDeltaRCM.init_tools.init_tools.init_subsidence` or :obj:`~pyDeltaRCM.iteration_tools.iteration_tools.apply_subsidence` methods, or even more simply the relevant model hooks.

An example of using subclassing to create a model setup with subsidence confined to only part of the model domain is included in the documentation:

* :doc:`/examples/subsidence_region`

