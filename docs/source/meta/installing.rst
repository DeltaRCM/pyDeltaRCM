************
Installing
************

We recommend installing *pyDeltaRCM* in a virtual environment.
That said, *pyDeltaRCM* depends on a small number of packages (:ref:`list of dependencies <dependencies-list>`), many of which are likely already in a Python user/developer's regular use, so it's probably safe to install *pyDeltaRCM* in your base environment, too.


Installing
==========

We describe installation flavors for both users and developers below.

.. hint::

    If you are looking to make any modifications to the model source code, you should follow the developer instructions.

We suggest using the Anaconda Python distribution, which you can obtain via `the project website <https://www.anaconda.com/products/individual>`_.

Before proceeding, you may wish to create a virtual environment for the *pyDeltaRCM* project.
With Anaconda on Linux:

.. code:: console

    $ conda create -n deltarcm python=3
    $ conda activate deltarcm

For more informtaion, see `this guide <https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#activating-an-environment>`_ for help on creating and activating a virtual environment with Anaconda on other platforms.
See `this helpful guide <https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/#creating-a-virtual-environment>`_ for creating virtual environments with `venv` if you do not use Anaconda.


User installation
-----------------

For a user installation, simply install from the pypi package repository:

.. code:: console

    $ pip install pyDeltaRCM

.. note::

    You may need to `first install <https://pip.pypa.io/en/stable/installing/>`_ `pip`.


.. _dev-install:

Developer installation
----------------------

For a developer installation, you should first fork the repository on Github.
This will allow you to submit suggestions and contribute to *pyDeltaRCM*.

.. note::

    You do not *need* to create a fork if your are just testing, but it may save you time and headache down the road. If you choose not to, just use the main repository url below (https://github.com/DeltaRCM/pyDeltaRCM.git).

First, you will need to `install git <https://git-scm.com/book/en/v2/Getting-Started-Installing-Git>`_  if you do not already have it.
Then, download or clone your fork of the project:

.. code:: console

    $ git clone https://github.com/<your-username>/pyDeltaRCM.git

Then, with current working directory as the root of the repository (e.g., ``cd pyDeltaRCM``), run the following commands:

.. code:: console

    $ pip install -r requirements.txt
    $ pip install -r requirements-docs.txt
    $ pip install -r requirements-test.txt
    $ pip install -e .

To check installation, run the complete test suite with:

.. code:: console

    $ pytest --mpl --mpl-baseline-path=tests/imgs_baseline

Finally, add the `upstream` repository to your `remote` repository list:

.. code:: console

    $ git remote add upstream https://github.com/DeltaRCM/pyDeltaRCM.git

You can build a local copy of the documentation with:

.. code:: console

    $ (cd docs && make html)


Next steps
==========

Consider reading through the :doc:`10-minute tutorial </guides/10min>` or the :doc:`User Guide </guides/user_guide>` to help get you started using *pyDeltaRCM*.


.. _dependencies-list:

Dependencies
============

.. literalinclude:: ../../../requirements.txt
   :linenos:
