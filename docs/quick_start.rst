.. _quick_start:

Quick start guide
*****************

Run a simulation
================

It's easy to setup and run a quick simulation

Visualization
=============
so many nice plots

Analysis
========

that is straightforward

Notebooks
=========
list of notebooks

Scripts
=======
`scripts/run_production.py`
---------------------------
Script used by the CI to test the code, it runs a small production, defined as a set of simulations and the analysis of
ghosts spots, then saved in a parquet file. The `ghosts.beam_configs.BASE_BEAM_SET` (see :ref:`beam_configs`) is used on a number of random geometries generated on
the fly. The script takes the number of random geometries to generate and a production name.

.. code-block::

    > python scripts/run_production.py 5 testprod

`scripts/run_fit_example.py`
----------------------------
Script to show how to implement the fit of the camera geometry is a very simple way.
Takes the number of calls and precision as arguments.

.. code-block::

    > python -i scripts/run_fit_example.py 100 1e-6

`scripts/run_fit_example_oo.py`
-------------------------------
Script to show how to implement the fit of the camera geometry is a very simple way, but using a class for cleaner code.
Takes the number of calls and precision as arguments.

.. code-block::

    > python -i scripts/run_fit_example_oo.py 100 1e-6
