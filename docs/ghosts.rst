.. _ghosts:


API Documentation for ghosts
*******************************

Stuff

camera interface
===================
.. _camera:

`ghosts.camera`
---------------------

.. automodule:: ghosts.camera
    :members:
    :undoc-members:

beam configuration interface
============================
.. _beam_configs:

`ghosts.beam_configs`
---------------------
Beam configurations are defined as simple dictionaries with minimal information.

.. code-block::

    BEAM_CONFIG_0 = {'beam_id': 0, 'wl': 500e-9, 'n_photons': 1000, 'radius': 0.00125,
                     'x_offset': 0., 'y_offset': 0, 'z_offset': 2.2,
                     'z_euler': 0., 'y_euler': 0., 'x_euler': 0.}

beam interface
===================
.. _beam:

`ghosts.beam`
---------------------

.. automodule:: ghosts.beam
    :members:
    :undoc-members:

geom configuration interface
============================
.. _geom_configs:

`ghosts.geom_configs`
---------------------
Geometry configurations are defined as dictionaries with translations and rotations
for each of the optical element

.. code-block::

    GEOM_CONFIG_0 = {'geom_id': 0,
                     'L1_dx': 0.0, 'L1_dy': 0.0, 'L1_dz': 0.0, 'L1_rx': 0.0, 'L1_ry': 0.0, 'L1_rz': 0.0,
                     'L2_dx': 0.0, 'L2_dy': 0.0, 'L2_dz': 0.0, 'L2_rx': 0.0, 'L2_ry': 0.0, 'L2_rz': 0.0,
                     'L3_dx': 0.0, 'L3_dy': 0.0, 'L3_dz': 0.0, 'L3_rx': 0.0, 'L3_ry': 0.0, 'L3_rz': 0.0,
                     'Filter_dx': 0.0, 'Filter_dy': 0.0, 'Filter_dz': 0.0,
                     'Filter_rx': 0.0, 'Filter_ry': 0.0, 'Filter_rz': 0.0,
                     'Detector_dx': 0.0, 'Detector_dy': 0.0, 'Detector_dz': 0.0,
                     'Detector_rx': 0.0, 'Detector_ry': 0.0, 'Detector_rz': 0.0}

geom interface
===================
.. _geom:

`ghosts.geom`
---------------------

.. automodule:: ghosts.geom
    :members:
    :undoc-members:

tweak optics interface
======================

`ghosts.tweak_optics`
---------------------

.. automodule:: ghosts.tweak_optics
    :members:
    :undoc-members:

reflectivity interface
======================

`ghosts.reflectivity`
---------------------

.. automodule:: ghosts.reflectivity
    :members:
    :undoc-members:

simulator interface
===================

`ghosts.simulator`
---------------------

.. automodule:: ghosts.simulator
    :members:
    :undoc-members:

analysis interface
==================

`ghosts.analysis`
---------------------

.. automodule:: ghosts.analysis
    :members:
    :undoc-members:

plotter interface
=================

`ghosts.plotter`
---------------------

.. automodule:: ghosts.plotter
    :members:
    :undoc-members:

tools interface
===============

`ghosts.tools`
---------------------

.. automodule:: ghosts.tools
    :members:
    :undoc-members: