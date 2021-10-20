================================================================
Rubin LSST CCOB narrow beam ghost images simulation and analysis
================================================================

`ghosts` is intended to simulate and analyze ghost images produced by
the CCOB narrow beam on the Rubin LSST camera assembly. Ghosts images generated
by the reflection of part of the beam light on the coating of the optical components
may be used to derive the alignment constants of the optical elements.


ghosts image simulation and analysis
------------------------------------

The four main components are (as usual):

1. optical setup

2. simulation

3. analysis

4. plotter


File formats
------------

Ghosts image analysis results are the main output and are written into parquet files.
We might also want full images into FITS files at some point, and all that might end up
into HDF5 files.

1. 'parquet', as used to store pandas datafames, e.g., as produced by
`make_data_frame` or `compute_ghost_separations`

2. 'FITS' might be used to store images

3. 'HDF5' might be used to store images and pandas data frame

Quick examples
==============

Here are some very quick high-level examples.


Optical setup
-------------

Build and visualize the camera elements

.. code-block:: python

    # Create a new CCOB camera
    telescope = batoid.Optic.fromYaml("LSST_CCOB_r.yaml")
    # Apply the coating
    tweak_optics.make_optics_reflective(telescope)
    # Run simulation with standard beam config
    traceFull, rForward, rReverse, rays = simulator.run_simulation(telescope, beam_config=BEAM_CONFIG_0)
    simulation = [traceFull, rForward, rReverse, rays]
    # Visualize the setup
    plotter.plot_setup(telescope, simulation)

Documentation Contents
----------------------

.. toctree::
   :includehidden:
   :maxdepth: 3

   install
   ghosts
   
