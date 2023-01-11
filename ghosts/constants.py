"""constants module

This module provides a number of all round constants.

Todo:
    * Can we get these constants from some official source?

"""


# Rough focal plane characteristics
LSST_CAMERA_EXTENT = [-0.35, 0.35, -0.35, 0.35]
LSST_CAMERA_PIXEL_SIZE = 10*1e-6  # 10 microns
LSST_CAMERA_FILL_FACTOR = 0.9  # >90% fill factor
LSST_CAM_PIXEL_SURFACE = LSST_CAMERA_PIXEL_SIZE*LSST_CAMERA_PIXEL_SIZE  # m^2
LSST_CAMERA_PIXEL_DENSITY = (1./LSST_CAM_PIXEL_SURFACE)*LSST_CAMERA_FILL_FACTOR  # number of pixel per square meter
LSST_CAMERA_PIXEL_DENSITY_MM2 = LSST_CAMERA_PIXEL_DENSITY/1e6
LSST_CAMERA_PIXEL_QE = 0.85  # data/sensor_reflectivity_qe_r+qe.xlsx
LSST_CAMERA_NOISE = 10  # 10 electrons

# Notes
# Filter reflection within the in-band wavelength shall be <0.05%
# R-band coating report - internal transmittance of internal reflections is 0.1%
# CCD Reflectance ~ 15%
# Lens reflectance ~ 0.25%

# from Aurelien's note
LSST_CAMERA_READOUT_NOISE = 5  # 5 electrons
LSST_CAMERA_DARK_CURRENT = 2  # electrons per second

# From Slack
# LSST_CAMERA_READOUT_NOISE = 5 to 15, poisson ~ 7.5
LSST_CAMERA_PIXEL_FULL_WELL = 75000  # 75k to 130k ~ 120k mostly >100k
# PTC_CALIB = 1.16 e-/ADU

# Distance from CCOB head to focal plane - tested in test_geom
CCOB_DISTANCE_TO_FOCAL_PLANE = 2.229424

# Approximate Camera geometry in meters
LSST_CAMERA_AMP_DX = 5.4125/1000.
LSST_CAMERA_AMP_DY = 21.665/1000.
LSST_CAMERA_CCD_DX = 43.333/1000.
LSST_CAMERA_RAFT_DX = 130./1000.
