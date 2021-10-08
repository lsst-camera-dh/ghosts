# Rough focal plane characteristics
LSST_CAMERA_PIXEL_SIZE = 10*1e-6  # 10 microns
LSST_CAMERA_FILL_FACTOR = 0.9  # >90% fill factor
LSST_CAM_PIXEL_SURFACE = (LSST_CAMERA_PIXEL_SIZE/2.)*(LSST_CAMERA_PIXEL_SIZE/2.)  # m^2
LSST_CAMERA_PIXEL_DENSITY = 1./LSST_CAM_PIXEL_SURFACE*LSST_CAMERA_FILL_FACTOR  # number of pixel per square meter
LSST_CAMERA_PIXEL_DENSITY_MM2 = LSST_CAMERA_PIXEL_DENSITY/1e6
LSST_CAMERA_PIXEL_QE = 0.9  # wl dependant
LSST_CAMERA_NOISE = 10  # 10 electrons

# Notes
# Filter reflection within the in-band wavelength shall be <0.05%
# R-band coating report - internal transmittance of internal reflections is 0.1%
