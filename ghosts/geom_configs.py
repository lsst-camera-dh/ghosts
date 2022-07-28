"""geom_configs module

This module is used to define the geometry of telescope through dictionaries.

"""
from copy import deepcopy

# Default geometry, all aligned, as the basic example
GEOM_CONFIG_0 = {'geom_id': 0,
                 'L1_dx': 0.0, 'L1_dy': 0.0, 'L1_dz': 0.0, 'L1_rx': 0.0, 'L1_ry': 0.0, 'L1_rz': 0.0,
                 'L2_dx': 0.0, 'L2_dy': 0.0, 'L2_dz': 0.0, 'L2_rx': 0.0, 'L2_ry': 0.0, 'L2_rz': 0.0,
                 'L3_dx': 0.0, 'L3_dy': 0.0, 'L3_dz': 0.0, 'L3_rx': 0.0, 'L3_ry': 0.0, 'L3_rz': 0.0,
                 'Filter_dx': 0.0, 'Filter_dy': 0.0, 'Filter_dz': 0.0,
                 'Filter_rx': 0.0, 'Filter_ry': 0.0, 'Filter_rz': 0.0,
                 'Detector_dx': 0.0, 'Detector_dy': 0.0, 'Detector_dz': 0.0,
                 'Detector_rx': 0.0, 'Detector_ry': 0.0, 'Detector_rz': 0.0}


# Rotate L1 over X axis by 0.1 degrees
GEOM_CONFIG_1 = deepcopy(GEOM_CONFIG_0)
GEOM_CONFIG_1['geom_id'] = 1
GEOM_CONFIG_1['L1_rx'] = 0.1

# Shift L1 along X axis by 1 mm
GEOM_CONFIG_2 = deepcopy(GEOM_CONFIG_0)
GEOM_CONFIG_2['geom_id'] = 2
GEOM_CONFIG_2['L1_dx'] = 0.001
