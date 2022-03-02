from copy import deepcopy

# Default geometry, all aligned, as the basic example
GEOM_CONFIG_0 = {'geom_id': 0,
                 'L1': {'shifts': [0.0, 0.0, 0.0], 'rotations': [0., 0., 0.]},
                 'L2': {'shifts': [0.0, 0.0, 0.0], 'rotations': [0., 0., 0.]},
                 'L3': {'shifts': [0.0, 0.0, 0.0], 'rotations': [0., 0., 0.]},
                 'Filter': {'shifts': [0.0, 0.0, 0.0], 'rotations': [0., 0., 0.]},
                 'Detector': {'shifts': [0.0, 0.0, 0.0], 'rotations': [0., 0., 0.]}}

# Rotate L1 over X axis by 0.1 degrees
GEOM_CONFIG_1 = deepcopy(GEOM_CONFIG_0)
GEOM_CONFIG_1['geom_id'] = 1
GEOM_CONFIG_1['L1']['rotations'] = [0.1, 0., 0.]

# Shift L1 along X axis by 1 mm
GEOM_CONFIG_2 = deepcopy(GEOM_CONFIG_0)
GEOM_CONFIG_2['geom_id'] = 2
GEOM_CONFIG_2['L1']['shifts'] = [0.001, 0., 0.]
