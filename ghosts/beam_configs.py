"""beam_configs module

This module is used to define the CCOB-NB beam configuration through dictionaries.

"""

from copy import deepcopy

# Default beam configs
BEAM_CONFIG_0 = {'beam_id': 0, 'wl': 500e-9, 'n_photons': 1000, 'radius': 0.00125,
                 'x_offset': 0., 'y_offset': 0, 'z_offset': 2.7974,
                 'z_euler': 0., 'y_euler': 0., 'x_euler': 0.}

BEAM_CONFIG_1 = deepcopy(BEAM_CONFIG_0)
BEAM_CONFIG_1['beam_id'] = 1
BEAM_CONFIG_1['x_offset'] = 0.1

FAST_BEAM_CONFIG_1 = deepcopy(BEAM_CONFIG_1)
FAST_BEAM_CONFIG_1['beam_id'] = 1000
FAST_BEAM_CONFIG_1['n_photons'] = 10

BEAM_CONFIG_2 = deepcopy(BEAM_CONFIG_0)
BEAM_CONFIG_2['beam_id'] = 2
BEAM_CONFIG_2['y_offset'] = 0.1

# spread ghosts
BEAM_CONFIG_3 = deepcopy(BEAM_CONFIG_0)
BEAM_CONFIG_3['beam_id'] = 3
BEAM_CONFIG_3['x_offset'] = -0.1
BEAM_CONFIG_3['y_offset'] = 0.3
BEAM_CONFIG_3['x_euler'] = 3
BEAM_CONFIG_3['y_euler'] = 6

BEAM_CONFIG_4 = deepcopy(BEAM_CONFIG_0)
BEAM_CONFIG_3['beam_id'] = 4
BEAM_CONFIG_4['x_offset'] = -0.25
BEAM_CONFIG_4['y_offset'] = 0.25
BEAM_CONFIG_4['x_euler'] = 3
BEAM_CONFIG_4['y_euler'] = 6

# 4 symmetrical spread ghosts beam configurations
BEAM_CONFIG_10 = deepcopy(BEAM_CONFIG_0)
BEAM_CONFIG_10['beam_id'] = 10
BEAM_CONFIG_10['n_photons'] = 100
BEAM_CONFIG_10['x_offset'] = -0.3
BEAM_CONFIG_10['y_offset'] = 0.6
BEAM_CONFIG_10['x_euler'] = 17
BEAM_CONFIG_10['y_euler'] = 6

BEAM_CONFIG_11 = deepcopy(BEAM_CONFIG_0)
BEAM_CONFIG_11['beam_id'] = 11
BEAM_CONFIG_11['n_photons'] = 100
BEAM_CONFIG_11['x_offset'] = +0.3
BEAM_CONFIG_11['y_offset'] = 0.6
BEAM_CONFIG_11['x_euler'] = 17
BEAM_CONFIG_11['y_euler'] = -6

BEAM_CONFIG_12 = deepcopy(BEAM_CONFIG_0)
BEAM_CONFIG_12['beam_id'] = 12
BEAM_CONFIG_12['n_photons'] = 100
BEAM_CONFIG_12['x_offset'] = +0.3
BEAM_CONFIG_12['y_offset'] = -0.6
BEAM_CONFIG_12['x_euler'] = -17
BEAM_CONFIG_12['y_euler'] = -6

BEAM_CONFIG_13 = deepcopy(BEAM_CONFIG_0)
BEAM_CONFIG_13['beam_id'] = 13
BEAM_CONFIG_13['n_photons'] = 100
BEAM_CONFIG_13['x_offset'] = -0.3
BEAM_CONFIG_13['y_offset'] = -0.6
BEAM_CONFIG_13['x_euler'] = -17
BEAM_CONFIG_13['y_euler'] = +6

# Beam configuration sets
BASE_BEAM_SET = [BEAM_CONFIG_10, BEAM_CONFIG_11, BEAM_CONFIG_12, BEAM_CONFIG_13]
