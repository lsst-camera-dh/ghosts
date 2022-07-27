from copy import deepcopy

# Default beam configs
BEAM_CONFIG_0 = {'beam_id': 0, 'wl': 500e-9, 'n_photons': 1000, 'radius': 0.00125,
                 'x_offset': 0., 'y_offset': 0, 'z_offset': 2.2,
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
