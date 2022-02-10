from copy import copy

# Default beam configs
BEAM_CONFIG_0 = {'id': 0, 'wl': 500e-9, 'n_photons': 1000,
                 'x_offset': 0., 'y_offset': 0, 'radius': 0.00125,
                 'theta': 0., 'phi': 0.}

BEAM_CONFIG_1 = copy(BEAM_CONFIG_0)
BEAM_CONFIG_1['id'] = 1
BEAM_CONFIG_1['x_offset'] = 0.1

FAST_BEAM_CONFIG_1 = copy(BEAM_CONFIG_1)
FAST_BEAM_CONFIG_1['n_photons'] = 10

BEAM_CONFIG_2 = copy(BEAM_CONFIG_0)
BEAM_CONFIG_2['id'] = 2
BEAM_CONFIG_2['y_offset'] = 0.1
