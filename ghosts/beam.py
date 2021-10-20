# Beam intensity, photon energy and number of photons
import numpy as np
from scipy.constants import Planck, lambda2nu, nu2lambda
from math import floor
from copy import copy

# batoid dependencies to create ray vectors
import batoid
from ghosts.beam_configs import BEAM_CONFIG_0


# Functions
def get_E_ph(nu):
    return Planck*nu

def get_n_phot_for_power(p, nu):
    E_ph = get_E_ph(nu)
    n_phot = p/E_ph
    return n_phot

def get_n_phot_for_power_nw_wl_nm(p, wl):
    return floor(get_n_phot_for_power(p*1e-9, lambda2nu(wl*1e-9)))

def beam(beam_config=BEAM_CONFIG_0):
    ''' Takes a beam configuration dictionnary
    Returns a batoid.RayVector of light rays
    '''
    c = '#ff7f00'
    radius = beam_config['radius']
    x_offset = beam_config['x_offset']
    y_offset = beam_config['y_offset']
    wl = beam_config['wl']
    n = beam_config['n_photons']

    r2 = np.random.uniform(low=0, high=radius * radius, size=n)  # radius
    theta = np.random.uniform(low=0, high=2 * np.pi, size=n)  # angle

    rays_x = np.sqrt(r2) * np.cos(theta) + x_offset
    rays_y = np.sqrt(r2) * np.sin(theta) + y_offset
    rays_z = np.zeros(n)

    rays_v = batoid.utils.normalized(np.array([0., 0., 1])) / 1.000277
    rays_vx = np.ones(n) * rays_v[0]
    rays_vy = np.ones(n) * rays_v[1]
    rays_vz = np.ones(n) * rays_v[2]

    rays_wl = wl
    rays_t = 1
    # Batoid that is!
    rays = batoid.RayVector(rays_x, rays_y, rays_z, rays_vx, rays_vy, rays_vz, rays_wl, rays_t)

    return rays

# define function to generate a round beam of light
def simple_beam(x_offset=0.1, y_offset=0, wl=500e-9, n=1000):
    beam_config = copy(BEAM_CONFIG_0)
    beam_config['x_offset'] = x_offset
    beam_config['y_offset'] = x_offset
    beam_config['wl'] = wl
    beam_config['n_photons'] = n
    return beam(beam_config)
