# Beam intensity, photon energy and number of photons
import numpy as np
from scipy.constants import Planck, lambda2nu
from math import floor
from copy import deepcopy

# batoid dependencies to create ray vectors
import batoid
from ghosts.beam_configs import BEAM_CONFIG_0


# Functions
def get_photon_energy(beam_nu):
    """ Compute the energy of a photon at a given frequency

    Parameters
    ----------
    beam_nu : `float`
        light frequency in Hz

    Returns
    -------
    photon_energy : `float`
        the photon energy in Joules
    """
    photon_energy = Planck * beam_nu
    return photon_energy


def get_n_phot_for_power(beam_power, beam_nu):
    """ Compute the number of photons in a beam of light of a given power for a given frequency

    Parameters
    ----------
    beam_power : `float`
        light beam power in nanowatt
    beam_nu : `float`
        light frequency in Hz

    Returns
    -------
    n_photon : `int`
        a number of photons
    """
    photon_energy = get_photon_energy(beam_nu)
    n_photon = beam_power / photon_energy
    return n_photon


def get_n_phot_for_power_nw_wl_nm(beam_power, wl):
    """ Compute the number of photons in a beam of light of a given power for a given wavelength

    Parameters
    ----------
    beam_power : `float`
        light beam power in nanowatt
    wl : `int`
        light wavelength

    Returns
    -------
    n_photon : `int`
        a number of photons
    """
    n_photons = floor(get_n_phot_for_power(beam_power * 1e-9, lambda2nu(wl * 1e-9)))
    return n_photons


def beam(beam_config=BEAM_CONFIG_0):
    """ Generates a beam of ligth rays to be used for a simulation

    Parameters
    ----------
    beam_config : `dict`
        a dictionary with the light beam configuration, see :ref:`beam_configs`.

    Returns
    -------
    rays : `batoid.RayVectors`
        a light beam as many photons of the requested wave length
    """
    radius = beam_config['radius']
    x_offset = beam_config['x_offset']
    y_offset = beam_config['y_offset']
    wl = beam_config['wl']
    n_photons = beam_config['n_photons']

    r2 = np.random.uniform(low=0, high=radius * radius, size=n_photons)  # radius
    theta = np.random.uniform(low=0, high=2 * np.pi, size=n_photons)  # angle

    rays_x = np.sqrt(r2) * np.cos(theta) + x_offset
    rays_y = np.sqrt(r2) * np.sin(theta) + y_offset
    rays_z = np.zeros(n_photons)

    rays_v = batoid.utils.normalized(np.array([0., 0., 1])) / 1.000277
    rays_vx = np.ones(n_photons) * rays_v[0]
    rays_vy = np.ones(n_photons) * rays_v[1]
    rays_vz = np.ones(n_photons) * rays_v[2]

    rays_wl = wl  # wavelength
    rays_t = 1  # the ray time, used by Batoid to propagate light
    # Batoid that is!
    rays = batoid.RayVector(rays_x, rays_y, rays_z, rays_vx, rays_vy, rays_vz, rays_wl, rays_t)

    return rays


# define function to generate a round beam of light
def simple_beam(x_offset=0.1, y_offset=0, wl=500e-9, n_photons=1000):
    """ Proxy to generate a simple beam of light rays to be used for a simulation
    
    Offsets are with respect to the default beam configuration, see `BEAM_CONFIG_0` at :ref:`beam_configs`.

    Parameters
    ----------
    x_offset : `float`
        the beam offset along x in meters
    y_offset : `float`
        the beam offset along y in meters
    wl : `int`
        the light wavelength in nanometers
    n_photons : `int`
        the number of light rays to generate

    Returns
    -------
    rays : `batoid.RayVectors`
        a light beam as many photons of the requested wave length
    """
    beam_config = deepcopy(BEAM_CONFIG_0)
    beam_config['x_offset'] = x_offset
    beam_config['y_offset'] = y_offset
    beam_config['wl'] = wl
    beam_config['n_photons'] = n_photons
    return beam(beam_config)
