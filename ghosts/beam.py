# Beam intensity, photon energy and number of photons
import numpy as np
from scipy.constants import Planck, lambda2nu
from math import floor
from copy import deepcopy
import pandas as pd

# batoid dependencies to create ray vectors
import batoid
from ghosts.beam_configs import BEAM_CONFIG_0



def to_panda(beam_config):
    """ Convert a beam configuration dictionary to a panda data frame

    Indexing is done using the beam configuration `beam_id`.

    Parameters
    ----------
    beam_config : `dict`
        a dictionary with beam characteristics

    Returns
    -------
    data_frame : `pandas.DataFrame`
        a `pandas` data frame with beam information
    """
    data_frame = pd.DataFrame(data=beam_config, index=[beam_config['beam_id']])
    return data_frame


def to_dict(beam_frame):
    """ Convert a beam panda data frame to a dictionary of use with `simulator`

    The beam data frame is expected to have only one beam configuration.

    Parameters
    ----------
    beam_frame : `pandas.DataFrame`
        a `pandas` data frame with one beam configuration

    Returns
    -------
    beam_config : `dict`
        a dictionary with a beam configuration
    """
    beam_id = beam_frame['beam_id'].to_list()[0]
    beam_config = beam_frame.to_dict('index')[beam_id]
    return beam_config


def concat_frames(beam_frame_list):
    """ Concatenates beam configuration data frames within one table

     Parameters
     ----------
     beam_frame_list : `list` of `pandas.DataFrame`
         a list of beam configuration data frames

     Returns
     -------
     beam_concat : `pandas.DataFrame`
        a `pandas` data frame with several configurations of beams
     """
    tmp_concat = pd.concat(beam_frame_list)
    beam_concat = tmp_concat.fillna(0)
    beam_concat.sort_values('beam_id')
    return beam_concat


def concat_dicts(beam_dict_list):
    """ Concatenates geometry configuration dictionaries into a data frame

     Parameters
     ----------
     beam_dict_list : `list` of `dict`
         a list of beam configuration dictionaries

     Returns
     -------
     beam_concat : `pandas.DataFrame`
        a `pandas` data frame with several configurations of beams
     """
    frames = list()
    for one in beam_dict_list:
        frames.append(to_panda(one))
    beam_concat = concat_frames(frames)
    return beam_concat


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


def beam_on(beam_config=BEAM_CONFIG_0):
    """ Generates a beam of ligth rays to be used for a simulation

    .. todo::
        `beam_on` should implement usage of theta and phi incident angles!

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
    return beam_on(beam_config)


def build_translation_set(base_beam_config, axis, shifts_list, base_id=0):
    """ Build a set of beams for the given list of translations

    Parameters
    ----------
    base_beam_config : `dict`
        the base beam configuration dictionary to start from
    axis : `string`
        the name of the translation axis, usually x or y
    shifts_list : `list` of `float`
        the list of distances to scan in meters
    base_id : `int`
        the id of the first beam configuration created, following ids will be `id+1`

    Returns
    -------
     beams : `list` of `geom_config`
        a list of beam configuration dictionaries
    """
    beams = list()
    for i, shift in enumerate(shifts_list):
        beam_config = deepcopy(base_beam_config)
        beam_config['beam_id'] = base_id + i
        beam_config[f'{axis}_offset'] = shift
        beams.append(beam_config)
    return beams


def build_rotation_set(base_beam_config, angle_name, angles_list, base_id=0):
    """ Build a set of beam configurations for the given list of rotations
    starting from the given beam configuration

    Parameters
    ----------
    base_beam_config : `dict`
        the base beam configuration dictionary to start from
    angle_name : `string`
        either "theta" or "phi"
    angles_list : `list` of `float`
        the list of angles to scan in degrees
    base_id : `int`
        the id of the first beam configuration created, following ids will be `id+1`

    Returns
    -------
     beams : `list` of `geom_config`
        a list of geometry configuration dictionaries
    """
    beams = list()
    for i, angle in enumerate(angles_list):
        beam_config = deepcopy(base_beam_config)
        beam_config['beam_id'] = base_id + i
        beam_config[angle_name] = angle
        beams.append(beam_config)
    return beams
