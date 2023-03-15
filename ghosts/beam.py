"""beam module

This module provides tools to manipulate beam geometries,
and generate light rays as `batoid.RayVector`
"""

# Beam intensity, photon energy and number of photons
import numpy as np
from scipy.constants import Planck, lambda2nu
from scipy.spatial.transform import Rotation as transform_rotation
from math import floor, cos, sin, radians, atan, degrees
from copy import deepcopy
import pandas as pd

# batoid dependencies to create ray vectors
import batoid
from ghosts.beam_configs import BEAM_CONFIG_0
from ghosts.constants import CCOB_DISTANCE_TO_FOCAL_PLANE


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
    frames = []
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


def _get_angles_to_center(x_offset, y_offset):
    """ Compute the Euler angles to point the beam at the camera center
    given a beam position

    Parameters
    ----------
    x_offset : `float`
        beam position on the X-axis
    y_offset : `float`
        beam position on the Y-axis

    Returns
    -------
    x_euler, y_euler : `tuple` of `floats`
        Euler angles around X and Y axis as a tuple
    """
    y_euler = -atan(x_offset/CCOB_DISTANCE_TO_FOCAL_PLANE)
    x_euler = atan(y_offset/CCOB_DISTANCE_TO_FOCAL_PLANE)
    return degrees(x_euler), degrees(y_euler)


def _get_angles_to_xy(x_pos, y_pos, x_offset, y_offset):
    """ Compute the Euler angles to point the beam at a given point
     on the camera given a beam position

    Parameters
    ----------
    x_pos : `float`
        position on the X-axis of the camera plane
    y_pos : `float`
        position on the Y-axis of the camera plane
    x_offset : `float`
        beam position on the X-axis
    y_offset : `float`
        beam position on the Y-axis

    Returns
    -------
    x_euler, y_euler : `tuple` of `floats`
        Euler angles around X and Y axis as a tuple
    """
    dx = x_offset - x_pos
    dy = y_offset - y_pos
    return _get_angles_to_center(dx, dy)


def point_beam_to_target(beam_config, target_x=0., target_y=0.):
    """ Compute the Euler angles to point the beam at the camera center
    given a beam position

    Parameters
    ----------
    beam_config : `dict`
        a dictionary with the light beam configuration, see :ref:`beam_configs`.
    target_x : `float`
        target position on the X-axis of the camera plane
    target_y : `float`
        target position on the Y-axis of the camera plane

    Returns
    -------
    new_beam : `dict`
        a dictionary with the light beam configuration (see :ref:`beam_configs`)
        that points toward the requested position on the camera plane
    """
    # make the new config
    new_beam = deepcopy(beam_config)
    new_beam['x_euler'], new_beam['y_euler'] = \
        _get_angles_to_xy(target_x, target_y, new_beam['x_offset'], new_beam['y_offset'])
    return new_beam


def beam_on(beam_config):
    """ Generates a beam of light rays to be used for a simulation

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
    z_offset = beam_config['z_offset']
    rz = beam_config['z_euler']
    ry = beam_config['y_euler']
    rx = beam_config['x_euler']
    wl = beam_config['wl']
    n_photons = beam_config['n_photons']

    # draw uniform distribution from a disc
    r2 = np.random.uniform(low=0, high=radius * radius, size=n_photons)  # radius
    disc_theta = np.random.uniform(low=0, high=2 * np.pi, size=n_photons)  # angle

    # apply offsets
    rays_x = np.sqrt(r2) * np.cos(disc_theta) + x_offset
    rays_y = np.sqrt(r2) * np.sin(disc_theta) + y_offset
    rays_z = np.ones(n_photons)*z_offset

    # set direction, start from straight light, then rotate from Euler angles
    straight_ray = np.array([0., 0., 1])
    # watch out here, switching rx and ry on purpose (this makes rotation and offsets consistent in visualization)
    rot_zyx = transform_rotation.from_euler('zxy', [rz, rx, ry], degrees=True)
    rotated_ray = rot_zyx.apply(straight_ray)
    # put direction in the form batoid likes (at speed of light)
    rays_v = batoid.utils.normalized(rotated_ray) / 1.000277
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


def rotate_config_to_ub(beam_config, u, b):
    """ Rotate a beam configuration to given b and u CCOB motor angles

    Email from Andy:
    - Check that U=0, B rotates around the Y axis, B>0 -> +X, B<0 -> -X
    - Check that U>0 rotates around -Z in the right-handed sense.

    Euler angles in extrinsic rotations
    https://en.wikipedia.org/wiki/Euler_angles#Conventions_by_extrinsic_rotations

    Parameters
    ----------
    beam_config : `ghost.beam_config`
        a beam configuration as a dictionary
    u : `float`
        the value of the rotation angle in degrees around "-Z" (when b=0)
    b : `float`
        the value of the rotation angle in degrees around "Y" (when u=0)

    Returns
    -------
    new_beam : `ghost.beam_config`
        a new beam configuration corresponding to CCOB motor set as (u, b) angles
    """
    # copy input beam configuration
    new_beam = deepcopy(beam_config)
    # Convert ub angles to Euler angles
    rot = transform_rotation.from_euler('zxy', [u, 0., b], degrees=True)
    euler_angles = rot.as_euler('ZXY', degrees=True)
    new_beam['z_euler'] = euler_angles[0]
    new_beam['x_euler'] = euler_angles[1]
    new_beam['y_euler'] = euler_angles[2]
    return new_beam


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
    beams = []
    for i, shift in enumerate(shifts_list):
        beam_config = deepcopy(base_beam_config)
        beam_config['beam_id'] = base_id + i
        beam_config[f'{axis}_offset'] = shift
        beams.append(beam_config)
    return beams


def build_rotation_set(base_beam_config, axis, angles_list, base_id=0):
    """ Build a set of beam configurations for the given list of rotations
    starting from the given beam configuration

    Parameters
    ----------
    base_beam_config : `dict`
        the base beam configuration dictionary to start from
    axis : `string`
        axis around which to rotate as Euler rotations: "x_euler" or "y_euler"
    angles_list : `list` of `float`
        the list of angles to scan in degrees
    base_id : `int`
        the id of the first beam configuration created, following ids will be `id+1`

    Returns
    -------
     beams : `list` of `geom_config`
        a list of geometry configuration dictionaries
    """
    beams = []
    for i, angle in enumerate(angles_list):
        beam_config = deepcopy(base_beam_config)
        beam_config['beam_id'] = base_id + i
        beam_config[f'{axis}_euler'] = angle
        beams.append(beam_config)
    return beams


def build_square_set(distances, base_id=0):
    """ Build a square set of beams for a given list of distances

    Parameters
    ----------
    distances : `list` of `float`
        list of distances from -x/-y to +x/+Y
    base_id : `int`
        the id of the first beam configuration created, following ids will be `id+1`

    Returns
    -------
     beams : `list` of `geom_config`
        a list of beam configuration dictionaries
    """
    beams = []
    start_config = deepcopy(BEAM_CONFIG_0)
    start_config['n_photons'] = 100

    bid = base_id
    for x in distances:
        for y in distances:
            new_config = deepcopy(start_config)
            new_config['beam_id'] = bid
            new_config['x_offset'] = x
            new_config['y_offset'] = y
            beams.append(new_config)
            bid = bid + 1

    return beams


def build_first_quadrant_square_set(delta=0.02, d_max=0.26, base_id=0):
    """ Build a set of beams to cover the first quadrant with a square grid

    Parameters
    ----------
    delta : `float`
        distance between 2 points in x or y, usually 2 cm = 0.02 m
    d_max : `float`
        maximum distance to go from center, up to ~0.26 m is fine, then beam does not converge on camera
    base_id : `int`
        the id of the first beam configuration created, following ids will be `id+1`

    Returns
    -------
     beams : `list` of `geom_config`
        a list of beam configuration dictionaries
    """
    # that fixes the number of points
    shifts_list = list(np.arange(0, d_max, delta))
    return build_square_set(shifts_list, base_id)


def build_full_frame_square_set(delta=0.02, d_max=0.26, base_id=0):
    """ Build a full square set of beams on the full camera frame

    Parameters
    ----------
    delta : `float`
        distance between 2 points in x or y, usually 2 cm = 0.02 m
    d_max : `float`
        maximum distance to go from center, up to ~0.26 m is fine, then beam does not converge on camera
    base_id : `int`
        the id of the first beam configuration created, following ids will be `id+1`

    Returns
    -------
     beams : `list` of `geom_config`
        a list of beam configuration dictionaries
    """
    # that fixes the number of points
    shifts_list = list(np.arange(-d_max, d_max+delta/2., delta))
    return build_square_set(shifts_list, base_id)


def build_square_set_on_target(base_beam_config, distances,
                               target_x=0, target_y=0, base_id=0):
    """ Build a set of beams for the given list of distances from target

    Parameters
    ----------
    base_beam_config : `dict`
        the base beam configuration dictionary to start from
    distances : `list` of `float`
        list of distances from -x/-y to +x/+Y
    target_x : `float`
        target position on the X-axis of the camera plane
    target_y : `float`
        target position on the Y-axis of the camera plane
    base_id : `int`
        the id of the first beam configuration created, following ids will be `id+1`

    Returns
    -------
     beams : `list` of `geom_config`
        a list of beam configuration dictionaries
    """
    # starting with central beam
    new_beams = []
    start_config = deepcopy(base_beam_config)
    start_config['beam_id'] = base_id
    # then build other configs
    i = base_id
    for dx in distances:
        for dy in distances:
            beam_config = deepcopy(start_config)
            beam_config['beam_id'] = i
            pos_x = target_x + dx
            pos_y = target_y + dy
            new_config = point_beam_to_target(beam_config, pos_x, pos_y)
            new_beams.append(new_config)
            i = i + 1

    return new_beams


def build_polar_set(distances, angles, base_id=0):
    """ Build a set of beams for the given list of distances to center and a list of angles

    Parameters
    ----------
    distances : `list` of `float`
        list of distances to center to sample lenses
    angles : `list` of `float`
        list of polar angles to sample lenses
    base_id : `int`
        the id of the first beam configuration created, following ids will be `id+1`

    Returns
    -------
     beams : `list` of `geom_config`
        a list of beam configuration dictionaries
    """
    # starting with central beam
    hex_beams = []
    start_config = deepcopy(BEAM_CONFIG_0)
    start_config['n_photons'] = 100
    start_config['base_id'] = base_id
    hex_beams.extend([start_config])
    # then build other configs
    i = base_id + 1
    for dist in distances[1:]:
        for theta in angles:
            beam_config = deepcopy(start_config)
            beam_config['beam_id'] = i
            beam_config['x_offset'] = dist * cos(radians(theta))
            beam_config['y_offset'] = dist * sin(radians(theta))
            hex_beams.append(beam_config)
            i = i + 1

    return hex_beams


def build_first_quadrant_polar_set(delta=0.02, d_max=0.36, base_id=0):
    """ Build a set of beams for the given list of distances, and rotate on the first quadrant

    Parameters
    ----------
    delta : `float`
        distance between 2 points in x or y, usually 2 cm = 0.02 m
    d_max : `float`
        maximum distance to go from center, up to ~0.26 m is fine, then beam does not converge on camera
    base_id : `int`
        the id of the first beam configuration created, following ids will be `id+1`

    Returns
    -------
     beams : `list` of `geom_config`
        a list of beam configuration dictionaries
    """
    distances = list(np.arange(0, d_max, delta))
    thetas = list(np.arange(0, 105, 15))
    return build_polar_set(distances, thetas, base_id=base_id)


def build_full_frame_polar_set(base_id=0, set_size='large'):
    """ Build a set of beams for the given list of translations

    Parameters
    ----------
    base_id : `int`
        the id of the first beam configuration created, following ids will be `id+1`
    set_size : 'string'
        small, medium, large are the 3 default configurations
    Returns
    -------
     beams : `list` of `geom_config`
        a list of beam configuration dictionaries
    """
    if set_size == 'test':
        distances = list(np.arange(0, 0.36, 0.18))
        thetas = list(np.arange(0, 375, 120))
    elif set_size == 'small':
        distances = list(np.arange(0, 0.36, 0.06))
        thetas = list(np.arange(0, 375, 60))
    elif set_size == 'medium':
        distances = list(np.arange(0, 0.36, 0.04))
        thetas = list(np.arange(0, 375, 30))
    else:
        distances = list(np.arange(0, 0.36, 0.02))
        thetas = list(np.arange(0, 375, 15))
    return build_polar_set(distances, thetas, base_id=base_id)


def set_n_photons_on_beam_set(beam_set, n_photons=1000):
    """ Build a beam configuration matching the reference one, but with fewer photons so that the simulations
    called by the fit are faster

    Parameters
    ----------
    beam_set : `list` of `dict`
        the list of beam configurations to adjust
    n_photons : `int`
        the number of rays to simulate for the new configuration

    Returns
    -------
    fix_beam_set : `list` of `dict`
        the list of beam configuration with the adjusted number of photons
    """
    fix_beam_set = []
    for beam in beam_set:
        fit_beam = deepcopy(beam)
        fit_beam['n_photons'] = n_photons
        fix_beam_set.append(fit_beam)
    return fix_beam_set
