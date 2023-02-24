"""geom module

This module provides tools to manipulate telescope geometries, i.e. shifts and rotations
"""

import copy
import pandas as pd
import numpy as np
from ghosts.geom_configs import GEOM_CONFIG_0


def get_optics_translation(optics, geom_config):
    """ Return a vector with the translations of the given optics

    Parameters
    ----------
    optics : `string`
        the name of an optical element in L1, L2, L3, Filter, Detector
    geom_config : `dict`
        a dictionary with shifts and rotations for each optical element

    Returns
    -------
    translation_vector : `list` of `float`
        a vector of translations for the optical element
    """
    translation_vector = [geom_config.get(f'{optics}_d{axis}', 0.) for axis in ['x', 'y', 'z']]
    return translation_vector


def get_optics_rotation(optics, geom_config):
    """ Return a vector with the rotations of the given optics

    Parameters
    ----------
    optics : `string`
        the name of an optical element in L1, L2, L3, Filter, Detector
    geom_config : `dict`
        a dictionary with shifts and rotations for each optical element

    Returns
    -------
    rotation_vector : `list` of `float`
        a vector of rotations for the optical element
    """

    return [geom_config.get(f'{optics}_r{axis}', 0.) for axis in ['x', 'y', 'z']]


def to_panda(geom_config):
    """ Convert a geometry configuration dictionary to a panda data frame

    Indexing is done using the beam configuration `geom_id`.

    Parameters
    ----------
    geom_config : `dict`
        a dictionary with shifts and rotations for each optical element

    Returns
    -------
    data_frame : `pandas.DataFrame`
        a `pandas` data frame with shifts and rotations information
    """
    data_frame = pd.DataFrame(data=geom_config, index=[geom_config['geom_id']])
    return data_frame


def to_dict(geom_frame):
    """ Convert a geometry panda data frame to a dictionary of use with `tweak_optics`

    The geom data frame is expected to have only one geometry configuration.

    Parameters
    ----------
    geom_frame : `pandas.DataFrame`
        a `pandas` data frame with shifts and rotations information

    Returns
    -------
    geom_config : `dict`
        a dictionary with shifts and rotations for each optical element
    """
    geom_id = geom_frame['geom_id'].to_list()[0]
    geom_config = geom_frame.to_dict('index')[geom_id]
    return geom_config


def concat_frames(geom_frame_list):
    """ Concatenates geometry configuration data frames within one table

     Parameters
     ----------
     geom_frame_list : `list` of `pandas.DataFrame`
         a list of geometry configuration data frames

     Returns
     -------
     geom_concat : `pandas.DataFrame`
        a `pandas` data frame with several configurations of shifts and rotations information
     """
    tmp_concat = pd.concat(geom_frame_list)
    geom_concat = tmp_concat.fillna(0)
    geom_concat.sort_values('geom_id')
    return geom_concat


def concat_dicts(geom_dict_list):
    """ Concatenates geometry configuration dictionaries into a data frame

     Parameters
     ----------
     geom_dict_list : `list` of `dict`
         a list of geometry configuration dictionaries

     Returns
     -------
     geom_concat : `pandas.DataFrame`
        a `pandas` data frame with several configurations of shifts and rotations information
     """
    frames = []
    for one in geom_dict_list:
        frames.append(to_panda(one))
    geom_concat = concat_frames(frames)
    return geom_concat


# Helpers to create a set of geometries translations
def translate_optic(optic_name, axis, distance, geom_id=1000000):
    """ Create a dictionary to translate a piece of optic along an axis

    Parameters
    ----------
    optic_name : `string`
        the name of an optical element
    axis : `string`
        the name of the translation axis, in [x, y , z]
    distance : `float`
        the value of the shift in meters
    geom_id : `int`
        the id of the new geometry configuration

    Returns
    -------
    geom : `dict`
        a `geom_config` dictionary for the application of the translation
     """
    if axis not in ['x', 'y', 'z']:
        print(f'Unknown axis {axis}, doing nothing.')
        return None
    geom = copy.deepcopy(GEOM_CONFIG_0)
    geom['geom_id'] = geom_id
    opt_key = f'{optic_name}_d{axis}'
    geom[opt_key] = distance
    return geom


def rotate_optic(optic_name, axis, angle, geom_id=1000000):
    """ Rotate one optical element of a telescope given a list of Euler angles

    Parameters
    ----------
    optic_name : `string`
        the name of an optical element
    axis : `string`
        the name of the rotation axis, usually y
    angle : `float`
        the values of angle in degrees
    geom_id : `int`
        the id of the new geometry configuration

    Returns
    -------
     geom : `dict`
        a `geom_config` dictionary for the application of the rotation
    """
    geom = copy.deepcopy(GEOM_CONFIG_0)
    geom['geom_id'] = geom_id
    opt_key = f'{optic_name}_r{axis}'
    geom[opt_key] = angle
    return geom


def build_translation_set(optic_name, axis, shifts_list, base_id=0):
    """ Build a set of geometries for the given list of translations

    Parameters
    ----------
    optic_name : `string`
        the name of an optical element
    axis : `string`
        the name of the rotation axis, usually y
    shifts_list : `list` of `float`
        the list of distances to scan in meters
    base_id : `int`
        the id of the first geometry configuration created, following ids will be `id+1`

    Returns
    -------
     geoms : `list` of `geom_config`
        a list of geometry configuration dictionaries
    """
    geoms = []
    for i, shift in enumerate(shifts_list):
        geoms.append(translate_optic(optic_name, axis, shift, geom_id=base_id+i))
    return geoms


def build_rotation_set(optic_name, axis, angles_list, base_id=0):
    """ Build a set of geometries for the given list of rotations

    Parameters
    ----------
    optic_name : `string`
        the name of an optical element
    axis : `string`
        the name of the rotation axis, usually y
    angles_list : `list` of `float`
        the list of angles to scan in degrees
    base_id : `int`
        the id of the first geometry configuration created, following ids will be `id+1`

    Returns
    -------
     geoms : `list` of `geom_config`
        a list of geometry configuration dictionaries
    """
    geoms = []
    for i, angle in enumerate(angles_list):
        geoms.append(rotate_optic(optic_name, axis, angle, geom_id=base_id+i))
    return geoms


def build_random_geom(max_angle=0.1, max_shift=0.001):
    """ Build a random geometry from a base geometry configuration

    Parameters
    ----------
    max_angle : `float`
        the maximum value of the rotation angle in degree
    max_shift : `floats`
        the maximum value of the shift in meters

    Returns
    -------
     rnd_geom : `geom.geom_configs`
        a random geometry
    """
    # generate 30 random numbers
    numbers = np.random.random([30])
    rnd_geom_dict = copy.deepcopy(GEOM_CONFIG_0)
    optics_keys = list(rnd_geom_dict.keys())
    optics_keys.remove('geom_id')
    for optic, rnd in zip(optics_keys, numbers):
        mv_type = optic.split('_')[1]
        if mv_type in ['dx', 'dy', 'dz']:
            rnd_shift = max_shift * 2 * (rnd - 0.5)
            rnd_geom_dict[optic] = np.round(rnd_shift, 6)
        elif mv_type in ['rx', 'ry', 'rz']:
            rnd_euler_angle = max_angle * 2 * (rnd - 0.5)
            rnd_geom_dict[optic] = np.round(rnd_euler_angle, 6)
    # assign a random id
    rnd_geom_dict['geom_id'] = np.random.randint(1e9)
    return rnd_geom_dict
