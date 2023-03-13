"""Tools moduls

This module provides a number of generic tools.

"""

import os
import numpy as np


def get_default_yaml_path():
    """ Hack to find where the default yaml is, to be fixed properly
    """
    first_path = '../data/LSST_CCOB_r.yaml'
    if os.path.exists(first_path):
        return first_path
    second_path = './data/LSST_CCOB_r.yaml'
    if os.path.exists(second_path):
        return second_path
    return None


def get_vector(axis, value):
    """ Returns a vector containing the given value at the right spot for the given axis

    Parameters
    ----------
    axis : `string`
        the name of the rotation axis, x, y or z
    value : `float`
        the value of the rotation angle in degrees or of the shift in meters

    Returns
    -------
    vector : `list` of `floats`
        a list containing value at the corresponding spot of the given axis
    """
    if axis == 'x':
        vector = [value, 0, 0]
    elif axis == 'y':
        vector = [0, value, 0]
    elif axis == 'z':
        vector = [0, 0, value]
    else:
        raise Exception(f'Unknown axis {axis}: axis should be x, y or z')
    return vector


# define function to get nice ranges
def get_ranges(x, y, dr=0.010):
    """ Get x and y ranges around their mean values with a delta of dr

    This is useful to center plots around a region of interest

    Parameters
    ----------
    x : `numpy.array`
        The input array along the x-axis
    y : `numpy.array`
        The input array along the y-axis
    dr : `float`
        The delta around the mean value, or the box size if you wish

    Returns
    -------
    x_min, x_max, y_min, y_max : a tuple of 4 `floats`
        The min and max values for x- and y-axis, i.e. the box boundaries.
    """
    x_min = x.mean() - dr
    x_max = x.mean() + dr
    y_min = y.mean() - dr
    y_max = y.mean() + dr
    return x_min, x_max, y_min, y_max


def get_main_impact_point(r_forward):
    """ Return main image light rays

    Direct path will be r_forward with fewest number of things in "path"

    .. todo::
        `get_main_impact_point` should compute a real barycentre?

    Parameters
    ----------
    r_forward : `list` of `batoid.RayVector`
        a list of batoid RayVector with a bunch of rays propagated through the system.

    Returns
    -------
    i_straight : `int`
        the index of the main image
    direct_x, direct_y : `float`, `float`
        the x and y coordinate of the center of the main image
    direct_f : `float`
        the main image flux, relative to 1
    """
    # the main image corresponds to the "shortest" path
    i_straight = np.argmin([len(rrr.path) for rrr in r_forward])
    direct_x = np.mean(r_forward[i_straight].x)
    direct_y = np.mean(r_forward[i_straight].y)
    direct_f = r_forward[i_straight].flux[0]  # these are all equal
    return i_straight, direct_x, direct_y, direct_f


def unpack_geom_params(geom_params, geom_labels):
    """ Convert a list of geometry parameters into a dictionary as a telescope geometry configuration

    Parameters
    ----------
    geom_params : `list`
        an ordered list of parameters corresponding to a geometry configuration
    geom_labels : `list`
        a list of the geometry parameter labels (names) matching the list above

    Returns
    -------
    fitted_geom_config : `dict`
        a dictionary with the geometry of the telescope to fit
    """
    fitted_geom_config = {}
    for i, lab in enumerate(geom_labels):
        fitted_geom_config[lab] = geom_params[i]
    return fitted_geom_config
