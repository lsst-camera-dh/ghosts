import numpy as np


# define function to get nice rangs
def get_ranges(x, y, dr=0.010):
    """ Get x and y ranges around their mean values with a delta of dr

    This is useful to center plots around a region of interest

    Parameters
    ----------
    x : `numpy.array`
        The input array along the x axis
    y : `numpy.array`
        The input array along the xy axis
    dr : `float`
        The delta around the mean value, or the box size if you wish

    Returns
    -------
    x_min, x_max, y_min, y_max : a tuple of 4 `floats`
        The min and max values for x and y axis, i.e. the box boundaries.
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
        `get_main_impact_point` should compute a real baricenter
    """
    i_straight = np.argmin([len(rrr.path) for rrr in r_forward])
    direct_x = np.mean(r_forward[i_straight].x)
    direct_y = np.mean(r_forward[i_straight].y)
    direct_f = r_forward[i_straight].flux[0]  # these are all equal
    return i_straight, direct_x, direct_y, direct_f
