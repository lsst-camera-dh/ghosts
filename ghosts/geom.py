"""Geom module

This module provides tools to manipulate telescope geometries, i.e. shifts and rotations

Todo:
    *

"""
import pandas as pd


def to_panda(geom_config):
    """ Convert a geometry configuration dictionary to a panda data frame

    Parameters
    ----------
    geom_config : `dict`
        a dictionary with shifts and rotations for each optical element

    Returns
    -------
    data_frame : `pandas.DataFrame`
        a `pandas` data frame with shifts and rotations information
    """
    data_frame = pd.DataFrame(data=geom_config)
    return data_frame


def to_dict(geom_frame):
    """ Convert a geometry panda data frame to a dictionary of use with `tweak_optics`

    Parameters
    ----------
    geom_frame : `pandas.DataFrame`
        a `pandas` data frame with shifts and rotations information

    Returns
    -------
    geom_config : `dict`
        a dictionary with shifts and rotations for each optical element
    """
    geom_config = geom_frame.to_dict()
    if 'shifts' in geom_config['geom_id'].keys():
        geom_config['geom_id'] = geom_config['geom_id']['shifts']
    else:
        geom_config['geom_id'] = geom_config['geom_id']['rotations']
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
    frames = list()
    for one in geom_dict_list:
        frames.append(to_panda(one))
    geom_concat = concat_frames(frames)
    return geom_concat
