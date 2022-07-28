"""analysis module

This module provides functions to analyze ghosts spots on the full focal plane, like getting ghosts positions
and features, computing separations between ghosts spots, associating ghosts spots and computing distances
between to sets of ghosts spots.

"""

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from ghosts.tools import get_ranges
from ghosts.beam import get_n_phot_for_power_nw_wl_nm
from ghosts.constants import LSST_CAMERA_PIXEL_DENSITY_MM2, LSST_CAMERA_PIXEL_QE


def get_full_light_on_camera(r_forward):
    """ Convert list of forward batoid ray vectors to list of impact points on camera

    Parameters
    ----------
    r_forward : `list` of `batoid.RayVector`
        a list of batoid RayVector with a bunch of rays propagated through the system.

    Returns
    -------
    all_x : `list`
        list of x coordinates of all light rays impact points on the detector
    all_y : `list`
        list of y coordinates of all light rays impact points on the detector
    all_f : `list`
        list of fluxes coordinates of all light rays impact points on the detector
    """
    # Plot light on detector on the right
    all_x = r_forward[0].x.tolist()
    all_y = r_forward[0].y.tolist()
    all_f = r_forward[0].flux.tolist()
    for rr in r_forward[1:]:
        all_x = all_x + rr.x.tolist()
        all_y = all_y + rr.y.tolist()
        all_f = all_f + rr.flux.tolist()
    return all_x, all_y, all_f


def get_ghost_name(ghost, debug=False):
    """ Just the name of a ghost spot as a tuple containing the
    names of the surface on which the first and second light reflection occured.

    Parameters
    ----------
    ghost : `batoid.RayVector`
        a batoid RayVector with a bunch of rays propagated through the system.
    debug : `bool`
        debug mode or not

    Returns
    -------
    ghost_name_tuple : `tuple` of `strings`
        name of the ghost as a tuple of strings, e.g. ('Filter_entrance', 'L2_exit') or ('L2_entrance', 'L1_entrance')
    """
    ghost_name_tuple = ('main', 'main')
    for i, opt in enumerate(ghost.path):
        if debug:
            print(ghost.path[i - 2], opt)
        if i >= 2 and ghost.path[i - 2] == opt:
            if debug:
                print(f'{i} bounce on {ghost.path[i - 1]} between {ghost.path[i - 2]} and {opt}')
            if ghost_name_tuple == ('main', 'main'):
                ghost_name_tuple = (ghost.path[i - 1],)
            else:
                ghost_name_tuple = (ghost_name_tuple[0], ghost.path[i - 1])
    return ghost_name_tuple


def get_ghost_stats(ghost):
    """ Compute some basic stats for a simulated ghost spot image

    .. todo::
        `get_ghost_stats` is only working on simulated images for now

    Parameters
    ----------
    ghost : `batoid.RayVector`
        A batoid RayVector with a bunch of rays propagated through the system.

    Returns
    -------
    mean_x, std_x, mean_y, std_y : `floats`
        spot position with uncertainty as standard deviation in x and y
    x_width, y_width : `floats`
        spot width in x and y
    radius, radius_err : `float`
        beam spot radius from x and y widths, and an estimator of the uncertainty
    weights_sum : `float`
        total flux as computed by batoid `flux.sum()`
    mean_intensity : `float`
        average flux of all rays
    spot_surface_mm2 : `float`
        spot surface, using `x_width` as the diameter
    density_phot_mm2 : `float`
        density of photons per :math:`mm^2`, for one simulated photon, as the `mean_intensity`/`spot_surface_mm2`
    """
    mean_x = ghost.x.mean()
    std_x = ghost.x.std()
    mean_y = ghost.y.mean()
    std_y = ghost.x.std()
    x_width = ghost.x.max() - ghost.x.min()
    y_width = ghost.y.max() - ghost.y.min()
    radius = (x_width + y_width) / 2.  # simple mean
    radius_err = math.fabs(x_width - y_width) / 2.
    weights_sum = ghost.flux.sum()
    mean_intensity = weights_sum / len(ghost.x)
    spot_surface_mm2 = math.pi * (radius * 1000.) * (radius * 1000.)
    density_phot_mm2 = mean_intensity / spot_surface_mm2

    return mean_x, std_x, mean_y, std_y, x_width, y_width, radius, radius_err, \
        weights_sum, mean_intensity, spot_surface_mm2, density_phot_mm2


def get_ghost_spot_data(i, ghost, p=100, wl=500):
    """ Get some basic information for a simulated ghost spot image

    .. todo::
        `get_ghost_spot_data` should be made wavelength dependent

    Parameters
    ----------
    i : `int`
        the ghost index, useful really only on simulated data
    ghost : `batoid.RayVector`
        a batoid RayVector with a bunch of rays propagated through the system.
    p : `float`
        beam power in nW to compute the photon density
    wl : `int`
        the beam wavelength

    Returns
    -------
    ghost_spot_data : `dict`
        a dictionnary with ghost spot information : index, name, pos_x, width_x, pos_x, width_y, surface,
        pixel_signal and photon_density
    """
    # identify ghost
    ghost_name = get_ghost_name(ghost)
    # normalized stats
    mean_x, std_x, mean_y, std_y, x_width, y_width, radius, radius_err, \
        weights_sum, mean_intensity, spot_surface_mm2, density_phot_mm2 = get_ghost_stats(ghost)
    # number of photons for 100 nW at 500 nm
    n_phot_total = get_n_phot_for_power_nw_wl_nm(p, wl)
    n_e_pixel = density_phot_mm2 / LSST_CAMERA_PIXEL_DENSITY_MM2 * n_phot_total * LSST_CAMERA_PIXEL_QE

    ghost_spot_data = {'index': i, 'name': ghost_name,
                       'pos_x': mean_x, 'std_x': std_x, 'width_x': x_width,
                       'pos_y': mean_y, 'std_y': std_y, 'width_y': y_width,
                       'radius': radius, 'radius_err': radius_err,
                       'flux': weights_sum,
                       'surface': spot_surface_mm2, 'pixel_signal': n_e_pixel,
                       'photon_density': density_phot_mm2}

    return ghost_spot_data


def map_ghost(ghost, ax, n_bins=100, dr=0.01):
    """ Builds a binned image, as a `matplotlib.hexbin` of a ghost

    Parameters
    ----------
    ghost : `batoid.RayVector`
        a batoid RayVector with a bunch of rays propagated through the system.
    ax : `matplotlib.axis`
        an axis object to draw the histogram
    n_bins : `int`
        the number of bins of the histogram, that is also the number of "pixel" of the "image"
    dr : `float`
        the extra space around the ghost spot image, to get nice axis ranges
    Returns
    -------
    ghost_map : `matplotlib.axis.hexbin`
        a binned image (2D histogram) of the ghost on the detector plane
    """
    # bin data
    ghost_map = ax.hexbin(ghost.x, ghost.y, C=ghost.flux, reduce_C_function=np.sum,
                          gridsize=n_bins, extent=get_ranges(ghost.x, ghost.y, dr))

    return ghost_map


def reduce_ghosts(r_forward):
    """ Builds a binned image, as a `matplotlib.hexbin` of a ghost

    Parameters
    ----------
    r_forward : `batoid.RayVector`
        a batoid RayVector with a bunch of rays propagated through the system.

    Returns
    -------
    spots_data : `list` of `dict`
        a list of dictionaries of ghost spot data (position, radius, brightness)
    ghost_maps : `list` of `matplotlib.axis.hexbin`
        a list of images of ghosts as 2D histograms
    """
    # store some stats roughly
    spots_data = []
    ghost_maps = []
    _fig, ax = plt.subplots(len(r_forward))
    axs = ax.ravel()
    for i, ghost in enumerate(r_forward):
        # bin data (and make plot)
        hb_map = map_ghost(ghost, axs[i])
        ghost_spot_data = get_ghost_spot_data(i, ghost)
        ghost_maps.append(hb_map)
        spots_data.append(ghost_spot_data)
    plt.close(_fig)
    return spots_data, ghost_maps


def make_data_frame(spots_data, beam_id=0, geom_id=0):
    """ Create a pandas data frame from the ghost spots data dictionary
    and a beam configuration.

    Parameters
    ----------
    spots_data : `dict`
        a dictionary with ghost spots data
    beam_id : `int`
        a beam configuration id
    geom_id : `int`
        a geometry configuration id

    Returns
    -------
    data_frame : `pandas.DataFrame`
        a panda data frame with ghost spot data information, including beam and geometry configuration ids
    """
    # creating a nice pandas data frame
    data_frame = pd.DataFrame(
        {
            "beam_id": beam_id,
            "geom_id": geom_id,
            "index": np.array([data['index'] for data in spots_data], dtype="int"),
            "name": [data['name'] for data in spots_data],
            "pos_x": np.array([data['pos_x'] for data in spots_data], dtype="float"),
            "std_x": np.array([data['std_x'] for data in spots_data], dtype="float"),
            "pos_y": np.array([data['pos_y'] for data in spots_data], dtype="float"),
            "std_y": np.array([data['std_y'] for data in spots_data], dtype="float"),
            "width_x": np.array([data['width_x'] for data in spots_data], dtype="float"),
            "width_y": np.array([data['width_y'] for data in spots_data], dtype="float"),
            "radius": np.array([data['radius'] for data in spots_data], dtype="float"),
            "radius_err": np.array([data['radius_err'] for data in spots_data], dtype="float"),
            "flux": np.array([data['flux'] for data in spots_data], dtype="float"),
            "surface": np.array([data['surface'] for data in spots_data], dtype="float"),
            "pixel_signal": np.array([data['pixel_signal'] for data in spots_data], dtype="float"),
        }
    )
    return data_frame


def compute_ghost_separations(data_frame):
    """ Compute ghosts images separations and various ratios from a ghosts spot data frame

    .. todo::
        `compute_ghost_separations` remove main image from the distance since it's saturated in data.

    Parameters
    ----------
    data_frame : `pandas.DataFrame`
        a ghost spots data frame
    Returns
    -------
    data_frame : `pandas.DataFrame`
        a pandas data frame with information on ghost spots data separations and ratios
    """
    # computing distances ghost to ghost, and ghosts overlap
    dist_data = []
    n = len(data_frame['pos_x']) - 1
    for i in range(n):
        for k in range(1, n - i):
            # distance center to center
            distance = math.dist([data_frame['pos_x'][i], data_frame['pos_y'][i]],
                                 [data_frame['pos_x'][i + k], data_frame['pos_y'][i + k]])
            # distance border to border, assuming round spot - overlap
            r1 = data_frame['radius'][i]
            r2 = data_frame['radius'][i + k]
            overlap = distance - (r1 + r2)
            # surface and pixel signal ratio
            flux_ratio = data_frame['flux'][i + k] / data_frame['flux'][i]
            surface_ratio = data_frame['surface'][i + k] / data_frame['surface'][i]
            signal_ratio = data_frame['pixel_signal'][i + k] / data_frame['pixel_signal'][i]
            # ghost names
            name_1 = data_frame['name'][i]
            name_2 = data_frame['name'][i + k]
            # add data container
            dist_data.append([i, i + k, name_1, name_2, distance, overlap, flux_ratio, surface_ratio, signal_ratio])

    ghosts_separation = pd.DataFrame(
        {
            "ghost_1": np.array([data[0] for data in dist_data], dtype="int"),
            "ghost_2": np.array([data[1] for data in dist_data], dtype="int"),
            "name_1": [data[2] for data in dist_data],
            "name_2": [data[3] for data in dist_data],
            "distance": np.array([data[4] for data in dist_data], dtype="float"),
            "overlap": np.array([data[5] for data in dist_data], dtype="float"),
            "flux_ratio": np.array([data[6] for data in dist_data], dtype="float"),
            "surface_ratio": np.array([data[7] for data in dist_data], dtype="float"),
            "signal_ratio": np.array([data[8] for data in dist_data], dtype="float"),
        }
    )
    return ghosts_separation


def compute_distance_spot_to_spot(df_slice_1, df_slice_2, radius_scale_factor=100):
    """ Compute a 3D geometric distance between 2 ghosts spots centers, considering the spot radius
    as the 3rd dimension.

    Parameters
    ----------
    df_slice_1 : `pandas.DataFrame`
        a ghost spots data frame slice, with one line corresponding to one ghost
    df_slice_2 : `pandas.DataFrame`
        a ghost spots data frame slice, with one line corresponding to one ghost
    radius_scale_factor : `float`
        as the radius is considered a 3rd dimension, we scale it to the same range as the x and y axis, e.g. the spot
        radius is 2.5 mm to scale to the 60 cm of the focal plane ~ 100

    Returns
    -------
    dist_2d : `float`
        the distance between 2 spots for the 2D distance
    dist_2d_err : `float`
        the error on that distance from the std error on the position centers and radius  for the 2D distance
    dist_3d : `float`
        the distance between 2 spots for the 3D distance
    dist_3d_err : `float`
        the error on that distance from the std error on the position centers and radius  for the 3D distance
    """
    dist_2d = math.dist([df_slice_1['pos_x'], df_slice_1['pos_y']],
                        [df_slice_2['pos_x'], df_slice_2['pos_y']])
    d1_2d_sq = df_slice_1['std_x'] * df_slice_1['std_x'] + df_slice_1['std_y'] * df_slice_1['std_y']
    d2_2d_sq = df_slice_2['std_x'] * df_slice_2['std_x'] + df_slice_2['std_y'] * df_slice_2['std_y']
    dist_2d_err = math.sqrt(d1_2d_sq + d2_2d_sq)

    dist_3d = math.dist([df_slice_1['pos_x'], df_slice_1['pos_y'], df_slice_1['radius'] * radius_scale_factor],
                        [df_slice_2['pos_x'], df_slice_2['pos_y'], df_slice_2['radius'] * radius_scale_factor])
    d1_3d_sq = df_slice_1['std_x'] * df_slice_1['std_x'] + df_slice_1['std_y'] * df_slice_1['std_y'] \
        + df_slice_1['radius_err'] * df_slice_1['radius_err'] * radius_scale_factor * radius_scale_factor
    d2_3d_sq = df_slice_2['std_x'] * df_slice_2['std_x'] + df_slice_2['std_y'] * df_slice_2['std_y'] \
        + df_slice_2['radius_err'] * df_slice_2['radius_err'] * radius_scale_factor * radius_scale_factor
    dist_3d_err = math.sqrt(d1_3d_sq + d2_3d_sq)
    return dist_2d, dist_2d_err, dist_3d, dist_3d_err


def find_nearest_ghost(ghost_slice, ghosts_df, radius_scale_factor=100):
    """ Find the nearest ghost spot to a given ghost spot and report its distance with its error

    This is done using both the 2D and 3D distances.

    Parameters
    ----------
    ghost_slice : `pandas.DataFrame`
        a ghost spots data frame slice, with one line corresponding to one ghost
    ghosts_df : `pandas.DataFrame`
        a `pandas` data frame with information on ghost spots data separations and ratios
    radius_scale_factor : `float`
        a kind of weight for the spot radius to be used in the distance computation

    Returns
    -------
    index_of_min_2d : `int`
        the index in the data frame of the nearest ghost for the 2D distance
    min_distance_2d : `float`
        distance of the given ghost spot to the nearest ghost spot for the 2D distance
    min_distance_2d_err : `float`
        the uncertainty on the distance with the nearest ghost spot for the 2D distance
    index_of_min_3d : `int`
        the index in the data frame of the nearest ghost for the 3D distance
    min_distance_3d : `float`
        distance of the given ghost spot to the nearest ghost spot for the 3D distance
    min_distance_3d_err : `float`
        the uncertainty on the distance with the nearest ghost spot for the 3D distance
    """
    dist_2d_data = []
    dist_2d_err_data = []
    dist_3d_data = []
    dist_3d_err_data = []
    n = len(ghosts_df['pos_x'])
    for i in range(n):
        dist_2d, dist_2d_err, dist_3d, dist_3d_err = \
            compute_distance_spot_to_spot(ghost_slice, ghosts_df.xs(i), radius_scale_factor)
        dist_2d_data.append(dist_2d)
        dist_2d_err_data.append(dist_2d_err)
        dist_3d_data.append(dist_3d)
        dist_3d_err_data.append(dist_3d_err)

    dist_2d_array = np.array(dist_2d_data)
    index_of_min_2d = np.argmin(dist_2d_array)
    min_distance_2d = dist_2d_array[index_of_min_2d]
    min_distance_2d_err = dist_2d_err_data[index_of_min_2d]

    dist_3d_array = np.array(dist_3d_data)
    index_of_min_3d = np.argmin(dist_3d_array)
    min_distance_3d = dist_3d_array[index_of_min_3d]
    min_distance_3d_err = dist_3d_err_data[index_of_min_3d]

    return index_of_min_2d, min_distance_2d, min_distance_2d_err, \
        index_of_min_3d, min_distance_3d, min_distance_3d_err


def match_ghosts(ghosts_df_1, ghosts_df_2, radius_scale_factor=100):
    """ Match ghosts positions from two ghosts data frames

    Parameters
    ----------
    ghosts_df_1 : `pandas.DataFrame`
        a `pandas` data frame with information on ghost spots data separations and ratios
    ghosts_df_2 : `pandas.DataFrame`
        a `pandas` data frame with information on ghost spots data separations and ratios
    radius_scale_factor : `float`
        a kind of weight for the spot radius to be used in the distance computation

    Returns
    -------
    ghosts_match : `pandas.DataFrame`
        a `pandas` data frame with the indices of each ghost and nearest ghost, and the distance between the two
    """
    match_i1 = []
    match_i2_2d = []
    match_i2_3d = []
    match_min_dist_2d = []
    match_min_dist_3d = []
    match_min_dist_2d_err = []
    match_min_dist_3d_err = []

    n = len(ghosts_df_1['pos_x'])
    for i in range(n):
        index_of_min_2d, min_distance_2d, min_distance_2d_err, \
            index_of_min_3d, min_distance_3d, min_distance_3d_err = \
            find_nearest_ghost(ghosts_df_1.xs(i), ghosts_df_2, radius_scale_factor)
        match_i1.append(i)
        # 2D distance
        match_i2_2d.append(index_of_min_2d)
        match_min_dist_2d.append(min_distance_2d)
        match_min_dist_2d_err.append(min_distance_2d_err)
        # 3D distance
        match_i2_3d.append(index_of_min_3d)
        match_min_dist_3d.append(min_distance_3d)
        match_min_dist_3d_err.append(min_distance_3d_err)

    ghosts_match = pd.DataFrame(
        {
            "beam_id_1": ghosts_df_1['beam_id'],
            "geom_id_1": ghosts_df_1['geom_id'],
            "beam_id_2": ghosts_df_2['beam_id'],
            "geom_id_2": ghosts_df_2['geom_id'],
            "ghost_1": np.array(match_i1, dtype="int"),
            "ghost_2_2d": np.array(match_i2_2d, dtype="int"),
            "distance_2d": np.array(match_min_dist_2d, dtype="float"),
            "distance_2d_err": np.array(match_min_dist_2d_err, dtype="float"),
            "ghost_2_3d": np.array(match_i2_3d, dtype="int"),
            "distance_3d": np.array(match_min_dist_3d, dtype="float"),
            "distance_3d_err": np.array(match_min_dist_3d_err, dtype="float"),
        }
    )
    return ghosts_match


def compute_reduced_distance(ghosts_match):
    """ Compute a kind of reduced distance between two lists of ghosts

    .. math::
        L = \\frac{\\sqrt{\\sum_{i=1}^{n} \\frac{d(g_{s,i}, g_{r,k_i})^2}{\\sigma_d(g_{s,i}, g_{r,k_i})^2}}}{n}

    Parameters
    ----------
    ghosts_match : `pandas.DataFrame`
        a data frame with information about matching ghosts spots, see `match_ghosts`

    Returns
    -------
    reduced_distance : `float`
        a reduced distance computed as the average of the square root of the sum of squared input distances divided
        by the square of the errors on the distance.
    """
    n_matches = len(ghosts_match['distance_3d'])
    reduced_distance = math.sqrt(sum(ghosts_match['distance_3d'] * ghosts_match['distance_3d'] /
                                     ghosts_match['distance_3d_err'] * ghosts_match['distance_3d_err'])) / n_matches
    return reduced_distance


def compute_2d_reduced_distance(ghosts_match):
    """ Compute a simple 2D reduced distance between two lists of ghosts

    Parameters
    ----------
    ghosts_match : `pandas.DataFrame`
        a data frame with information about matching ghosts spots, see `match_ghosts`

    Returns
    -------
    reduced_distance : `float`
        a reduced distance computed as the average of the square root of the sum of squared input distances divided

    """
    n_matches = len(ghosts_match['distance_2d'])
    reduced_distance = math.sqrt(sum(ghosts_match['distance_2d'] * ghosts_match['distance_2d'] /
                                     ghosts_match['distance_2d_err'] * ghosts_match['distance_2d_err'])) / n_matches
    return reduced_distance
