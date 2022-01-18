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
    r_forward : `batoid.RayVector`
        a batoid RayVector with a bunch of forward rays propagated  through the system.

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
        `get_ghost_stats` is likely not working for real image analysis

    Parameters
    ----------
    ghost : `batoid.RayVector`
        A batoid RayVector with a bunch of rays propagated through the system.

    Returns
    -------
    mean_x, mean_y, x_width, y_width : `floats`
        spot position and width in x and y
    radius : `float`
        beam spot radius from x and y widths
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
    mean_y = ghost.y.mean()
    x_width = ghost.x.max() - ghost.x.min()
    y_width = ghost.y.max() - ghost.y.min()
    radius = math.sqrt(x_width*x_width + y_width*y_width)
    weights_sum = ghost.flux.sum()
    mean_intensity = weights_sum / len(ghost.x)
    spot_surface_mm2 = math.pi * (x_width * 1000. / 2.) * (x_width * 1000. / 2.)
    density_phot_mm2 = mean_intensity / spot_surface_mm2

    return mean_x, mean_y, x_width, y_width, radius, weights_sum, mean_intensity, spot_surface_mm2, density_phot_mm2


def get_ghost_spot_data(i, ghost, p=100, wl=500):
    """ Get some basic information for a simulated ghost spot image

    .. todo::
        `get_ghost_spot_data` should be made wavelength dependent

    Parameters
    ----------
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
    mean_x, mean_y, x_width, y_width, radius, weights_sum, mean_intensity, spot_surface_mm2, density_phot_mm2 = \
        get_ghost_stats(ghost)
    # number of photons for 100 nW at 500 nm
    n_phot_total = get_n_phot_for_power_nw_wl_nm(p, wl)
    n_e_pixel = density_phot_mm2 / LSST_CAMERA_PIXEL_DENSITY_MM2 * n_phot_total * LSST_CAMERA_PIXEL_QE

    ghost_spot_data = {'index': i, 'name': ghost_name,
                       'pos_x': mean_x, 'width_x': x_width,
                       'pos_y': mean_y, 'width_y': y_width,
                       'radius' : radius,
                       'surface': spot_surface_mm2, 'pixel_signal': n_e_pixel,
                       'photon_density': density_phot_mm2}
    return ghost_spot_data


def map_ghost(ghost, ax, n_bins=100, dr=0.01):
    """ Builds a binned image, as a `matplotlib.hexbin` of a ghost

    Parameters
    ----------
    ghost : `batoid.RayVector`
        a batoid RayVector with a bunch of rays propagated through the system.

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
    spots_data = list()
    ghost_maps = list()
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


def make_data_frame(spots_data):
    """ Create a pandas data frame from the ghost spots data dictionary

        .. todo::
            beam config is hardcoded in make_data_frame

        Parameters
        ----------
        spots_data : `dict`
            a dictionary with ghost spots data
        Returns
        -------
        data_frame : `pandas.DataFrame`
            a pandas data frame with ghost spot data information, including beam configuration
    """
    # creating a nice pandas data frame
    data_frame = pd.DataFrame(
        {
            "config": 0,
            "n_photons": 1000,
            "beam_x": 0.1, "beam_y": 0.,
            "beam_theta": 0., "beam_phi": 0.,
            "index": np.array([data['index'] for data in spots_data], dtype="int"),
            "name": [data['name'] for data in spots_data],
            "pos_x": np.array([data['pos_x'] for data in spots_data], dtype="float"),
            "pos_y": np.array([data['pos_y'] for data in spots_data], dtype="float"),
            "width_x": np.array([data['width_x'] for data in spots_data], dtype="float"),
            "width_y": np.array([data['width_y'] for data in spots_data], dtype="float"),
            "radius": np.array([data['radius'] for data in spots_data], dtype="float"),
            "surface": np.array([data['surface'] for data in spots_data], dtype="float"),
            "pixel_signal": np.array([data['pixel_signal'] for data in spots_data], dtype="float"),
        }
    )
    return data_frame


def compute_ghost_separations(data_frame):
    """ Compute ghosts images separations and various ratios from a ghosts spot data frame

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
    dist_data = list()
    n = len(data_frame['pos_x']) - 1
    for i in range(n):
        for k in range(1, n - i):
            # distance center to center
            distance = math.dist([data_frame['pos_x'][i], data_frame['pos_y'][i]],
                                 [data_frame['pos_x'][i + k], data_frame['pos_y'][i + k]])
            # distance border to border, assuming round spot - overlap
            r1 = data_frame['radius'][i]
            r2 = data_frame['radius'][i + k]
            overlap = distance - (r1 / 2. + r2 / 2.)
            # surface and pixel signal ratio
            surface_ratio = data_frame['surface'][i + k] / data_frame['surface'][i]
            signal_ratio = data_frame['pixel_signal'][i + k] / data_frame['pixel_signal'][i]
            # ghost names
            name_1 = data_frame['name'][i]
            name_2 = data_frame['name'][i + k]
            # add data container
            dist_data.append([i, i + k, name_1, name_2, distance, overlap, surface_ratio, signal_ratio])

    ghosts_separation = pd.DataFrame(
        {
            "ghost_1": np.array([data[0] for data in dist_data], dtype="int"),
            "ghost_2": np.array([data[1] for data in dist_data], dtype="int"),
            "name_1": [data[2] for data in dist_data],
            "name_2": [data[3] for data in dist_data],
            "distance": np.array([data[4] for data in dist_data], dtype="float"),
            "overlap": np.array([data[5] for data in dist_data], dtype="float"),
            "surface_ratio": np.array([data[6] for data in dist_data], dtype="float"),
            "signal_ratio": np.array([data[7] for data in dist_data], dtype="float"),
        }
    )
    return ghosts_separation
