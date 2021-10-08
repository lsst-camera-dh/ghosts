import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from ghosts.tools import get_ranges, get_main_impact_point
from ghosts.beam import get_n_phot_for_power_nw_wl_nm
from ghosts.constants import LSST_CAMERA_PIXEL_DENSITY_MM2, LSST_CAMERA_PIXEL_QE

def get_full_light_on_camera(rForward):
    ''' Convert rForward to list of impact points on camera
    '''
    # Plot light on detector on the right
    all_x = rForward[0].x.tolist()
    all_y = rForward[0].y.tolist()
    all_f = rForward[0].flux.tolist()
    for rr in rForward[1:]:
        all_x = all_x + rr.x.tolist()
        all_y = all_y + rr.y.tolist()
        all_f = all_f + rr.flux.tolist()
    return all_x, all_y, all_f


def get_ghost_name(ghost, debug=False):
    ghost_tuple = ('main', 'main')
    for i, opt in enumerate(ghost.path):
        if debug:
            print(ghost.path[i - 2], opt)
        if i >= 2 and ghost.path[i - 2] == opt:
            if debug:
                print(f'{i} bounce on {ghost.path[i - 1]} between {ghost.path[i - 2]} and {opt}')
            if ghost_tuple == ('main', 'main'):
                ghost_tuple = (ghost.path[i - 1],)
            else:
                ghost_tuple = (ghost_tuple[0], ghost.path[i - 1])
    return ghost_tuple


def get_ghost_stats(ghost):
    mean_x = ghost.x.mean()
    mean_y = ghost.y.mean()
    x_width = ghost.x.max() - ghost.x.min()
    y_width = ghost.y.max() - ghost.y.min()
    weights_sum = ghost.flux.sum()
    mean_intensity = weights_sum / len(ghost.x)
    spot_surface_mm2 = 3.14 * (x_width * 1000. / 2) * (x_width * 1000. / 2)
    density_phot_mm2 = mean_intensity / spot_surface_mm2
    return mean_x, mean_y, x_width, y_width, weights_sum, mean_intensity, spot_surface_mm2, density_phot_mm2


def get_ghost_spot_data(i, ghost, p=100, wl=500):
    # identify ghost
    ghost_name = get_ghost_name(ghost)

    # normalized stats
    mean_x, mean_y, x_width, y_width, weights_sum, mean_intensity, \
    spot_surface_mm2, density_phot_mm2 = get_ghost_stats(ghost)
    # for 100 nW at 500 nm
    n_phot_total = get_n_phot_for_power_nw_wl_nm(p, wl)
    n_e_pixel = density_phot_mm2 / LSST_CAMERA_PIXEL_DENSITY_MM2 * n_phot_total * LSST_CAMERA_PIXEL_QE

    ghost_spot_data = {'index': i, 'name': ghost_name,
                       'pos_x': mean_x, 'width_x': x_width,
                       'surface': spot_surface_mm2, 'pixel_signal': n_e_pixel,
                       'photon_density': density_phot_mm2}
    return ghost_spot_data


def map_ghost(ghost, ax, n_bins=100, dr=0.01, wl=500, p=100):
    # bin data
    hb = ax.hexbin(ghost.x, ghost.y, C=ghost.flux, reduce_C_function=np.sum,
                   gridsize=n_bins, extent=get_ranges(ghost.x, ghost.y, dr))
    return hb

def reduce_ghosts(rForward):
    # store some stats roughly
    spots_data = list()
    ghost_maps = list()
    _fig, ax = plt.subplots(len(rForward))
    axs = ax.ravel()
    for i, ghost in enumerate(rForward):
        # bin data (and make plot)
        hb_map = map_ghost(ghost, axs[i])
        ghost_spot_data = get_ghost_spot_data(i, ghost)
        ghost_maps.append(hb_map)
        spots_data.append(ghost_spot_data)
    plt.close(_fig)
    return spots_data, ghost_maps


def make_data_frame(rForward, spots_data):
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
            "width_x": np.array([data['width_x'] for data in spots_data], dtype="float"),
            "surface": np.array([data['surface'] for data in spots_data], dtype="float"),
            "pixel_signal": np.array([data['pixel_signal'] for data in spots_data], dtype="float"),
        }
    )
    return data_frame

def compute_ghost_separations(data_frame):
    # computing distances ghost to ghost, and ghosts overlap
    dist_data = list()
    n = len(data_frame['pos_x']) - 1
    for i in range(n):
        for k in range(1, n - i):
            # distance center to center
            # d = data_frame['pos_x'][i]-data_frame['pos_x'][i+k]
            d = math.fabs(data_frame['pos_x'][i] - data_frame['pos_x'][i + k])
            # distance border to border - overlap
            w1 = data_frame['width_x'][i]
            w2 = data_frame['width_x'][i + k]
            overlap = d - (w1 / 2. + w2 / 2.)
            # surface and pixel signal ratio
            surface_ratio = data_frame['surface'][i + k] / data_frame['surface'][i]
            signal_ratio = data_frame['pixel_signal'][i + k] / data_frame['pixel_signal'][i]
            # ghost names
            name_1 = data_frame['name'][i]
            name_2 = data_frame['name'][i + k]
            # add data container
            dist_data.append([i, i + k, name_1, name_2, d, overlap, surface_ratio, signal_ratio])
            # data.append([i, i+k, d, overlap, surface_ratio, signal_ratio])

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


