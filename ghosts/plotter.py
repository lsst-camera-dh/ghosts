"""plotter module

This module provides functions to plot every single thing that the `ghosts` module produces.

"""
import batoid
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy import stats
# import ipyvolume as ipv
import numpy as np
from ghosts.tools import get_ranges, get_main_impact_point
from ghosts.analysis import get_full_light_on_camera, map_ghost, get_ghost_spot_data
from ghosts.constants import LSST_CAMERA_EXTENT


def plot_setup(telescope, simulation):
    """ Plots a standard CCOB optical setup, including wide and close views on the beam
    and the image obtained on the full focal plane through the simulation

    Parameters
    ----------
    telescope : `batoid.telescope`
        the optical setup as defined in `batoid`
    simulation : `list` of rays from the simulation
        a list with like this `[trace_full, forward_rays, rReverse, rays]`, `r_reverse` is not used.
        This list is provided by the output of :meth:`ghosts.simulator.run_simulation`

    Returns
    -------
    0 : 0
        0 if all is well
    """
    trace_full = simulation[0]
    forward_rays = simulation[1]
    # r_reverse = simulation[2]
    rays = simulation[3]

    # Create figure with 2 columns
    fig1 = plt.figure(figsize=(12, 11), constrained_layout=True)
    gs = fig1.add_gridspec(3, 2)
    f1_ax1 = fig1.add_subplot(gs[:, 0])
    f1_ax2 = fig1.add_subplot(gs[0, 1])
    f1_ax3 = fig1.add_subplot(gs[1, 1])
    f1_ax4 = fig1.add_subplot(gs[2, 1])

    # Draw camera on the left
    telescope.draw2d(f1_ax1, c='k')
    # now draw ray tracing on top of telescope
    batoid.drawTrace2d(f1_ax1, trace_full, c='orange')
    # set axis titles and draw small referential
    f1_ax1.set_xlabel('x (m)', fontsize=20)
    f1_ax1.set_ylabel('z (m)', fontsize=20)

    # Plot input beam spot full scale
    beam_spot = rays.positionAtTime(3.397)
    hb1 = f1_ax2.hexbin(beam_spot[:, 0], beam_spot[:, 1], reduce_C_function=np.sum,
                        extent=LSST_CAMERA_EXTENT, gridsize=(150, 150))
    f1_ax2.set_aspect('equal')
    f1_ax2.set_title('Beam spot')
    f1_ax2.set_xlabel('x (m)', fontsize=16)
    f1_ax2.set_ylabel('y (m)', fontsize=16)
    fig1.colorbar(hb1, ax=f1_ax2)

    # Plot input beam spot zoom in
    hb2 = f1_ax3.hexbin(beam_spot[:, 0], beam_spot[:, 1], reduce_C_function=np.sum,
                        extent=get_ranges(np.array(beam_spot[:, 0]), np.array(beam_spot[:, 1]), 0.01),
                        gridsize=(50, 50))
    f1_ax3.set_aspect("equal")
    f1_ax3.set_title('Beam spot zoom')
    f1_ax3.set_xlabel('x (m)', fontsize=16)
    f1_ax3.set_ylabel('y (m)', fontsize=16)
    fig1.colorbar(hb2, ax=f1_ax3)

    # Plot light on detector on the right
    all_x, all_y, all_f = get_full_light_on_camera(forward_rays)
    hb3 = f1_ax4.hexbin(all_x, all_y, C=all_f, reduce_C_function=np.sum,
                        extent=LSST_CAMERA_EXTENT, gridsize=(150, 150))

    # Plot approximate focal plane radius
    th = np.linspace(0, 2 * np.pi, 1000)
    plt.plot(0.32 * np.cos(th), 0.32 * np.sin(th), c='r')

    # Plot direct path location on focal plane
    i_straight, direct_x, direct_y, direct_f = get_main_impact_point(forward_rays)
    plt.text(direct_x, direct_y, '+', horizontalalignment='center',
             verticalalignment='center', color='m')
    f1_ax4.set_aspect("equal")
    f1_ax4.set_title('Image with ghosts')
    f1_ax4.set_xlabel('x (m)', fontsize=16)
    f1_ax4.set_ylabel('y (m)', fontsize=16)
    fig1.colorbar(hb3, ax=f1_ax4)

    # add info on direct path
    print(f'Direct path is number {i_straight}')
    print(f'  central impact point is ({direct_x:.6f}, {direct_y:.6f})')
    print(f'  transmission is {direct_f:.4f}\n')

    # check bins content
    # hex_centers = hb3.get_offsets()
    hex_val = hb3.get_array()
    print(f'Maximum expected flux is {max(all_f):.4f}')
    print(f'Maximum bin content is {max(hex_val):.4f}')

    # Show plot
    plt.show()
    # return 0 if the plot is shown
    return 0


def plot_zoom_on_ghosts(forward_rays):
    """ Plots the 2D image of the focal plane, and its projection along the x-axis

    Parameters
    ----------
    forward_rays : `list` of `batoid.RayVector`
        a list of forward rays, as each item in list comes from one distinct path through the optic exiting in
        the forward direction.  see `batoid.optic.traceSplit`

    .. todo::
        `plot_zoom_on_ghosts` should automatically zoom on the ghosts, if possible

    Returns
    -------
    0 : 0
        0 if all is well
    """
    # integrated data
    all_x, all_y, all_f = get_full_light_on_camera(forward_rays)
    # Trying to zoom in on ghosts images

    plt.rcParams["figure.figsize"] = [18, 6]
    fig, ax = plt.subplots(2, 1)
    axs = ax.ravel()
    # ghost images
    _ = axs[0].hexbin(all_x, all_y, C=all_f, reduce_C_function=np.sum,
                      extent=[-0.02, 0.27, -0.005, 0.005], gridsize=(100, 100))
    axs[0].set_aspect("equal")
    axs[0].set_title('Beam spot')

    # "Projection" on the x-axis shows that ghosts spots are nicely separated
    axs[1].hist(all_x, bins=1000, weights=all_f, log=True)
    axs[1].set_title('Projection of ghosts image on the x-axis')
    axs[1].set_xlabel('position x (m)')
    axs[1].set_ylabel('~n photons')
    return fig, ax


def plot_full_camera(forward_rays, log_scale=False):
    """ Plots the 2D image of the full focal plane

    Parameters
    ----------
    forward_rays : `list` of `batoid.RayVector`
        a list of forward rays, as each item in list comes from one distinct path through the optic exiting in
        the forward direction.  see `batoid.optic.traceSplit`
    log_scale : `bool`
        set to True to have a log color scale on the z-axis

    Returns
    -------
    0 : 0
        0 if all is well
    """
    plt.rcParams["figure.figsize"] = [18, 6]
    fig, ax = plt.subplots(1, 1)

    # ghosts images
    # Plot light on detector on the right
    all_x, all_y, all_f = get_full_light_on_camera(forward_rays)
    hb = ax.hexbin(all_x, all_y, C=all_f, reduce_C_function=np.sum,
                   extent=LSST_CAMERA_EXTENT, gridsize=(150, 150),
                   bins='log' if log_scale else None)

    # Plot approximate focal plane radius
    th = np.linspace(0, 2 * np.pi, 1000)
    plt.plot(0.32 * np.cos(th), 0.32 * np.sin(th), c='r')

    # Plot direct path location on focal plane
    i_straight, direct_x, direct_y, direct_f = get_main_impact_point(forward_rays)
    plt.text(direct_x, direct_y, '+', horizontalalignment='center',
             verticalalignment='center', color='m')
    ax.set_aspect("equal")
    ax.set_title('Image with ghosts')
    ax.set_xlabel('x (m)', fontsize=16)
    ax.set_ylabel('y (m)', fontsize=16)
    fig.colorbar(hb, ax=ax)

    # add info on direct path
    print(f'Direct path is number {i_straight}')
    print(f'  central impact point is ({direct_x:.6f}, {direct_y:.6f})')
    print(f'  transmission is {direct_f:.4f}\n')

    # check bins content
    # hex_centers = hb3.get_offsets()
    hex_val = hb.get_array()
    print(f'Maximum expected flux is {max(all_f):.4f}')
    print(f'Maximum bin content is {max(hex_val):.4f}')

    # return 0 if all is wel
    return fig, ax


def plot_ghosts_map(forward_rays):
    """ Plots a canvas with thumbnails of each one of the 37 possible images of the input beam
    on the focal plane, a.k.a. ghosts map.

    Parameters
    ----------
    forward_rays : `list` of `batoid.RayVector`
        a list of forward rays, as each item in list comes from one distinct path through the optic exiting in
        the forward direction.  see `batoid.optic.traceSplit`

    Returns
    -------
    spots_data : `list` of `dict`
        a list of dictionaries of ghost spot data (position, radius, brightness)
    """
    # plot all ghosts
    print("Ghosts map for 100 nW beam at 500 nm with a diameter of 2.5 mm")
    # get main impact point
    i_straight, direct_x, direct_y, _ = get_main_impact_point(forward_rays)
    # store some stats roughly
    spots_data = []
    # adjust rows and columns
    n_spots = len(forward_rays)
    if n_spots > 30:
        n_cols = 7
        n_rows = 6
    else:
        n_cols = 5
        n_rows = 5
    # build plot
    _, ax = plt.subplots(n_cols, n_rows, constrained_layout=True, figsize=(32, 32))
    axs = ax.ravel()
    for i, ghost in enumerate(forward_rays):
        # get ghost stats
        ghost_spot_data = get_ghost_spot_data(i, ghost)
        # bin data (and make plot)
        map_ghost(ghost, axs[i])
        # adjust plots
        axs[i].set_title(ghost_spot_data['name'])
        axs[i].grid(True)
        # make nice plot on axis
        x_min = 0.05
        axs[i].text(x_min, 0.95, f'Pos. x = {ghost_spot_data["pos_x"] * 1000:.2f} mm',
                    color='black', transform=axs[i].transAxes)
        axs[i].text(x_min, 0.9, f'Radius = {ghost_spot_data["radius"] * 1000:.2f} mm',
                    color='black', transform=axs[i].transAxes)
        axs[i].text(x_min, 0.85, f'Spot S = {ghost_spot_data["surface"]:.3f} mm$^2$',
                    color='black', transform=axs[i].transAxes)
        axs[i].text(x_min, 0.8, f'Phot. density = {ghost_spot_data["photon_density"]:.2e} ph/mm$^2$',
                    color='black', transform=axs[i].transAxes)
        axs[i].text(x_min, 0.75, f'Signal = {ghost_spot_data["pixel_signal"]:.2e} e$^-$/pixel',
                    color='black', transform=axs[i].transAxes)
        axs[i].set_aspect("equal")
        if i == i_straight:
            axs[i].text(direct_x, direct_y, '+', horizontalalignment='center',
                        verticalalignment='center', color='m')
            axs[i].set_title('Main image', color='m')
        # store data here
        spots_data.append(ghost_spot_data)
    for i in range(n_spots, len(axs)):
        axs[i].set_axis_off()
    return spots_data


# Looking at overall spots stats
def plot_spots_stats(data_frame):
    """ Plots overall ghosts image spots statistics

    Parameters
    ----------
    data_frame : `pandas.DataFrame`
        a pandas data frame with ghost spots data information, including beam configuration

    Returns
    -------
    fig : `matplotlib.Figure`
        the `matplotlib` figure object
    axs : `matplotlib.Axis`
        the list of `matplotlib` axis
    """
    plt.rcParams["figure.figsize"] = [24, 24]
    fig, ax = plt.subplots(2, 3)
    axs = ax.flatten()
    i = 0
    for i, col in enumerate(['pos_x', 'pos_y', 'radius']):
        axs[i].hist(data_frame[col] * 1000)
        axs[i].set_title(col, fontsize=22)
        axs[i].set_xlabel(f'{col} (mm)', fontsize=22)

    axs[i+1].hist(data_frame['surface'])
    axs[i+1].set_title('surface', fontsize=22)
    axs[i+1].set_xlabel('spot surface (mm$^2$)', fontsize=22)
    axs[i+2].hist(np.log10(data_frame['pixel_signal']))
    axs[i+2].set_title('pixel signal', fontsize=22)
    axs[i+2].set_xlabel('log10(signal) ($e^-$/pixel)', fontsize=22)
    return fig, axs


def plot_ghosts_spots_distances(ghosts_separations):
    """ Plots distances, ghost to ghost centers and borders

    Parameters
    ----------
    ghosts_separations : `pandas.DataFrame`
            a pandas data frame with information on ghost spots data separations and ratios

    Returns
    -------
    ax : `matplotlib.Axis`
        the list of `matplotlib` axis
    """
    plt.rcParams["figure.figsize"] = [18, 12]
    fig, ax = plt.subplots(2, 2)
    ax[0][0].hist(ghosts_separations['distance'] * 1000)
    ax[0][0].set_title('Distance between ghost spot centers', fontsize=22)
    ax[0][0].set_xlabel('distance (mm)', fontsize=16)

    ax[0][1].hist(ghosts_separations['overlap'] * 1000)
    ax[0][1].set_title('Distance between ghost spot borders', fontsize=22)
    ax[0][1].set_xlabel('distance (mm)', fontsize=16)

    ax[1][0].hist(np.log10(ghosts_separations['surface_ratio']))
    ax[1][0].set_title('Ghost spot surface ratio', fontsize=22)
    ax[1][0].set_xlabel('log10(ratio)', fontsize=16)

    ax[1][1].hist(np.log10(ghosts_separations['signal_ratio']))
    ax[1][1].set_title('Ghost spot pixel signal ratio', fontsize=22)
    ax[1][1].set_xlabel('log10(ratio)', fontsize=16)

    print(f'{sum(ghosts_separations["overlap"] < 0)} ghost spots pairs are in overlap out of {len(ghosts_separations)}')
    return fig, ax


def plot_ghosts_displacements(merged_data_frame):
    """ Plots a histogram of the displacement of all the ghosts along the x axis

    Parameters
    ----------
    merged_data_frame : `pandas.DataFrame`
        a pandas data frame with all the ghosts spot data information, for each telescope optics configuration,
        including beam configuration, see :meth:`ghosts.analysis.make_data_frame` and
        :meth:`ghosts.sim_scan_translated_optic`

    Returns
    -------
    ax : `matplotlib.Axis`
        the `matplotlib` axis
    """
    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches(18, 6)
    dx_mm = (merged_data_frame['pos_x_x'] - merged_data_frame['pos_x_y']) * 1000
    axs[0].hist(dx_mm)
    axs[0].set_xlabel('Ghost spot displacement x (mm)')
    dy_mm = (merged_data_frame['pos_y_x'] - merged_data_frame['pos_y_y']) * 1000
    axs[1].hist(dy_mm)
    axs[1].set_xlabel('Ghost spot displacement y (mm)')
    return axs


def plot_max_displacement_for_sim_scan(merged_data_frame, scan_values, trans_type='rotation'):
    """ Plots the value of the maximum displacement among all the ghosts for a series of translated
    or rotated telescope simulations, see :meth:`ghosts.sim_scan_translated_optic`.

    A linear fit is also done and plotted with its residuals.

    Parameters
    ----------
    merged_data_frame : `pandas.DataFrame`
        a pandas data frame with all the ghosts spot data information, for each telescope optics configuration,
        including beam configuration, see :meth:`ghosts.analysis.make_data_frame` and
        :meth:`ghosts.sim_scan_translated_optic`
    scan_values : `list` of `floats`
        the list of angles or shifts of the simulation scan
    trans_type : `string`
        the transformation type is either 'rotation' or 'translation'

    Returns
    -------
    0 : 0
        0 if all is well
    """
    # Get list of signed maximum displacements in mm
    x_max_diff = []
    y_max_diff = []
    for df in merged_data_frame:
        # shift along x
        x_tmp_diff = df['pos_x_x'] - df['pos_x_y']
        x_max_abs = max(abs(x_tmp_diff))
        if x_max_abs == x_tmp_diff.max():
            x_max_diff.append(x_max_abs * 1000)
        else:
            x_max_diff.append(-x_max_abs * 1000)
        # shift along y
        y_tmp_diff = df['pos_y_x'] - df['pos_y_y']
        y_max_abs = max(abs(y_tmp_diff))
        if y_max_abs == y_tmp_diff.max():
            y_max_diff.append(y_max_abs * 1000)
        else:
            y_max_diff.append(-y_max_abs * 1000)

    # Linear fit for x shift
    x_lin_fit = stats.linregress(scan_values, x_max_diff)
    print(f'Filter fit results: intercept = {x_lin_fit.intercept:.6f}, slope = {x_lin_fit.slope:.3f}')
    # Linear fit for y shift
    y_lin_fit = stats.linregress(scan_values, y_max_diff)
    print(f'Filter fit results: intercept = {y_lin_fit.intercept:.6f}, slope = {y_lin_fit.slope:.3f}')

    # Scatter plot with fit
    plt.rcParams["figure.figsize"] = [18, 12]
    fig, ax = plt.subplots(2, 2)
    # check if rotation or translation to get labels right
    x_label = ''
    insert_label = ''
    slope_factor = 1
    if trans_type == 'rotation':
        x_label = 'rotation angle (°)'
        insert_label = 'deg'
        slope_factor = 1
    elif trans_type == 'shift':
        x_label = 'shift (m)'
        insert_label = 'mm'
        slope_factor = 1000
    # plot for x
    ax[0][0].plot(scan_values, x_max_diff, 'o', label='data')
    x_interp_ys = [x_lin_fit.intercept + x_lin_fit.slope * x for x in scan_values]
    ax[0][0].plot(scan_values, x_interp_ys, 'r', label='linear fit')
    ax[0][0].legend()
    ax[0][0].set_title(f'Maximum ghost displacement as a function of element {trans_type}')
    ax[0][0].set_xlabel(x_label)
    ax[0][0].set_ylabel('Ghost spot displacement (mm)')
    ax[0][0].text(0.4, 0.9, f'slope = {x_lin_fit.slope/slope_factor:.1f} mm/{insert_label}', color='black', size=15,
                  ha='center', va='center', transform=ax[0][0].transAxes)

    # Residuals and fit for x
    x_residuals = np.array(x_interp_ys) - np.array(x_max_diff)
    (x_mu, x_sigma) = stats.norm.fit(x_residuals)
    _, x_bins, _ = ax[1][0].hist(x_residuals, bins=10, density=True)
    x_bin_centers = 0.5 * (x_bins[1:] + x_bins[:-1])
    x_y = stats.norm.pdf(x_bin_centers, x_mu, x_sigma)
    ax[1][0].plot(x_bin_centers, x_y, 'r--', linewidth=2)
    ax[1][0].set_title('Fit residuals (mm)')

    # plot for y
    ax[0][1].plot(scan_values, y_max_diff, 'o', label='data')
    y_interp_ys = [y_lin_fit.intercept + y_lin_fit.slope * x for x in scan_values]
    ax[0][1].plot(scan_values, y_interp_ys, 'r', label='linear fit')
    ax[0][1].legend()
    ax[0][1].set_title(f'Maximum ghost displacement as a function of element {trans_type}')
    ax[0][1].set_xlabel(x_label)
    ax[0][1].set_ylabel('Ghost spot displacement (mm)')
    ax[0][1].text(0.4, 0.9, f'slope = {y_lin_fit.slope/slope_factor:.1f} mm/{insert_label}', color='black', size=15,
                  ha='center', va='center', transform=ax[0][1].transAxes)

    # Residuals and fit for y
    y_residuals = np.array(y_interp_ys) - np.array(y_max_diff)
    (y_mu, y_sigma) = stats.norm.fit(y_residuals)
    _, y_bins, _ = ax[1][1].hist(y_residuals, bins=10, density=True)
    y_bin_centers = 0.5 * (y_bins[1:] + y_bins[:-1])
    y_y = stats.norm.pdf(y_bin_centers, y_mu, y_sigma)
    ax[1][1].plot(y_bin_centers, y_y, 'r--', linewidth=2)
    ax[1][1].set_title('Fit residuals (mm)')

    return fig, ax


def plot_spots(data_frame_list, spot_size_scaling=10, range_x=(-0.35, 0.35), range_y=(-0.35, 0.35)):
    """ Plot spots positions and size from a list of data frames

    Each data frame has a different marker color.

    Parameters
    ----------
    data_frame_list : `list` of `pandas.dataframe`
        a list of data frame with spots positions and radius, e.g. from `make_data_frame`
    spot_size_scaling : `int`
        a scaling factor to see large or small circles
    range_x : `tuple` of `floats`
        min and max of the x-axis in meters, default is full camera
    range_y : `tuple` of `floats`
        min and max of the y-axis in meters, default is full camera
    Returns
    -------
    fig: `matplotlib.Figure`
        the figure object
    ax: `matplotlib.Axis`
        the axis object
    """
    plt.rcParams["figure.figsize"] = [12, 12]
    fig, ax = plt.subplots(1, 1)
    colors = ['black', 'r', 'b', 'g', 'c', 'm', 'y', 'k']
    for df, color in zip(data_frame_list, colors):
        spots_x = df['pos_x']
        spots_y = df['pos_y']
        spots_size = ((df['radius'] * 1000) ** 2) * spot_size_scaling
        ax.scatter(spots_x, spots_y, s=spots_size, facecolors='none', edgecolors=color)
    ax.set_xlim(range_x)
    ax.set_ylim(range_y)
    return fig, ax


def plot_full_camera_and_spots(forward_rays, data_frame, log_scale=False, spot_size_scaling=10):
    """ Plots the 2D image of the full focal plane

    Parameters
    ----------
    forward_rays : `list` of `batoid.RayVector`
        a list of forward rays, as each item in list comes from one distinct path through the optic exiting in
        the forward direction.  see `batoid.optic.traceSplit`
    data_frame : `pandas.dataframe`
        a data frame with spots positions and radius, e.g. from `make_data_frame`
    log_scale : `bool`
        set to True to have a log color scale on the z-axis
    spot_size_scaling : `int`
        a scaling factor to see large or small circles

    Returns
    -------
    0 : 0
        0 if all is well
    """
    plt.rcParams["figure.figsize"] = [18, 9]
    fig, ax = plt.subplots(1, 2)

    # first the camera ghosts image
    all_x, all_y, all_f = get_full_light_on_camera(forward_rays)
    hb = ax[0].hexbin(all_x, all_y, C=all_f, reduce_C_function=np.sum,
                      extent=LSST_CAMERA_EXTENT, gridsize=(150, 150),
                      bins='log' if log_scale else None)

    # Plot approximate focal plane radius
    th = np.linspace(0, 2 * np.pi, 1000)
    plt.plot(0.32 * np.cos(th), 0.32 * np.sin(th), c='r')

    # Plot direct path location on focal plane
    _, direct_x, direct_y, _ = get_main_impact_point(forward_rays)
    plt.text(direct_x, direct_y, '+', horizontalalignment='center',
             verticalalignment='center', color='m')
    ax[0].set_aspect("equal")
    ax[0].set_title('Image with ghosts')
    ax[0].set_xlabel('x (m)', fontsize=16)
    ax[0].set_ylabel('y (m)', fontsize=16)
    fig.colorbar(hb, ax=ax[0])

    # Now visualize ghosts on the right
    spots_x = data_frame['pos_x']
    spots_y = data_frame['pos_y']
    spots_size = ((data_frame['radius'] * 1000) ** 2) * spot_size_scaling
    ax[1].scatter(spots_x, spots_y, s=spots_size, facecolors='none', edgecolors='black')
    ax[1].set_xlim((LSST_CAMERA_EXTENT[0], LSST_CAMERA_EXTENT[1]))
    ax[1].set_ylim((LSST_CAMERA_EXTENT[2], LSST_CAMERA_EXTENT[3]))
    ax[1].set_title('Ghosts representation')
    ax[1].set_xlabel('x (m)', fontsize=16)
    ax[1].set_ylabel('y (m)', fontsize=16)
    ax[1].set_aspect("equal")

    # Done
    return fig, ax


def plot_distances_for_scan(scan_values, distances_2d, distances_3d, scan_type='rotation'):
    """ Plot likelihood value and profile

    Parameters
    ----------
    scan_values : `list` of `floats`
        a list of angles or shifts in degrees
    distances_2d : `list` of `floats`
        a list of reduced distances computed in 2D
    distances_3d : `list` of `floats`
        a list of reduced distances computed in 3D
    scan_type : `string`
        either rotation or translation

    Returns
    -------
    fig: `matplotlib.Figure`
        the figure object
    ax: `matplotlib.Axis`
        the axis object
    """
    plt.rcParams["figure.figsize"] = [18, 9]
    fig, ax = plt.subplots(1, 2)
    # distributions
    ax[0].hist(distances_2d)
    ax[0].hist(distances_3d)
    # profiles
    ax[1].plot(scan_values, distances_2d, label='2D')
    ax[1].plot(scan_values, distances_3d, label='3D')
    ax[1].legend()
    ax[1].set_ylabel('distance')
    if scan_type == 'rotation':
        ax[1].set_xlabel('rotation angle degrees')
        ax[1].set_title(f'Rotation scan from {min(scan_values):.3f} to {max(scan_values):.3f} degrees')
    elif scan_type == 'translation':
        ax[1].set_xlabel('shift in meters')
        ax[1].set_title(f'Translation scan from {min(scan_values):.5f} to {max(scan_values):.5f} m')
    return fig, ax


def plot_impact_point_vs_beam_offset(data_frame):
    """ Plot likelihood value and profile

    Parameters
    ----------
    data_frame : `pandas.DataFrame`
        a panda data frame with information on beam positions and main impact points
        typically created with `simulator.simulate_impact_points_for_beam_set`

    Returns
    -------
    fig: `matplotlib.Figure`
        the figure object
    ax: `matplotlib.Axis`
        the axis object
    """
    plt.rcParams["figure.figsize"] = [18, 24]
    fig, ax = plt.subplots(4, 2)
    # 1 the incident beam
    ax[0][0].scatter(data_frame['x_offset'], data_frame['y_offset'])
    ax[0][0].set_aspect('equal')
    ax[0][0].set_xlabel('x_offset (m)')
    ax[0][0].set_ylabel('y_offset (m)')
    ax[0][0].set_title('beam')
    # 2 the impact point
    ax[0][1].scatter(data_frame['x_spot'], data_frame['y_spot'])
    ax[0][1].set_aspect('equal')
    ax[0][1].set_xlabel('x_spot (m)')
    ax[0][1].set_ylabel('y_spot (m)')
    ax[0][1].set_title('impact point')
    # 3 offset vs spot on x
    ax[1][0].scatter(data_frame['x_offset'], data_frame['x_spot'])
    ax[1][0].set_aspect('equal')
    ax[1][0].set_xlabel('x_offset (m)')
    ax[1][0].set_ylabel('x_spot (m)')
    ax[1][0].set_title('beam vs impact point in x')
    # 4 offset vs spot on y
    ax[1][1].scatter(data_frame['y_offset'], data_frame['y_spot'])
    ax[1][1].set_aspect('equal')
    ax[1][1].set_xlabel('y_offset (m)')
    ax[1][1].set_ylabel('y_spot (m)')
    ax[1][1].set_title('beam vs impact point in y')
    # 5 offset vs spot on distance
    ax[2][0].scatter(data_frame['beam_dist_to_center'], data_frame['spot_dist_to_center'])
    ax[2][0].set_aspect('equal')
    ax[2][0].set_xlabel('beam distance to center (m)')
    ax[2][0].set_ylabel('spot distance to center (m)')
    ax[2][0].set_title('distance to center')
    # 6 offset vs spot on convergence
    ax[2][1].scatter(data_frame['convergence'], data_frame['beam_dist_to_center'])
    ax[2][1].set_xlabel('convergence')
    ax[2][1].set_ylabel('beam distance to center (m)')
    # 7 displacement from offset to impact point
    ax[3][0].scatter(data_frame['beam_dist_to_center'], data_frame['displacement'])
    ax[3][0].set_xlabel('beam distance to center (m)')
    ax[3][0].set_ylabel('displacement (m)')
    # 8 convergence histogram
    _, c_x_bins, _ = ax[3][1].hist(data_frame['convergence'], bins=50, density=True)
    ax[3][1].set_xlabel('convergence (m)')
    ax[3][1].set_ylabel('density')

    # Linear fit for distance to center
    x_lin_fit = stats.linregress(data_frame['beam_dist_to_center'], data_frame['displacement'])
    print(f'Filter fit results: intercept = {x_lin_fit.intercept:.6f}, slope = {x_lin_fit.slope:.3f}')
    x_interp_ys = [x_lin_fit.intercept + x_lin_fit.slope * x for x in data_frame['beam_dist_to_center']]
    ax[3][0].plot(data_frame['beam_dist_to_center'], x_interp_ys, 'r', label='linear fit')
    # ax[3][0].legend()
    # write text
    y_min = data_frame['displacement'].min()
    ax[3][0].text(0.02, y_min * 0.8, f'Slope = {x_lin_fit.slope * 1000:.1f} mm / m', fontsize=20)

    # gaussian fit to convergence
    conv = data_frame['convergence'].dropna()
    red_conv = conv[(conv > 0.98) & (conv < 1.02)]
    (c_x_mu, c_x_sigma) = stats.norm.fit(red_conv)
    ax[3][1].text(ax[3][1].get_xlim()[0], ax[3][1].get_ylim()[1] * 0.95, f'Mean = {c_x_mu:.4f}', fontsize=16)
    ax[3][1].text(ax[3][1].get_xlim()[0], ax[3][1].get_ylim()[1] * 0.9, f'Sigma = {c_x_sigma:.4f}', fontsize=16)
    # plot gaussian fit
    c_x_bin_centers = 0.5 * (c_x_bins[1:] + c_x_bins[:-1])
    c_x_y = stats.norm.pdf(c_x_bin_centers, c_x_mu, c_x_sigma)
    ax[3][1].plot(c_x_bin_centers, c_x_y, 'r--', linewidth=2)
    return fig, ax


def plot_beam_pointing_precision(data_frame, target_x, target_y):
    """ Plot beam pointing precision as histograms of the differences in position
    of the beam and the target

    Parameters
    ----------
    data_frame : `pandas.DataFrame`
        a panda data frame with information on beam positions and main impact points
        typically created with `simulator.simulate_impact_points_for_beam_set`
    target_x : `float`
        the beam target position along x on the camera focal plane
    target_y : `float`
        the beam target position along y on the camera focal plane

    Returns
    -------
    fig: `matplotlib.Figure`
        the figure object
    ax: `matplotlib.Axis`
        the axis object
    """
    # have a look at the precision of the beam pointing to center
    plt.rcParams["figure.figsize"] = [18, 9]
    # make figure
    fig, ax = plt.subplots(1, 2)
    ax[0].hist((data_frame['x_spot'] - target_x) * 1000)
    ax[0].set_xlabel('delta x (mm)')
    ax[0].set_ylabel('n spots')
    ax[0].set_title('Difference between target and real impact point (x)')
    ax[0].yaxis.set_major_locator(MaxNLocator(integer=True))

    ax[1].hist((data_frame['y_spot'] - target_y) * 1000)
    ax[1].set_xlabel('delta y (mm)')
    ax[1].set_ylabel('n spots')
    ax[1].set_title('Difference between target and real impact point (y)')
    ax[1].yaxis.set_major_locator(MaxNLocator(integer=True))

    print('The pointing precision at order 0 is of the order of ~1 mm')
    return fig, ax


def plot_impact_points_full_frame(data_frame):
    """ Plot impact point of a beam set on the full camera focal plane

    Parameters
    ----------
    data_frame : `pandas.DataFrame`
        a panda data frame with information on beam positions and main impact points
        typically created with `simulator.simulate_impact_points_for_beam_set`

    Returns
    -------
    0
    """
    plt.rcParams["figure.figsize"] = [9, 9]
    plt.scatter(data_frame['x_spot'], data_frame['y_spot'])
    plt.xlim((LSST_CAMERA_EXTENT[0], LSST_CAMERA_EXTENT[1]))
    plt.ylim((LSST_CAMERA_EXTENT[2], LSST_CAMERA_EXTENT[3]))
    plt.gca().set_aspect('equal')
    plt.xlabel('Camera X')
    plt.ylabel('Camera Y')
    plt.title('Beam set impact points')
    th = np.linspace(0, 2 * np.pi, 1000)
    plt.plot(0.32 * np.cos(th), 0.32 * np.sin(th), c='r')
    return 0
