import batoid
import matplotlib.pyplot as plt
from scipy import stats
# import ipyvolume as ipv
import numpy as np
from ghosts.tools import get_ranges, get_main_impact_point
from ghosts.analysis import get_full_light_on_camera, map_ghost, get_ghost_spot_data


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

    # Plot input beam spot full scale
    beam_spot = rays.positionAtTime(3.397)
    hb1 = f1_ax2.hexbin(beam_spot[:, 0], beam_spot[:, 1], reduce_C_function=np.sum,
                        extent=[-0.35, 0.35, -0.35, 0.35], gridsize=(150, 150))
    f1_ax2.set_aspect("equal")
    f1_ax2.set_title(f"Beam spot")
    fig1.colorbar(hb1, ax=f1_ax2)

    # Plot input beam spot zoom in
    hb2 = f1_ax3.hexbin(beam_spot[:, 0], beam_spot[:, 1], reduce_C_function=np.sum,
                        extent=get_ranges(np.array(beam_spot[:, 0]), np.array(beam_spot[:, 1]), 0.01),
                        gridsize=(50, 50))
    f1_ax3.set_aspect("equal")
    f1_ax3.set_title(f"Beam spot zoom")
    fig1.colorbar(hb2, ax=f1_ax3)

    # Plot light on detector on the right
    all_x, all_y, all_f = get_full_light_on_camera(forward_rays)
    hb3 = f1_ax4.hexbin(all_x, all_y, C=all_f, reduce_C_function=np.sum,
                        extent=[-0.35, 0.35, -0.35, 0.35], gridsize=(150, 150))

    # Plot approximate focal plane radius
    th = np.linspace(0, 2 * np.pi, 1000)
    plt.plot(0.32 * np.cos(th), 0.32 * np.sin(th), c='r')

    # Plot direct path location on focal plane
    i_straight, direct_x, direct_y, direct_f = get_main_impact_point(forward_rays)
    plt.text(direct_x, direct_y, '+', horizontalalignment='center',
             verticalalignment='center', color='m')
    f1_ax4.set_aspect("equal")
    f1_ax4.set_title(f"Image with ghosts")
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
    # ghosts images
    _hb1 = axs[0].hexbin(all_x, all_y, C=all_f, reduce_C_function=np.sum,
                         extent=[-0.02, 0.27, -0.005, 0.005], gridsize=(100, 100))
    axs[0].set_aspect("equal")
    axs[0].set_title(f"Beam spot")

    # "Projection" on the x-axis shows that ghosts spots are nicely separated
    axs[1].hist(all_x, bins=1000, weights=all_f, log=True)
    axs[1].set_title("Projection of ghosts image on the x-axis")
    axs[1].set_xlabel('position x (mm)')
    axs[1].set_ylabel('~n photons')
    plt.show()
    # return 0 if all is wel
    return 0


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
    i_straight, direct_x, direct_y, direct_f = get_main_impact_point(forward_rays)
    # store some stats roughly
    spots_data = list()
    fig, ax = plt.subplots(7, 6)
    axs = ax.ravel()
    for i, ghost in enumerate(forward_rays):
        # bin data (and make plot)
        _hb_map = map_ghost(ghost, axs[i])
        axs[i].set_aspect("equal")
        axs[i].set_title(f"Ghost image")
        axs[i].grid(True)

        ghost_spot_data = get_ghost_spot_data(i, ghost)
        # make nice plot on axis
        x_min, _x_max, y_min, y_max = get_ranges(ghost.x, ghost.y, dr=0.01)
        axs[i].text(x_min, 0.9 * y_max, f'Pos. x = {ghost_spot_data["pos_x"] * 1000:.2f} mm', color='black')
        axs[i].text(x_min, 0.7 * y_max, f'Radius = {ghost_spot_data["radius"] * 1000:.2f} mm', color='black')
        axs[i].text(x_min, 0.5 * y_min, f'Spot S = {ghost_spot_data["surface"]:.3f} mm$^2$', color='black')
        axs[i].text(x_min, 0.7 * y_min, f'Phot. density = {ghost_spot_data["photon_density"]:.2e} ph/mm$^2$',
                    color='black')
        axs[i].text(x_min, 0.9 * y_min, f'Signal = {ghost_spot_data["pixel_signal"]:.2e} e$^-$/pixel', color='black')

        if i == i_straight:
            axs[i].text(direct_x, direct_y, '+', horizontalalignment='center',
                        verticalalignment='center', color='m')
            axs[i].set_title('Main image', color='m')
        # store data here
        spots_data.append(ghost_spot_data)
    plt.tight_layout()
    plt.show()
    return spots_data


# Looking at overal spots stats
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
    for i, col in enumerate(['pos_x', 'pos_y', 'radius']):
        axs[i].hist(data_frame[col] * 1000)
        axs[i].set_title(col, fontsize=22)
        axs[i].set_xlabel('%s (mm)' % col, fontsize=22)

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
    _hd = ax[0][0].hist(ghosts_separations['distance'] * 1000)
    ax[0][0].set_title('Distance between ghost spot centers', fontsize=22)
    ax[0][0].set_xlabel('distance (mm)', fontsize=16)

    _ho = ax[0][1].hist(ghosts_separations['overlap'] * 1000)
    ax[0][1].set_title('Distance between ghost spot borders', fontsize=22)
    ax[0][1].set_xlabel('distance (mm)', fontsize=16)

    _hs = ax[1][0].hist(np.log10(ghosts_separations['surface_ratio']))
    ax[1][0].set_title('Ghost spot surface ratio', fontsize=22)
    ax[1][0].set_xlabel('log10(ratio)', fontsize=16)

    _hp = ax[1][1].hist(np.log10(ghosts_separations['signal_ratio']))
    ax[1][1].set_title('Ghost spot pixel signal ratio', fontsize=22)
    ax[1][1].set_xlabel('log10(ratio)', fontsize=16)

    print(f'{sum(ghosts_separations["overlap"] < 0)} ghost spots pairs are in overlap out of {len(ghosts_separations)}')
    return ax


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

    Returns
    -------
    0 : 0
        0 if all is well
    """
    # Get list of signed maximum displacements in mm
    x_max_diff = list()
    y_max_diff = list()
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
    plt.rcParams["figure.figsize"] = [18, 6]
    fig, ax = plt.subplots(2, 2)
    # plot for x
    ax[0][0].plot(scan_values, x_max_diff, 'o', label='data')
    x_interp_ys = [x_lin_fit.intercept + x_lin_fit.slope * x for x in scan_values]
    ax[0][0].plot(scan_values, x_interp_ys, 'r', label='linear fit')
    ax[0][0].legend()
    ax[0][0].set_title(f'Maximum ghost displacement as a function of element {trans_type}')
    ax[0][0].set_ylabel('Ghost spot displacement (mm)')
    if trans_type == 'rotation':
        ax[0][0].set_xlabel('rotation angle (°)')
    elif trans_type == 'shift':
        ax[0][0].set_xlabel('shift (m)')
    # Residuals and fit for x
    x_residuals = np.array(x_interp_ys) - np.array(x_max_diff)
    (x_mu, x_sigma) = stats.norm.fit(x_residuals)
    x_n, x_bins, x_patches = ax[1][0].hist(x_residuals, bins=10, density=True)
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
    ax[0][1].set_ylabel('Ghost spot displacement (mm)')
    if trans_type == 'rotation':
        ax[0][1].set_xlabel('rotation angle (°)')
    elif trans_type == 'shift':
        ax[0][1].set_xlabel('shift (m)')
    # Residuals and fit for y
    y_residuals = np.array(y_interp_ys) - np.array(y_max_diff)
    (y_mu, y_sigma) = stats.norm.fit(y_residuals)
    y_n, y_bins, y_patches = ax[1][1].hist(y_residuals, bins=10, density=True)
    y_bin_centers = 0.5 * (y_bins[1:] + y_bins[:-1])
    y_y = stats.norm.pdf(y_bin_centers, y_mu, y_sigma)
    ax[1][1].plot(y_bin_centers, y_y, 'r--', linewidth=2)
    ax[1][1].set_title('Fit residuals (mm)')

    plt.show()
    # return 0 if all is well
    return 0
