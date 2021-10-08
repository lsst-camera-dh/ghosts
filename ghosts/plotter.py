import batoid
import matplotlib.pyplot as plt
from scipy import stats
import ipyvolume as ipv
import numpy as np
from ghosts.tools import get_ranges, get_main_impact_point
from ghosts.analysis import get_full_light_on_camera, map_ghost, get_ghost_spot_data

def plot_setup(telescope, simulation):
    traceFull = simulation[0]
    rForward = simulation[1]
    rReverse = simulation[2]
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
    batoid.drawTrace2d(f1_ax1, traceFull, c='orange')

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
    all_x, all_y, all_f = get_full_light_on_camera(rForward)
    hb3 = f1_ax4.hexbin(all_x, all_y, C=all_f, reduce_C_function=np.sum,
                        extent=[-0.35, 0.35, -0.35, 0.35], gridsize=(150, 150))

    # Plot approximate focal plane radius
    th = np.linspace(0, 2 * np.pi, 1000)
    plt.plot(0.32 * np.cos(th), 0.32 * np.sin(th), c='r')

    # Plot direct path location on focal plane
    i_straight, direct_x, direct_y, direct_f = get_main_impact_point(rForward)
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
    hex_centers = hb3.get_offsets()
    hex_val = hb3.get_array()
    print(f'Maximum expected flux is {max(all_f):.4f}')
    print(f'Maximum bin content is {max(hex_val):.4f}')

    # Show plot
    plt.show()


def plot_zoom_on_ghosts(rForward):
    # integrated data
    all_x, all_y, all_f = get_full_light_on_camera(rForward)
    # Trying to zoom in on ghosts images

    plt.rcParams["figure.figsize"] = [18, 6]
    fig, ax = plt.subplots(2, 1)
    axs = ax.ravel()
    # ghosts images
    hb1 = axs[0].hexbin(all_x, all_y, C=all_f, reduce_C_function=np.sum,
                        extent=[-0.02, 0.27, -0.005, 0.005], gridsize=(100, 100))
    axs[0].set_aspect("equal")
    axs[0].set_title(f"Beam spot")

    # "Projection" on the x-axis shows that ghosts spots are nicely separated
    axs[1].hist(all_x, bins=1000, weights=all_f, log=True)
    axs[1].set_title("Projection of ghosts image on the x-axis")
    axs[1].set_xlabel('position x (mm)')
    axs[1].set_ylabel('~n photons')
    plt.show()


def plot_ghosts_map(rForward):
    # plot all ghosts
    print("Ghosts map for 100 nW beam at 500 nm with a diameter of 2.5 mm")
    # get main impact point
    i_straight, direct_x, direct_y, direct_f = get_main_impact_point(rForward)
    # store some stats roughly
    spots_data = list()
    fig, ax = plt.subplots(7, 6)
    axs = ax.ravel()
    for i, ghost in enumerate(rForward):
        # bin data (and make plot)
        hb_map = map_ghost(ghost, axs[i])
        axs[i].set_aspect("equal")
        axs[i].set_title(f"Ghost image")
        axs[i].grid(True)

        ghost_spot_data = get_ghost_spot_data(i, ghost)
        # make nice plot on axis
        x_min, _x_max, y_min, y_max = get_ranges(ghost.x, ghost.y, dr=0.01)
        axs[i].text(x_min, 0.9 * y_max, f'Pos. x = {ghost_spot_data["pos_x"] * 1000:.2f} mm', color='black')
        axs[i].text(x_min, 0.7 * y_max, f'Width x = {ghost_spot_data["width_x"] * 1000:.2f} mm', color='black')
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
    plt.rcParams["figure.figsize"] = [24, 24]
    fig, ax = plt.subplots(2, 2)
    axs = ax.flatten()
    for i, col in enumerate(['pos_x', 'width_x']):
        axs[i].hist(data_frame[col] * 1000)
        axs[i].set_title(col, fontsize=22)
        axs[i].set_xlabel('%s (mm)' % col, fontsize=22)

    axs[2].hist(data_frame['surface'])
    axs[2].set_title('surface', fontsize=22)
    axs[2].set_xlabel('spot %s (mm$^2$)' % col, fontsize=22)
    axs[3].hist(np.log10(data_frame['pixel_signal']))
    axs[3].set_title('pixel signal', fontsize=22)
    axs[3].set_xlabel('log10(signal) ($e^-$/pixel)', fontsize=22)
    return fig, axs

def plot_ghosts_spots_distances(ghosts_separations):
    # plotting distances ghost to ghost centers and borders
    plt.rcParams["figure.figsize"] = [18, 12]
    fig, ax = plt.subplots(2, 2)
    hd = ax[0][0].hist(ghosts_separations['distance'] * 1000)
    ax[0][0].set_title('Distance between ghost spot centers', fontsize=22)
    ax[0][0].set_xlabel('distance (mm)', fontsize=16)

    ho = ax[0][1].hist(ghosts_separations['overlap'] * 1000)
    ax[0][1].set_title('Distance between ghost spot borders', fontsize=22)
    ax[0][1].set_xlabel('distance (mm)', fontsize=16)

    hs = ax[1][0].hist(np.log10(ghosts_separations['surface_ratio']))
    ax[1][0].set_title('Ghost spot surface ratio', fontsize=22)
    ax[1][0].set_xlabel('log10(ratio)', fontsize=16)

    hp = ax[1][1].hist(np.log10(ghosts_separations['signal_ratio']))
    ax[1][1].set_title('Ghost spot pixel signal ratio', fontsize=22)
    ax[1][1].set_xlabel('log10(ratio)', fontsize=16)

    print(f'{sum(ghosts_separations["overlap"] < 0)} ghost spots pairs are in overlap out of {len(ghosts_separations)}')
    return ax

def plot_max_displacement_for_sim_scan(merged_data_frame, scan_angles, trans_type='rotation'):
    # Plot maximum displacement as a function of filter rotation angle
    # Get list of signed maximum displacements in mm
    x_max_diff = list()
    for df in merged_data_frame:
        tmp_diff = df['pos_x_x'] - df['pos_x_y']
        max_abs = max(abs(tmp_diff))
        if max_abs == tmp_diff.max():
            x_max_diff.append(max_abs * 1000)
        else:
            x_max_diff.append(-max_abs * 1000)

    # Linear fit
    lin_fit = stats.linregress(scan_angles, x_max_diff)
    print(f'Filter fit results: intercept = {lin_fit.intercept:.6f}, slope = {lin_fit.slope:.3f}')

    # Scatter plot with fit
    plt.rcParams["figure.figsize"] = [18, 6]
    fig, ax = plt.subplots(1, 2)
    ax[0].plot(scan_angles, x_max_diff, 'o', label='data')
    interp_ys = [lin_fit.intercept + lin_fit.slope * x for x in scan_angles]
    ax[0].plot(scan_angles, interp_ys, 'r', label='linear fit')
    ax[0].legend()
    ax[0].set_title(f'Maximum ghost displacement as a function of element {trans_type}')
    ax[0].set_ylabel('Ghost spot displacement (mm)')
    if trans_type == 'rotation':
        ax[0].set_xlabel('rotation angle (°)')
    elif trans_type == 'shift':
        ax[0].set_xlabel('shift (m)')

    # Residuals and fit
    residuals = np.array(interp_ys) - np.array(x_max_diff)
    (mu, sigma) = stats.norm.fit(residuals)
    n, bins, patches = ax[1].hist(residuals, bins=10, density=True)
    bincenters = 0.5 * (bins[1:] + bins[:-1])
    y = stats.norm.pdf(bincenters, mu, sigma)
    ax[1].plot(bincenters, y, 'r--', linewidth=2)
    ax[1].set_title('Fit residuals (mm)')
    plt.show()

