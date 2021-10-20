import math
import numpy as np
import pandas as pd
from ghosts.tweak_optics import rotate_optic, make_optics_reflective, translate_optic, randomized_telescope
from ghosts.beam_configs import BEAM_CONFIG_0
from ghosts.beam import beam
from ghosts.analysis import reduce_ghosts, make_data_frame, compute_ghost_separations


# run a ray tracing simulation
def run_simulation(telescope_setup, beam_config=BEAM_CONFIG_0):
    """ Takes a telescope optical object and a beam configuration as a dictionnary
    Simulates the light beam through the optics
    Returns the full ray tracing, including ghosts and the beam as a vector of light rays
    """
    # Beam On
    rays = beam(beam_config)

    # Trace full optics and plot on the camera system
    r_forward, r_reverse = telescope_setup.traceSplit(rays, minFlux=1e-4)
    trace_full = telescope_setup.traceFull(rays)
    return trace_full, r_forward, r_reverse, rays


# Rotate and simulate
def full_rotation(telescope, optic_name='L2', angle=0.1, debug=False):
    rotated_optic = rotate_optic(telescope, optic_name, axis='y', angle=angle)
    if debug:
        print(
            f'{optic_name} rotation of {angle:.3f}° means a displacement of {300 * math.tan(angle * 3.14 / 180.):.3f}\
             mm of the lens border.')
    make_optics_reflective(rotated_optic)
    trace_full_o, r_forward_o, r_reverse_o, rays_o = run_simulation(rotated_optic, beam_config=BEAM_CONFIG_0)
    spots_data_o, _spots = reduce_ghosts(r_forward_o)
    data_frame_o = make_data_frame(spots_data_o)
    ghost_separations_o = compute_ghost_separations(data_frame_o)
    return data_frame_o, ghost_separations_o


# Rotating L2 specifically
def full_rotation_L2(telescope, angle=0.1):
    return full_rotation(telescope, optic_name='L2', angle=angle)


# Helpers to run and plot a scan in one optical element rotation
def sim_scan_rotated_optic(telescope, optic_name, min_angle, max_angle, step_angle, ref_data_frame):
    """ @TODO handle better reference data frame
    """
    print(f'Starting {optic_name} rotation scan.')
    rotation_sims = list()
    scan_angles = list()
    for angle in np.arange(min_angle, max_angle, step_angle):
        scan_angles.append(angle)
        print(f'{angle:.3f}', end=' ')
        df, _ = full_rotation(telescope, optic_name=optic_name, angle=angle)
        rotation_sims.append(df)

    # Merge data frames
    merged_data_frame = [pd.merge(ref_data_frame, df, how='left', on='name') for df in rotation_sims]
    print('Done.')
    return merged_data_frame, scan_angles


# Translate and simulate
def full_translation(telescope, optic_name='L2', distance=0.01):
    translated_optic = translate_optic(telescope, optic_name, axis='x', distance=distance)
    make_optics_reflective(translated_optic)
    trace_full_s, r_forward_s, r_reverse_s, rays_s = run_simulation(translated_optic, beam_config=BEAM_CONFIG_0)
    spots_data_s, _spots = reduce_ghosts(r_forward_s)
    data_frame_s = make_data_frame(spots_data_s)
    ghost_separations_s = compute_ghost_separations(data_frame_s)
    return data_frame_s, ghost_separations_s


# Helpers to run and plot a scan in one optical element translation
def sim_scan_translated_optic(telescope, optic_name, min_dist, max_dist, step_dist, ref_data_frame):
    """ @TODO handle better reference data frame
    """
    print(f'Starting {optic_name} translation scan.')
    sims = list()
    scan_values = list()
    for shift in np.arange(min_dist, max_dist, step_dist):
        scan_values.append(shift)
        print(f'{shift:.6f}', end=' ')
        df, _ = full_translation(telescope, optic_name=optic_name, distance=shift)
        sims.append(df)

    # Merge data frames
    merged_data_frame = [pd.merge(ref_data_frame, df, how='left', on='name') for df in sims]
    print('Done.')
    return merged_data_frame, scan_values


def full_random_telescope_sim(telescope, max_angle, max_shift, beam_config=BEAM_CONFIG_0):
    rnd_tel = randomized_telescope(telescope, max_angle, max_shift)
    make_optics_reflective(rnd_tel)
    trace_full_r, r_forward_r, r_reverse_r, rays_r = run_simulation(rnd_tel, beam_config=beam_config)
    spots_data_r, _spots = reduce_ghosts(r_forward_r)
    data_frame_r = make_data_frame(spots_data_r)
    ghost_separations_r = compute_ghost_separations(data_frame_r)
    return data_frame_r, ghost_separations_r
