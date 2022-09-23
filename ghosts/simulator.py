"""simulator module

This module provides tools to run full simulations from telescope geometries and beam configurations,
some functions also run the full workflow to the beam spots data analysis.
"""

import concurrent
import math
import numpy as np
import pandas as pd
from ghosts.tweak_optics import rotate_optic, make_optics_reflective, translate_optic, randomized_telescope
from ghosts.tweak_optics import build_telescope_from_geom, build_telescope, tweak_telescope
from ghosts.beam_configs import BEAM_CONFIG_1
from ghosts.beam import beam_on, concat_dicts
from ghosts.analysis import reduce_ghosts, make_data_frame, compute_ghost_separations, match_ghosts,\
                            compute_reduced_distance, compute_2d_reduced_distance
from ghosts.tools import get_main_impact_point, get_default_yaml_path


# run a ray tracing simulation
def run_simulation(telescope, beam_config):
    """ Runs a ray tracing simulation of a light beam into the CCOB.

    Takes a telescope optical object and a beam configuration as a dictionary,
    simulates the light beam through the optics and returns the full ray tracing,
    including ghosts and the beam as a vector of light ays

    Parameters
    ----------
    telescope : `batoid.telescope`
        the optical setup
    beam_config : `dict`
        a dictionary with the light beam configuration, see :ref:`beam_configs`.

    Returns
    -------
    trace_full : `OrderedDict of dict`
        an ordered dictionary of dictionaries for incoming and outgoing rays for each optics interface,
        see `batoid.optic.traceFull`
    forward_rays : `list` of `batoid.RayVector`
        a list of forward rays, as each item in list comes from one distinct path through the optic exiting in
        the forward direction.  see `batoid.optic.traceSplit`
    reverse_rays : `list` of `batoid.RayVector`
        the list of reverse rays, as each item in list comes from one distinct path through the optic exiting
        in the reverse direction., see `batoid.optic.traceSplit`
    rays : `batoid.RayVector`
        the input light beam, generated by the `beam` functions see :meth:`ghosts.beam.beam`.
    """
    # Beam On
    rays = beam_on(beam_config)
    # Trace full optics and plot on the camera system
    forward_rays, reverse_rays = telescope.traceSplit(rays, minFlux=1e-4)
    trace_full = telescope.traceFull(rays)
    return trace_full, forward_rays, reverse_rays, rays


def run_and_analyze_simulation(telescope, geom_id, beam_config):
    """ Runs a ray tracing simulation of a light beam into the CCOB
    and analyze beam spots data.

    Parameters
    ----------
    telescope : `batoid.telescope`
        the optical setup
    geom_id : `int`
        an integer corresponding to the id of the geometry of the telescope
    beam_config : `dict`
        a dictionary with the light beam configuration, see :ref:`beam_configs`.

    Returns
    -------
    spots_data_frame : `pandas.DataFrame`
        a `pandas` data frame with ghost spot data information, including beam and geometry configurations,
        see :meth:`ghosts.analysis.make_data_frame`
    """
    _, forward_rays, _, _ = run_simulation(telescope, beam_config)
    spots_data, _ = reduce_ghosts(forward_rays)
    spots_data_frame = make_data_frame(spots_data,
                                       beam_id=beam_config['beam_id'],
                                       geom_id=geom_id)
    return spots_data_frame


# run a ray tracing simulation
def run_simulation_from_configs(geom_config, beam_config):
    """ Runs a ray tracing simulation of a light beam into the CCOB.

    Takes a telescope optical object and a beam configuration as a dictionary,
    simulates the light beam through the optics and returns the full ray tracing,
    including ghosts and the beam as a vector of light ays

    Parameters
    ----------
    geom_config : `dict`
        a dictionary with the geometry configuration, see :ref:`geom_configs`.
    beam_config : `dict`
        a dictionary with the light beam configuration, see :ref:`beam_configs`.

    Returns
    -------
    trace_full : `OrderedDict of dict`
        an ordered dictionary of dictionaries for incoming and outgoing rays for each optics interface,
        see `batoid.optic.traceFull`
    forward_rays : `list` of `batoid.RayVector`
        a list of forward rays, as each item in list comes from one distinct path through the optic exiting in
        the forward direction.  see `batoid.optic.traceSplit`
    reverse_rays : `list` of `batoid.RayVector`
        the list of reverse rays, as each item in list comes from one distinct path through the optic exiting
        in the reverse direction., see `batoid.optic.traceSplit`
    rays : `batoid.RayVector`
        the input light beam, generated by the `beam` functions see :meth:`ghosts.beam.beam`.
    """
    # Beam On
    rays = beam_on(beam_config)
    # build telescope
    telescope = build_telescope_from_geom(geom_config)
    # Trace full optics and plot on the camera system
    forward_rays, reverse_rays = telescope.traceSplit(rays, minFlux=1e-4)
    trace_full = telescope.traceFull(rays)
    return trace_full, forward_rays, reverse_rays, rays


def run_full_simulation_from_configs(geom_config, beam_config):
    """ Runs and analyse a ray tracing simulation of a light beam into the CCOB.

    Parameters
    ----------
    geom_config : `dict`
        a dictionary with the geometry configuration, see :ref:`geom_configs`.
    beam_config : `dict`
        a dictionary with the light beam configuration, see :ref:`beam_configs`.

    Returns
    -------
    spots_data_frame : `pandas.DataFrame`
        a `pandas` data frame with ghost spot data information, including beam and geometry configurations,
        see :meth:`ghosts.analysis.make_data_frame`
    """
    _, forward_rays, _, _ = run_simulation_from_configs(geom_config, beam_config)
    spots_data, _ = reduce_ghosts(forward_rays)
    spots_data_frame = make_data_frame(spots_data,
                                       beam_id=beam_config['beam_id'],
                                       geom_id=geom_config['geom_id'])
    return spots_data_frame


def run_and_analyze_simulation_for_configs_sets(geom_set, beam_set):
    """ Runs and analyze a ray tracing simulation of a light beam into the CCOB
    for a set of beam configurations and geometry configurations

    Note that we first build a reference telescope and then tweak it at will,
    as building a telescope from yaml file is slow.

    Parameters
    ----------
    geom_set : `list` of `dict`
        a dictionary with the  geometry configuration, see :ref:`geom_configs`.
    beam_set : `list` of `dict`
        a dictionary with the light beam configuration, see :ref:`beam_configs`.

    Returns
    -------
    spots_data_frame : `pandas.DataFrame`
        a `pandas` data frame with ghost spot data information, including beam configuration,
        see :meth:`ghosts.analysis.make_data_frame`, merged from different configurations
    """
    # build one telescope to start with, as this is slow
    telescope = build_telescope(get_default_yaml_path())
    # go for the loops
    spots_df_list = []
    for one_geom in geom_set:
        current_tel = tweak_telescope(telescope, one_geom)
        make_optics_reflective(current_tel, coating='smart', r_frac=[0.02, 0.02, 0.15])
        for one_beam in beam_set:
            print(f'Run and analyze simulation: geom {one_geom["geom_id"]}, beam {one_beam["beam_id"]}', end='\r')
            results_data_frame = run_and_analyze_simulation(telescope, one_geom['geom_id'], one_beam)
            spots_df_list.append(results_data_frame)

    return spots_df_list


def run_and_analyze_simulation_for_configs_sets_parallel(geom_set, beam_set):
    """ Runs and analyze a ray tracing simulation of a light beam into the CCOB
    for a set of beam configurations and geometry configurations, multithread version.

    Note that we first build a reference telescope and then tweak it at will,
    as building a telescope from yaml file is slow. Threading is done with a pool
    on beam configuration for each geometry.

    Parameters
    ----------
    geom_set : `list` of `dict`
        a dictionary with the  geometry configuration, see :ref:`geom_configs`.
    beam_set : `list` of `dict`
        a dictionary with the light beam configuration, see :ref:`beam_configs`.

    Returns
    -------
    spots_data_frame : `pandas.DataFrame`
        a `pandas` data frame with ghost spot data information, including beam configuration,
        see :meth:`ghosts.analysis.make_data_frame`, merged from different configurations
    """
    # build one telescope to start with, as this is slow
    telescope = build_telescope(get_default_yaml_path())
    # go for the loops
    spots_df_list = []
    for one_geom in geom_set:
        current_tel = tweak_telescope(telescope, one_geom)
        make_optics_reflective(current_tel, coating='smart', r_frac=[0.02, 0.02, 0.15])
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(run_and_analyze_simulation, telescope, one_geom['geom_id'], one_beam) for
                       one_beam in beam_set]
            for future in concurrent.futures.as_completed(futures):
                spots_df_list.append(future.result())

    return spots_df_list


# Rotate and simulate
def full_rotation(telescope, optic_name='L2', axis='y', angle=0.1, beam_config=BEAM_CONFIG_1, debug=False):
    """ Runs a ray tracing simulation of a light beam into the CCOB with one of the optical element rotated.

    Takes a telescope optical object, the name of an optical element and a rotation angle, and
    simulates the light beam through the optics and returns the full ray tracing,
    including ghosts and the beam as a vector of light ays

    Parameters
    ----------
    telescope : `batoid.telescope`
        the optical setup as defined in `batoid`
    optic_name : `string`
        the name of the optical element to rotate
    axis : `string`
        x, y, z as the rotation axis you wish
    angle : `float`
        the rotation angle in degrees
    beam_config : `dict`
        a dictionary with the light beam configuration, see :ref:`beam_configs`.
    debug : `bool`
        debug mode or not

    Returns
    -------
    spots_data_frame : `pandas.DataFrame`
        a `pandas` data frame with ghost spot data information, including beam configuration,
        see :meth:`ghosts.analysis.make_data_frame`
    ghost_separations : `pandas.DataFrame`
        a `pandas` data frame with information on ghost spots data separations and ratios,
        see :meth:`ghosts.analysis.compute_ghost_separations`
    """
    rotated_optic = rotate_optic(telescope, optic_name, axis=axis, angle=angle)
    if debug:
        print(
            f'{optic_name} rotation of {angle:.3f}° means a displacement of'
            f' {300 * math.tan(angle * math.pi / 180.):.3f}\
             mm of the lens border.')
    make_optics_reflective(rotated_optic)
    _, r_forward, _, _ = run_simulation(rotated_optic, beam_config)
    spots_data, _ = reduce_ghosts(r_forward)
    spots_data_frame = make_data_frame(spots_data)
    ghost_separations = compute_ghost_separations(spots_data_frame)
    return spots_data_frame, ghost_separations


# Rotating L2 specifically
def full_rotation_L2(telescope, angle=0.1):
    """ Simple proxy to run a simulation with a rotation of L2, see :meth:`ghosts.simulator.full_rotation`

    Parameters
    ----------
    telescope : `batoid.telescope`
        the optical setup as defined in `batoid`
    angle : `float`
        the rotation angle in degrees

    Returns
    -------
    spots_data_frame : `pandas.DataFrame`
        a `pandas` data frame with ghost spot data information, including beam configuration,
        see :meth:`ghosts.analysis.make_data_frame`
    ghost_separations : `pandas.DataFrame`
        a `pandas` data frame with information on ghost spots data separations and ratios,
        see :meth:`ghosts.analysis.compute_ghost_separations`
    """
    return full_rotation(telescope, optic_name='L2', angle=angle)


def sim_scan_rotated_optic(telescope, optic_name, min_angle, max_angle, step_angle, ref_data_frame):
    """ Helper to run and plot a scan in one optical element rotation

    .. todo::
        `sim_scan_rotated_optic` should handle better reference data frame

    Parameters
    ----------
    telescope : `batoid.telescope`
        the optical setup as defined in `batoid`
    optic_name : `string`
        the name of the optical element to rotate
    min_angle : `float`
        the first rotation angle in degrees
    max_angle : `float`
        the last rotation angle in degrees
    step_angle : `float`
        the angle step size for the rotation in degrees
    ref_data_frame : `pandas.DataFrame`

    Returns
    -------
    merged_data_frame : `pandas.DataFrame`
        a`pandas` data frame with all the ghosts spot data information, for each telescope optics configuration,
        including beam configuration, see :meth:`ghosts.analysis.make_data_frame`
    scan_angles : `list` of `floats`
        the list of the rotation angles used for the scan
    """
    print(f'Starting {optic_name} rotation scan.')
    rotation_sims = []
    scan_angles = []
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
def full_translation(telescope, optic_name, axis, distance, beam_config=BEAM_CONFIG_1):
    """ Runs a ray tracing simulation of a light beam into the CCOB, with one optical element translated.

    Takes a telescope optical object, the name of an optical element, and a translation distance,
    simulates the light beam through the optics and returns the full ray tracing,
    including ghosts and the beam as a vector of light ays

    Parameters
    ----------
    telescope : `batoid.telescope`
        the optical setup as defined in `batoid`
    optic_name : `string`
        the name of the optical element to rotate
    axis : `string`
        x, y, z as the translation axis you wish
    distance : `float`
        the distance in meters
    beam_config : `dict`
        a dictionary with the light beam configuration, see :ref:`beam_configs`.

    Returns
    -------
    spots_data_frame : `pandas.DataFrame`
        a`pandas` data frame with ghost spot data information, including beam configuration,
        see :meth:`ghosts.analysis.make_data_frame`
    ghost_separations : `pandas.DataFrame`
        a`pandas` data frame with information on ghost spots data separations and ratios,
        see :meth:`ghosts.analysis.compute_ghost_separations`
    """
    translated_optic = translate_optic(telescope, optic_name, axis=axis, distance=distance)
    make_optics_reflective(translated_optic)
    _, forward_rays, _, _ = run_simulation(translated_optic, beam_config=beam_config)
    spots_data, _ = reduce_ghosts(forward_rays)
    spots_data_frame = make_data_frame(spots_data)
    ghost_separations = compute_ghost_separations(spots_data_frame)
    return spots_data_frame, ghost_separations


# Helpers to run and plot a scan in one optical element translation
def sim_scan_translated_optic(telescope, optic_name, min_dist, max_dist, step_dist, ref_data_frame):
    """ Helper to run a set of ray tracing simulation of a light beam into the CCOB,
     with one optical element translated along one axis of a set of difference distance

    Takes a telescope optical object, the name of an optical element, and a range for the translation distances,
    simulates the light beam through the optics and returns the full ray tracing,
    including ghosts and the beam as a vector of light ays

    .. todo::
        `sim_scan_translated_optic` needs to better handle the reference data frame

    .. todo::
        `sim_scan_translated_optic` should take a list of distances rather than min, max, step

    Parameters
    ----------
    telescope : `batoid.telescope`
        the optical setup as defined in `batoid`
    optic_name : `string`
        the name of the optical element to rotate
    min_dist : `float`
        the start distance in meters
    max_dist : `float`
        the stop distance in meters
    step_dist : `float`
        the step size as a distance in meters
    ref_data_frame : `pandas.dataFrame`
        a data frame with your ghosts spots reference data

    Returns
    -------
    merged_data_frame : `pandas.DataFrame`
        a`pandas` data frame with all the ghosts spot data information, for each telescope optics configuration,
        including beam configuration, see :meth:`ghosts.analysis.make_data_frame`
    scan_values : `list` of `floats`
        the list of distance of translation used for the scan
    """
    print(f'Starting {optic_name} translation scan.')
    sims = []
    scan_values = []
    for shift in np.arange(min_dist, max_dist, step_dist):
        scan_values.append(shift)
        print(f'{shift:.6f}', end=' ')
        df, _ = full_translation(telescope, optic_name=optic_name, axis='x', distance=shift)
        sims.append(df)
    # Merge data frames
    merged_data_frame = [pd.merge(ref_data_frame, df, how='left', on='name') for df in sims]
    print('Done.')
    return merged_data_frame, scan_values


def full_random_telescope_sim(telescope, max_angle, max_shift, beam_config):
    """ Runs a ray tracing simulation of a light beam into the CCOB, with optics with randomized positions
     and rotation angles.

    Takes a telescope optical object, the maximum value of the rotation angle and translation distance, and the
    beam configuration dictionary. Random distributions for the angle and distance are uniform.

    Parameters
    ----------
    telescope : `batoid.telescope`
        the optical setup as defined in `batoid`
    max_angle : `float`
        the maximum value of the rotation angle, a value will be taken out of a uniform distribution in
        [-max_angle ; +max_angle]
    max_shift : `float`
        the maximum value of the translation distance, a value will be taken out of a uniform distribution in
        [-max_shift ; +max_shift]
    beam_config : `dict`
        a beam configuration dictionary

    Returns
    -------
    spots_data_frame : `pandas.DataFrame`
        a`pandas` data frame with ghost spot data information, including beam configuration,
        see :meth:`ghosts.analysis.make_data_frame`
    ghost_separations : `pandas.DataFrame`
        a`pandas` data frame with information on ghost spots data separations and ratios,
        see :meth:`ghosts.analysis.compute_ghost_separations`
    """
    rnd_tel = randomized_telescope(telescope, max_angle, max_shift)
    make_optics_reflective(rnd_tel)
    _, r_forward_r, _, _ = run_simulation(rnd_tel, beam_config=beam_config)
    spots_data_r, _ = reduce_ghosts(r_forward_r)
    data_frame_r = make_data_frame(spots_data_r)
    ghost_separations_r = compute_ghost_separations(data_frame_r)
    return data_frame_r, ghost_separations_r


def scan_dist_rotation(telescope, ref_data_frame, optic_name, axis, angles_list, r_scale=10):
    """ Run simulation to scan a given list of angles on one optic around an axis,
    and computes the reduced distance in 2D and 3D.

    Parameters
    ----------
    telescope : `batoid.telescope`
        the optical setup as defined in `batoid`
    ref_data_frame : `pandas.DataFrame`
        the reference set of beam spots to compute distances to
    optic_name : `string`
        the name of the optical element to rotate
    axis : `string`
        x, y, z as the translation axis you wish
    angles_list : `list` of `float`
        a list of angles to scan
    r_scale : `float`
        the 3D distance scale factor to take into account the spots sizes

    Returns
    -------
    angles_list : `list` of `float`
        the list of angles scanned
    distances_2d : `list` of `float`
        the list of 2D reduced distance computed for each angle
    distances_3d : `list` of `float`
        the list of 3D reduced distance computed for each angle
    """
    distances_2d = []
    distances_3d = []
    for angle in angles_list:
        df_i, _ = full_rotation(telescope, optic_name=optic_name, axis=axis, angle=angle,
                                beam_config=BEAM_CONFIG_1)
        match_i = match_ghosts(ref_data_frame, df_i, radius_scale_factor=r_scale)
        dist_i = compute_reduced_distance(match_i)
        distances_3d.append(dist_i)

        match_i2 = match_ghosts(ref_data_frame, df_i, radius_scale_factor=r_scale)
        dist_i2 = compute_2d_reduced_distance(match_i2)
        distances_2d.append(dist_i2)

        print(f'{angle} ', end='', flush=True)

    return angles_list, distances_2d, distances_3d


def scan_dist_translation(telescope, ref_data_frame, optic_name, axis, shifts_list, r_scale=10):
    """ Run simulation to scan a given list of shifts on one optic along an axis,
    and computes the reduced distance in 2D and 3D.

    Parameters
    ----------
    telescope : `batoid.telescope`
        the optical setup as defined in `batoid`
    ref_data_frame : `pandas.DataFrame`
        the reference set of beam spots to compute distances to
    optic_name : `string`
        the name of the optical element to translate
    axis : `string`
        x, y, z as the translation axis you wish
    shifts_list : `list` of `float`
        a list of shifts to scan
    r_scale : `float`
        the 3D distance scale factor to take into account the spots sizes

    Returns
    -------
    shifts_list : `list` of `float`
        a list of shifts to scan
    distances_2d : `list` of `float`
        the list of 2D reduced distance computed for each angle
    distances_3d : `list` of `float`
        the list of 3D reduced distance computed for each angle
    """
    distances_2d = []
    distances_3d = []
    for delta in shifts_list:
        df_i, _ = full_translation(telescope, optic_name=optic_name, axis=axis, distance=delta,
                                   beam_config=BEAM_CONFIG_1)
        match_i = match_ghosts(ref_data_frame, df_i, radius_scale_factor=r_scale)
        dist_i = compute_reduced_distance(match_i)
        distances_3d.append(dist_i)

        match_i2 = match_ghosts(ref_data_frame, df_i, radius_scale_factor=r_scale)
        dist_i2 = compute_2d_reduced_distance(match_i2)
        distances_2d.append(dist_i2)

        print(f'{delta:.6f} ', end='', flush=True)

    return shifts_list, distances_2d, distances_3d


def simulate_impact_points_for_beam_set(telescope, beam_set):
    """ Runs a ray tracing simulation of a light beam into the CCOB for a list
    of beam configurations

    Parameters
    ----------
    telescope : `batoid.telescope`
        the optical setup
    beam_set : `list` of `dict`
        a list of dictionaries with the light beam configuration, see :ref:`beam_configs`.

    Returns
    -------
    data_frame : `pandas.DataFrame`
        a panda data frame with information on beam positions and main impact points
    -------
    """
    # initialize lists
    impact_x = []
    impact_y = []
    impact_id = []

    # run simulation and get impact point
    for one_beam in beam_set:
        print('Simulating beam id ', one_beam['beam_id'], end='\r')
        _, forward_rays, _, _ = run_simulation(telescope, one_beam)
        _, x, y, _ = get_main_impact_point(forward_rays)
        impact_x.append(x)
        impact_y.append(y)
        impact_id.append(one_beam['beam_id'])

    # now make a data frame
    impact_df = pd.DataFrame({'beam_id': impact_id, 'x_spot': impact_x, 'y_spot': impact_y})

    # make a data frame of the beam set
    beam_set_df = concat_dicts(beam_set)

    # join tables
    data_frame = beam_set_df.join(impact_df.set_index('beam_id'), on='beam_id')

    # compute additional stuff
    data_frame['beam_dist_to_center'] = np.sqrt(
        data_frame['x_offset'] * data_frame['x_offset'] + data_frame['y_offset'] * data_frame['y_offset'])
    data_frame['spot_dist_to_center'] = np.sqrt(
        data_frame['x_spot'] * data_frame['x_spot'] + data_frame['y_spot'] * data_frame['y_spot'])
    data_frame['convergence'] = data_frame['spot_dist_to_center'] / data_frame['beam_dist_to_center']
    data_frame['displacement'] = data_frame['spot_dist_to_center'] - data_frame['beam_dist_to_center']
    data_frame.replace([np.inf, -np.inf], np.nan, inplace=True)

    # done
    return data_frame
