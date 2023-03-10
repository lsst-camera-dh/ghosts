""" A very simple script to run a fit of the geometry
"""
import logging
import copy
import numpy as np
from iminuit import Minuit

import batoid
import ghosts.simulator as simulator
import ghosts.tweak_optics as tweak_optics

from ghosts.beam_configs import BEAM_CONFIG_0
from ghosts.geom_configs import GEOM_LABELS_15
from ghosts.analysis import reduce_ghosts, match_ghosts, compute_2d_reduced_distance
from ghosts.analysis import make_data_frame


def build_ref_beam(base_config=BEAM_CONFIG_0):
    """ Build the reference telescope from a yaml geometry.

    Parameters
    ----------
    base_config : `dict`
        the base beam configuration to tweak

    Returns
    -------
    ref_beam : `dict`
        the reference beam configuration to be used for thed fit
    """
    # Cool ghost configuration
    ref_beam = copy.deepcopy(base_config)
    ref_beam['beam_id'] = 999042
    ref_beam['n_photons'] = 10000
    ref_beam['x_offset'] = +0.3
    ref_beam['y_offset'] = -0.55
    ref_beam['x_euler'] = -17
    ref_beam['y_euler'] = -6
    return ref_beam


def build_ref_telescope(yaml_geom="./data/LSST_CCOB_r_aligned.yaml"):
    """ Build the reference telescope from a yaml geometry.

    Parameters
    ----------
    yaml_geom : `string`
        path to the yaml file with the reference geometry to be used.

    Returns
    -------
    ref_telescope : `batoid.telescope`
        the reference optical setup as defined in `batoid`
    """
    # A few numbers, specific to 600 nm
    ccd_reflectivity_600nm = 0.141338
    lens_reflectivity_600nm = 0.004  # 0.4% code by Julien Bolmont
    filter_reflectivity_600nm = 0.038  # r band filter documentation stated transmission is 96.2%

    # CCOB like geometry, i.e. lenses but no filter
    ref_telescope = batoid.Optic.fromYaml(yaml_geom)

    # Make refractive interfaces partially reflective
    # Call on current telescope, smart coating is [lens, filter, camera]
    tweak_optics.make_optics_reflective(ref_telescope, coating='smart',
                                        r_frac=[lens_reflectivity_600nm, filter_reflectivity_600nm,
                                                ccd_reflectivity_600nm])
    return ref_telescope


def build_ref_ghosts_catalog(telescope, beam):
    """ Run a simulation with the reference telescope and the reference beam configuration in order to produce
    the reference ghost catalog.

    Parameters
    ----------
    telescope : `batoid.telescope`
        the reference optical setup as defined in `batoid`
    beam : `dict`
        the reference beam configuration

    Returns
    -------
    ref_spots_data_frame : `pandas.DataFrame`
        a panda data frame with ghost spot data information, including beam and geometry configuration ids
    """
    # Ray trace one config for debugging
    trace_full, r_forward, r_reverse, rays = simulator.run_simulation(telescope, beam_config=beam)

    # reduce ghosts
    ref_spots_data, _spots = reduce_ghosts(r_forward)
    ref_spots_data_frame = make_data_frame(ref_spots_data, beam_id=beam['beam_id'], geom_id=0)
    return ref_spots_data_frame


def get_beam_for_fit(ref_beam, n_photons=1000):
    """ Build a beam configuration matching the reference one, but with fewer photons so that the simulations
    called by the fit are faster

    Parameters
    ----------
    ref_beam : `dict`
        the reference beam configuration
    n_photons : `int`
        the number of rays to simulate for the new configuration

    Returns
    -------
    fit_beam : `dict`
        the beam configuration to be used during the fit simulations
    """
    # make a copy with fewer photons for the fit
    fit_beam = copy.deepcopy(ref_beam)
    fit_beam['n_photons'] = n_photons
    return fit_beam


def unpack_geom_params(geom_params, geom_labels=GEOM_LABELS_15):
    """ Convert a list of geometry parameters into a dictionary as a telescope geometry configuration

    Parameters
    ----------
    geom_params : `list`
        an ordered list of parameters corresponding to a geometry configuration
    geom_labels : `list`
        a list of the geometry parameter labels (names) matching the list above

    Returns
    -------
    fitted_geom_config : `dict`
        a dictionary with the geometry of the telescope to fit
    """

    fitted_geom_config = {}
    for i, lab in enumerate(geom_labels):
        fitted_geom_config[lab]=geom_params[i]
    return fitted_geom_config


def build_telescope_to_fit(ref_telescope, geom_params):
    """ Build telescope to fit from reference telescope

    Parameters
    ----------
    ref_telescope : `batoid.telescope`
        the reference optical setup as defined in `batoid`
    geom_params : `list`
        a list with the geometry of the telescope to fit

    Returns
    -------
    fitted_telescope : `batoid.telescope`
        the telescope to be used for the ray tracing simulation called for the ghosts fitting procedure
    """

    # Build telescope
    fitted_geom_config = unpack_geom_params(geom_params)
    fitted_telescope = tweak_optics.tweak_telescope(ref_telescope, fitted_geom_config)
    # Make refractive interfaces partially reflective
    ccd_reflectivity_600nm = 0.141338
    lens_reflectivity_600nm = 0.004  # 0.4% code by Julien Bolmont
    filter_reflectivity_600nm = 0.038  # r band filter documentation stated transmission is 96.2%
    tweak_optics.make_optics_reflective(fitted_telescope, coating='smart',
                                        r_frac=[lens_reflectivity_600nm, filter_reflectivity_600nm, ccd_reflectivity_600nm])
    return fitted_telescope


# prepare reference catalog
GLOBAL_REF_TELESCOPE = build_ref_telescope(yaml_geom="./data/LSST_CCOB_r_aligned.yaml")
GLOBAL_REF_BEAM = build_ref_beam(base_config=BEAM_CONFIG_0)
GLOBAL_SPOTS_DF = build_ref_ghosts_catalog(GLOBAL_REF_TELESCOPE, GLOBAL_REF_BEAM)
GLOBAL_FIT_BEAM = get_beam_for_fit(GLOBAL_REF_BEAM, n_photons=1000)


def compute_distance_for_fit(geom_params_array):
    """ Callable function for the fit

    Parameters
    ----------
    geom_params_array : `np.array`
        a numpy array of `double`
    Returns
    -------
    dist_2d : `double`
        the distance between the two catalogs of ghosts, to be minimized by the fitting procedure
    """
    # reference objects
    geom_params=geom_params_array.tolist()
    # new telescope
    fitted_telescope = build_telescope_to_fit(GLOBAL_REF_TELESCOPE, geom_params)
    fit_spots_df = simulator.run_and_analyze_simulation(fitted_telescope, geom_id=0, beam_config=GLOBAL_FIT_BEAM)
    # save spots figure
    #save_spot_fig(fit_spots_df)
    # match ghosts
    match = match_ghosts(GLOBAL_SPOTS_DF, fit_spots_df, radius_scale_factor=10)
    dist_2d = compute_2d_reduced_distance(match)
    fitted_geom_config = unpack_geom_params(geom_params)
    # Minuit can actually take a callback function
    if not np.random.randint(10)%9:
        msg = f'{dist_2d:.6f} {fitted_geom_config["L1_dx"]:.6f} {fitted_geom_config["L1_dy"]:.6f} {fitted_geom_config["L1_dz"]:.6f} {fitted_geom_config["L1_rx"]:.6f} {fitted_geom_config["L1_ry"]:.6f} '
        msg += f'{dist_2d:.6f} {fitted_geom_config["L2_dx"]:.6f} {fitted_geom_config["L2_dy"]:.6f} {fitted_geom_config["L2_dz"]:.6f} {fitted_geom_config["L2_rx"]:.6f} {fitted_geom_config["L2_ry"]:.6f} '
        msg += f'{dist_2d:.6f} {fitted_geom_config["L3_dx"]:.6f} {fitted_geom_config["L3_dy"]:.6f} {fitted_geom_config["L3_dz"]:.6f} {fitted_geom_config["L3_rx"]:.6f} {fitted_geom_config["L3_ry"]:.6f}'
        logging.debug(msg)
    # clean up
    del fitted_telescope
    return dist_2d


def run(n_calls=50, precision=1e-6):
    """ Run the fit

    Parameters
    ----------
    n_calls : `int`
        maximum number of calls to the fit callable allowed for the fit
    precision : `double`
        target precision of the fitted parameters

    Returns
    -------
    m : `Minuit`
        the Minuit object at the end of the fitting procedure
    """
    logging.info(f'Fitting for {args.n_calls} calls to get a precision of {args.precision}')
    # init
    geom_params_init = np.array([0.0] * 15)
    # bounds
    dxs = [(-0.0005, 0.0005)] * 3
    rxs = [(-0.01, 0.01)] * 2
    list_of_bounds = (dxs + rxs) + (dxs + rxs) + (dxs + rxs)

    # Minuit
    m = Minuit(compute_distance_for_fit, geom_params_init, name=tuple(GEOM_LABELS_15))
    m.limits = list_of_bounds
    m.precision = precision
    logging.info(f'\n{m.params}')

    m.migrad(ncall=n_calls, iterate=5)  # run optimiser
    m.hesse()  # run covariance estimator
    return m


if __name__ == '__main__':
    ''' This script runs a basic fit with Minuit over an aligned geometry and just one beam configuration
    
    Comments
    --------
    Typical displacement values (L3)
    x: 2.73175526e-05 m 
    y: -8.83090766e-05 m
    rotX: 0.00011657042290169468 rad
    rotY: 3.6059894624233535e-05 rad
    '''
    # Get args and parse these
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("n_calls")
    parser.add_argument("precision")
    args = parser.parse_args()

    # Set logging level
    logging.basicConfig(level=logging.DEBUG)

    # Run!
    minuit = run(int(args.n_calls), float(args.precision))

    # Log results
    logging.info(minuit.values)
    logging.info(minuit.errors)
