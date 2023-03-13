""" A simple script to run a fit of the geometry, using an OO implementation
"""
import logging
import copy
import numpy as np
import pickle
from iminuit import Minuit


import batoid
import ghosts.simulator as simulator
import ghosts.tweak_optics as tweak_optics
import ghosts.tools as tools

from ghosts.beam_configs import BEAM_CONFIG_0
from ghosts.geom_configs import GEOM_LABELS_15
from ghosts.analysis import match_ghosts
from ghosts.analysis import compute_2d_reduced_distance


class SimpleGhostsFitter(object):
    """ Class to handle the fitting procedure

    Needed because some functions have to share some data

    Attributes
    ----------
    ref_telescope : `batoid.telescope`
        the reference optical setup as defined in `batoid`
    ref_beam : `dict`
        the reference beam configuration to be used for thed fit
    spots_df : `pandas.DataFrame`
        a panda data frame with ghost spot data information, including beam and geometry configuration ids
    fit_beam : `dict`
        the beam configuration to be used during the fit simulations
    minuit : `iminuit.Minuit`
        the Minuit object at the end of the fitting procedure
    """
    def __init__(self):
        """ Constructor, builds an object so that it holds the reference telescope, beam and ghosts catalog
        """
        # reference telescope
        self.ref_telescope = self.build_ref_telescope(yaml_geom="./data/LSST_CCOB_r_aligned.yaml")
        # reference beam
        self.ref_beam = self.build_ref_beam(base_config=BEAM_CONFIG_0)
        # reference catalog
        self.spots_df = simulator.run_and_analyze_simulation(self.ref_telescope, geom_id=0,
                                                             beam_config=self.ref_beam)
        # beam to use for the fit
        self.fit_beam = self.get_beam_for_fit(self.ref_beam, n_photons=1000)
        # Minuit object
        self.minuit = None

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def build_telescope_to_fit(ref_telescope, geom_params):
        """ Build telescope to fit from reference telescope

        Parameters
        ----------
        ref_telescope : `batoid.telescope`
            the reference optical setup as defined in `batoid`
        geom_params : `list`
            a dictionary with the geometry of the telescope to fit

        Returns
        -------
        fitted_telescope : `batoid.telescope`
            the telescope to be used for the ray tracing simulation called for the ghosts fitting procedure
        """

        # Build telescope
        fitted_geom_config = tools.unpack_geom_params(geom_params, GEOM_LABELS_15)
        fitted_telescope = tweak_optics.tweak_telescope(ref_telescope, fitted_geom_config)
        # Make refractive interfaces partially reflective
        ccd_reflectivity_600nm = 0.141338
        lens_reflectivity_600nm = 0.004  # 0.4% code by Julien Bolmont
        filter_reflectivity_600nm = 0.038  # r band filter documentation stated transmission is 96.2%
        tweak_optics.make_optics_reflective(fitted_telescope, coating='smart',
                                            r_frac=[lens_reflectivity_600nm, filter_reflectivity_600nm,
                                                    ccd_reflectivity_600nm])
        return fitted_telescope

    def compute_distance_for_fit(self, geom_params_array):
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
        geom_params = geom_params_array.tolist()
        # new telescope
        fitted_telescope = self.build_telescope_to_fit(self.ref_telescope, geom_params)
        fit_spots_df = simulator.run_and_analyze_simulation(fitted_telescope, geom_id=0, beam_config=self.fit_beam)
        # match ghosts
        match = match_ghosts(self.spots_df, fit_spots_df, radius_scale_factor=10)
        dist_2d = compute_2d_reduced_distance(match)
        fitted_geom_config = tools.unpack_geom_params(geom_params, GEOM_LABELS_15)
        # Minuit can actually take a callback function
        if not np.random.randint(10) % 9:
            msg = f'{dist_2d:.6f} {fitted_geom_config["L1_dx"]:.6f} {fitted_geom_config["L1_dy"]:.6f} {fitted_geom_config["L1_dz"]:.6f} {fitted_geom_config["L1_rx"]:.6f} {fitted_geom_config["L1_ry"]:.6f} '
            msg += f'{dist_2d:.6f} {fitted_geom_config["L2_dx"]:.6f} {fitted_geom_config["L2_dy"]:.6f} {fitted_geom_config["L2_dz"]:.6f} {fitted_geom_config["L2_rx"]:.6f} {fitted_geom_config["L2_ry"]:.6f} '
            msg += f'{dist_2d:.6f} {fitted_geom_config["L3_dx"]:.6f} {fitted_geom_config["L3_dy"]:.6f} {fitted_geom_config["L3_dz"]:.6f} {fitted_geom_config["L3_rx"]:.6f} {fitted_geom_config["L3_ry"]:.6f}'
            logging.debug(msg)
        # clean up
        del fitted_telescope
        return dist_2d

    def run(self, n_calls=50, precision=1e-6):
        """ Run the fit

        Parameters
        ----------
        n_calls : `int`
            maximum number of calls to the fit callable allowed for the fit
        precision : `double`
            target precision of the fitted parameters

        Returns
        -------
        m : `iminuit.Minuit`
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
        m = Minuit(self.compute_distance_for_fit, geom_params_init, name=tuple(GEOM_LABELS_15))
        m.limits = list_of_bounds
        m.precision = precision
        logging.info(f'\n{m.params}')

        m.migrad(ncall=n_calls, iterate=5)  # run optimiser
        m.hesse()  # run covariance estimator
        # attach minuit object to GhostsFitter object
        self.minuit = m
        return self.minuit


if __name__ == '__main__':
    ''' This script runs a basic fit with Minuit over an aligned geometry and just one beam configuration.
    
    The implementation uses a class `SimpleGhostsFitter` for the code to be clean and obvious to understand.
    
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
    fit = SimpleGhostsFitter()
    minuit = fit.run(int(args.n_calls), float(args.precision))

    # Log results
    logging.info(minuit.values)
    logging.info(minuit.errors)

    # Save fitter to disk with pickle
    with open('fit.pickle', 'wb') as f:
        pickle.dump(fit, f)

    # To reload the fitter, use
    # from scripts.run_fit_example_oo import SimpleGhostsFitter
    # >>> with open('fit.pickle', 'rb') as f:
    # ...     fit = pickle.load(f)
    # >>> fit
    # <scripts.run_fit_example_oo.SimpleGhostsFitter object at 0x7f61eff5d950>


