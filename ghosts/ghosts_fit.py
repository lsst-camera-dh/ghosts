"""ghosts_fit module

This module provides classes and functions to fit the camera geometry using ghosts images catalogs.

"""

import logging
import numpy as np
import pandas as pd
import concurrent.futures
from iminuit import Minuit

from ghosts import simulator, tweak_optics, tools, beam

from ghosts.beam_configs import BASE_BEAM_SET
from ghosts.geom_configs import GEOM_LABELS_15
from ghosts.analysis import match_ghosts, compute_2d_reduced_distance


class GhostsFitter(object):
    """ Class to handle the fitting procedure

    Needed because some functions have to share some data

    Attributes
    ----------
    ref_yaml_geom : `string`
        path to the yaml file with the geometry configuration
    ref_beam_set : `list[dict]`
        the list of beam configurations used to generate the reference ghosts catalogs
    ref_telescope : `batoid.telescope`
        the reference optical setup as defined in `batoid`
    spots_df_list : `list[pandas.DataFrame]`
        a `list` of panda data frame with ghost spot data information, including beam and geometry configuration ids
    fit_beam_set : `list[dict]`
        the beam configuration to be used during the fit simulations
    minuit : `iminuit.Minuit`
        the Minuit object at the end of the fitting procedure
    """
    def __init__(self, yaml_geom="./data/LSST_CCOB_r_aligned.yaml", beam_set=BASE_BEAM_SET):
        """ Constructor, builds an object so that it holds the reference telescope, beam and ghosts catalog
        Parameters
        ----------
            yaml_geom : `string`
                path to the yaml file with the geometry configuration
            beam_set : `list[dict]`
                the list of beam configurations used to generate the reference ghosts catalogs
        """
        self.ref_yaml_geom = yaml_geom
        self.ref_beam_set = beam_set
        self.ref_telescope = None
        self.spots_df_list = None
        self.fit_beam_set = None
        self.minuit = None

    def setup(self):
        """ Setup fitter by generating the reference telescope, running the simulation for all the beams
        in the beam set and producing the output spots data frame.

        Parameters
        ----------

        Returns
        -------
        """
        # reference telescope
        self.ref_telescope = tweak_optics.build_telescope_at_600nm(self.ref_yaml_geom)
        # run simulations to get the list of reference catalogs for all beams
        self.spots_df_list = simulator.run_and_analyze_simulation_for_telescope_and_beam_set(
                                    self.ref_telescope, geom_id=0, beam_set=self.ref_beam_set)
        # build beams to use for the fit
        self.fit_beam_set = beam.set_n_photons_on_beam_set(self.ref_beam_set, n_photons=1000)

    def build_telescope_to_fit(self, geom_params):
        """ Build telescope to fit from reference telescope

        Parameters
        ----------
        geom_params : `list`
            a dictionary with the geometry of the telescope to fit

        Returns
        -------
        fitted_telescope : `batoid.telescope`
            the telescope to be used for the ray tracing simulation called for the ghosts fitting procedure
        """
        # Build telescope
        fitted_geom_config = tools.unpack_geom_params(geom_params, GEOM_LABELS_15)
        fitted_telescope = tweak_optics.tweak_telescope(self.ref_telescope, fitted_geom_config)
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
        fitted_telescope = self.build_telescope_to_fit(geom_params)
        # simulate the fit geometry through the full set of beams in the beam set
        # as the function is multithread, the list is not ordered
        fit_spots_df_list = simulator.run_and_analyze_simulation_for_telescope_and_beam_set(
                                                        fitted_telescope, geom_id=0, beam_set=self.fit_beam_set)
        # rebuild ordered list for futures
        merged_fit_df = pd.concat(fit_spots_df_list)
        ordered_fit_df = []
        for df in self.spots_df_list:
            ordered_fit_df.append(merged_fit_df.loc[merged_fit_df['beam_id'] == df['beam_id'][0]])
        # now compute distance
        dist_list = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(self.match_and_dist, ref_df, fit_df)
                       for ref_df, fit_df in zip(self.spots_df_list, ordered_fit_df)]
            for future in concurrent.futures.as_completed(futures):
                dist_list.append(future.result())

        # compute Uber distance
        uber_dist = np.sqrt(np.square(dist_list).sum())/len(dist_list)

        # Verification of what's happening
        fitted_geom_config = tools.unpack_geom_params(geom_params, GEOM_LABELS_15)
        # Minuit can actually take a callback function
        if not np.random.randint(10) % 9:
            msg = f'{uber_dist:.6f} '
            for lab in GEOM_LABELS_15:
                msg += f'{fitted_geom_config[lab]:.6f} '
            logging.debug(msg)
        return uber_dist

    @staticmethod
    def match_and_dist(ref_df, fit_df):
        match = match_ghosts(ref_df, fit_df, radius_scale_factor=10)
        dist_2d = compute_2d_reduced_distance(match)
        return dist_2d

    def run(self, n_calls=50, precision=1e-6, with_cov=True):
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
        logging.info(f'Fitting for {n_calls} calls to get a precision of {precision}')
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
        logging.info(f'Is covariance matrix valid and accurate ? -> {m.valid} / {m.accurate}')

        # attach minuit object to GhostsFitter object
        self.minuit = m
        # run covariance estimator if possible and requested
        if m.valid and with_cov:
            self.minuit.hesse()
            self.minuit.minos()
        return self.minuit


if __name__ == '__main__':
    # "Agg" for batch mode or try "GTK4Agg" on wayland
    #  need to install pygobject gtk4
    import matplotlib as mpl
    mpl.use('Agg')

    # Set logging level
    logging.basicConfig(level=logging.DEBUG)

    # test ghost_fit class
    fitter = GhostsFitter()
    fitter.setup()
    fitter.run(n_calls=1000, precision=1e-7)

    # Log results
    logging.info(fitter.minuit.values)
    logging.info(fitter.minuit.errors)

    # Matrix
    fig, ax = fitter.minuit.draw_mnmatrix(cl=[1, 2, 3])
    fig.savefig('matrix.png')
