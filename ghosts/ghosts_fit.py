"""ghosts_fit module

This module provides a class to fit the camera geometry using ghosts images catalogs.

"""

import logging
import numpy as np
import pandas as pd
from iminuit import Minuit

from ghosts import simulator, tweak_optics, tools, beam

from ghosts.beam_configs import BASE_BEAM_SET
from ghosts.geom_configs import GEOM_LABELS_15
from ghosts.analysis import compute_uber_distance_2d


class GhostsFitter:
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

    def setup_reference_ghosts(self):
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
        # as the function uses futures, the list is not ordered
        fit_spots_df_list = simulator.run_and_analyze_simulation_for_telescope_and_beam_set(
                                                        fitted_telescope, geom_id=0, beam_set=self.fit_beam_set)
        # rebuild ordered list for futures in `compute_uber_distance_2d`
        merged_fit_df = pd.concat(fit_spots_df_list)
        ordered_fit_df = []
        for df in self.spots_df_list:
            ordered_fit_df.append(merged_fit_df.loc[merged_fit_df['beam_id'] == df['beam_id'][0]])

        # compute distance between the 2 list of ghosts catalogs
        uber_dist = compute_uber_distance_2d(self.spots_df_list, ordered_fit_df)

        # Log debug info - Minuit can actually take a callback function
        if not np.random.randint(10) % 9:
            # unpack parameters
            fitted_geom_config = tools.unpack_geom_params(geom_params, GEOM_LABELS_15)
            # build message
            msg = f'{uber_dist:.6f} '
            for lab in GEOM_LABELS_15:
                msg += f'{fitted_geom_config[lab]:.6f} '
            logging.debug(msg)
        return uber_dist

    def free_optics(self, ref_list=GEOM_LABELS_15):
        """ Free parameters of all elements in the list

        Parameters
        ----------
        ref_list : `list[str]`
            list of geometrical parameters ("L1_dx", "L1_dy", ...)

        Returns
        -------
        None
        """
        for key in ref_list:
            self.minuit.fixed[key] = False

    def fix_optic(self, optic, ref_list=GEOM_LABELS_15):
        """ Fix all parameters of the given optical element

        Parameters
        ----------
        optic : `str`
            optical element of the geometry, e.g. "L1"
        ref_list : `list[str]`
            list of geometrical parameters ("L1_dx", "L1_dy", ...)

        Returns
        -------
        None
        """
        for key in ref_list:
            if key.find(optic) >= 0:
                self.minuit.fixed[key] = False
            else:
                self.minuit.fixed[key] = True

    def setup_minuit(self, precision):
        """ Setup Minuit and attach it to the GhostFitter object

        Includes parameters initial values and limits
        
        Parameters
        ----------
        precision : `double`
            target precision of the fitted parameters

        Returns
        -------
        None
        """
        # init
        geom_params_init = np.array([0.0] * 15)
        # bounds
        dxs = [(-0.0005, 0.0005)] * 3
        rxs = [(-0.01, 0.01)] * 2
        list_of_bounds = (dxs + rxs) + (dxs + rxs) + (dxs + rxs)

        # Setup Minuit
        m = Minuit(self.compute_distance_for_fit, geom_params_init, name=tuple(GEOM_LABELS_15))
        m.limits = list_of_bounds
        m.precision = precision
        logging.info('Minuit now setup\n%s', m.params)
        # attach minuit object to GhostsFitter object
        self.minuit = m

    def run_fit_everything(self, n_calls=50, with_cov=True):
        """ Run the fit, all parameters free

        Parameters
        ----------
        n_calls : `int`
            maximum number of calls to the fit callable allowed for the fit
        with_cov : `bool`
            True if you wish to compute the HESSE asymptotic errors and MINOS confidence intervals,
            will be done only if the function minimum is valid and the covariance matrix accurate

        Returns
        -------
        None
        """
        # make sure all parameters are free
        self.free_optics()
        # run optimiser
        self.minuit.migrad(ncall=n_calls, iterate=5)
        logging.info('Is covariance matrix valid and accurate ? -> %s / %s', self.minuit.valid, self.minuit.accurate)
        # run covariance estimator if possible and requested
        if self.minuit.valid and with_cov:
            self.minuit.hesse()
            self.minuit.minos()

    def run_iterative_fit_per_element(self, optics_list, n_sub_calls=50):
        """ Run the fit, iteratively on each optical element

        Parameters
        ----------
        n_sub_calls : `int`
            maximum number of calls for the fit on a single optical element
        optics_list : `list[str]`
            list of optics to iterate over for the fit

        Returns
        -------
        None
        """
        # Set default here
        if optics_list is None:
            optics_list = ['L1', 'L2', 'L3']
        # Iterative fitting
        for optic in optics_list:
            self.fix_optic(optic)
            logging.info('Now fitting on %s', optic)
            self.minuit.migrad(ncall=n_sub_calls, iterate=5)  # run optimiser
            logging.info('Is covariance matrix valid and accurate ? -> %s / %s',
                         self.minuit.valid, self.minuit.accurate)
            logging.info(self.minuit.values)

    def run(self, mode="standard", n_calls=50, precision=1e-6, with_cov=True, optics_list=None, n_sub_calls=50):
        """ Run the fit

        Parameters
        ----------
        mode : `string`
            fitting mode as "standard", "iterative", "combined"
        n_calls : `int`
            maximum number of calls to the fit callable allowed for the fit
        precision : `double`
            target precision of the fitted parameters
        with_cov : `bool`
            True if you wish to compute the HESSE asymptotic errors and MINOS confidence intervals,
            will be done only if the function minimum is valid and the covariance matrix accurate
        optics_list : `list[str]`
            list of optics to iterate over for the fit, for the "iterative" mode
        n_sub_calls : `int`
            maximum number of calls for the fit on a single optical element, for the "iterative" mode

        Returns
        -------
        valid : `bool`
            True if the function minimum is valid
        """
        # setup Minuit
        self.setup_minuit(precision=precision)
        logging.info('Fitting for %d calls to get a precision of %.1e in %s mode.',
                     n_calls, precision, mode)
        # Mode
        match mode:
            case 'standard':
                self.run_fit_everything(n_calls=n_calls, with_cov=with_cov)
            case 'iterative':
                self.run_iterative_fit_per_element(n_sub_calls=n_sub_calls, optics_list=optics_list)
            case 'combined':
                self.run_iterative_fit_per_element(n_sub_calls=n_sub_calls, optics_list=optics_list)
                self.run_fit_everything(n_calls=n_calls, with_cov=with_cov)
            case _:  # default
                self.run_fit_everything(n_calls=n_calls, with_cov=with_cov)

        # Return if the function minimum is valid
        return self.minuit.valid


if __name__ == '__main__':
    # "Agg" for batch mode or try "GTK4Agg" on wayland
    #  need to install pygobject gtk4
    import matplotlib as mpl
    mpl.use('Agg')

    # Set logging level
    logging.basicConfig(level=logging.DEBUG)

    # test ghost_fit class
    fitter = GhostsFitter()
    fitter.setup_reference_ghosts()
    fitter.run()
    # fitter.run(n_calls=10, precision=1e-5)
    # fitter.run(mode="standard", n_calls=200, precision=1e-6, with_cov=True, n_sub_calls=50)

    # Log results
    logging.info(fitter.minuit.values)
    logging.info(fitter.minuit.errors)
