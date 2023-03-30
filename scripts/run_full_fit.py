""" A script to run a full fit of the geometry, using the GhostFitter class
"""
import logging
from ghosts.ghosts_fit import GhostsFitter
from ghosts.beam_configs import BASE_BEAM_SET


if __name__ == '__main__':
    ''' This script runs a full fit calling the GhostFitter class
        
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
    parser.add_argument("mode")
    parser.add_argument("n_calls")
    parser.add_argument("precision")
    args = parser.parse_args()

    # Create Matplotlib GUI here
    import matplotlib as mpl
    mpl.use('qtagg')

    # Set logging level
    logging.basicConfig(level=logging.DEBUG)

    # Run!
    fitter = GhostsFitter(yaml_geom="./data/LSST_CCOB_r_aligned.yaml", beam_set=BASE_BEAM_SET)
    fitter.setup_reference_ghosts()
    fitter.run(mode=args.mode, n_calls=int(args.n_calls), precision=float(args.precision),
               with_cov=True, optics_list=['L1', 'L2', 'L3'], n_sub_calls=50)

    # Log results
    logging.info(fitter.minuit.values)
    logging.info(fitter.minuit.errors)

    # Save fitter to disk with pickle
    # with open('fit.pickle', 'wb') as f:
    #     pickle.dump(fit, f)

    # To reload the fitter, use
    # from scripts.run_fit_example_oo import SimpleGhostsFitter
    # >>> with open('fit.pickle', 'rb') as f:
    # ...     fit = pickle.load(f)
    # >>> fit
    # <scripts.run_fit_example_oo.SimpleGhostsFitter object at 0x7f61eff5d950>
