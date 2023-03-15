""" Unit test for the whole fit procedure
"""
import unittest

import os
import pickle
from ghosts.ghosts_fit import GhostsFitter


class FitTestCase(unittest.TestCase):
    """ Test class for the ghosts module"""
    def test_fit(self):
        """ Verify that the fitting procedure can be run and save to disk
        """
        # "Agg" for batch mode or try "GTK4Agg" on wayland
        import matplotlib as mpl
        mpl.use('Agg')
        # test ghost_fit class
        fitter = GhostsFitter()
        fitter.setup()
        fitter.run(n_calls=5, precision=1e-4, with_cov=False)
        with open('fit.pickle', 'wb') as f:
            pickle.dump(fitter, f)
        self.assertTrue(os.path.exists('fit.pickle'))


if __name__ == '__main__':
    unittest.main()
