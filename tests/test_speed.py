""" Unit test for the code efficiency
"""
import unittest
import timeit

import batoid
from ghosts import tweak_optics


class SpeedTestCase(unittest.TestCase):
    """ Test class for the ghosts module"""
    def test_build_geom_from_yaml(self):
        """ Verify that the building telescope from yaml is not too slow
        """
        start_time = timeit.default_timer()
        tels = list()
        for i in range(100):
            telescope = batoid.Optic.fromYaml("./data/LSST_CCOB_r.yaml")
            tweak_optics.make_optics_reflective(telescope, coating='smart', r_frac=[0.02, 0.02, 0.15])
            tels.append(telescope)
        stop_time = timeit.default_timer()
        time_diff = stop_time - start_time
        print(f'The time difference is :{time_diff:.3f} s')
        self.assertLess(time_diff, 5.)

    def test_randomize_telescope(self):
        """ Verify that building randomized telescope is not too slow
        """
        start_time = timeit.default_timer()
        # build one telescope and tweak it around
        telescope = batoid.Optic.fromYaml("./data/LSST_CCOB_r.yaml")
        tels = list()
        for i in range(100):
            rnd_tel = tweak_optics.randomized_telescope(telescope, max_angle=0.1, max_shift=0.001, verbose=False)
            tels.append(rnd_tel)
        stop_time = timeit.default_timer()
        time_diff = stop_time - start_time
        print(f'The time difference is :{time_diff:.3f} s')
        self.assertLess(time_diff, 0.5)


if __name__ == '__main__':
    unittest.main()
