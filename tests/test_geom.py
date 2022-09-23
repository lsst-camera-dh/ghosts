""" Unit test for the geom module
"""
import unittest
import batoid

from ghosts import tweak_optics
from ghosts.constants import CCOB_DISTANCE_TO_FOCAL_PLANE


class GeomTestCase(unittest.TestCase):
    """ Test class for the ghosts.geom module"""
    def test_CCOB_default_geometry(self):
        """ Verify that the default geometry is a CCOB"""
        telescope = batoid.Optic.fromYaml("./data/LSST_CCOB_r.yaml")
        self.assertEqual(tweak_optics.get_list_of_optics(telescope),
                         ['L1', 'L2', 'L3', 'Detector'],
                         'Not a CCOB optical setup')

    def test_CCOB_geometry_with_filter(self):
        """ Verify that the default geometry is a CCOB"""
        telescope = batoid.Optic.fromYaml("./data/LSST_CCOB_r_with_filter.yaml")
        self.assertEqual(tweak_optics.get_list_of_optics(telescope),
                         ['L1', 'L2', 'Filter', 'L3', 'Detector'],
                         'Not a CCOB optical setup')

    def test_distance_to_focal_plane(self):
        """ Verify that the CCOB distance to the focal plane is coherent between
        the default geometry and the value stored in the constants module
        """
        # Second CCOB like geometry, i.e. lenses + filters
        telescope = batoid.Optic.fromYaml("./data/LSST_CCOB_r.yaml")
        distance = tweak_optics.get_optics_position_z(telescope, 'Detector') - \
            tweak_optics.get_optics_position_z(telescope, 'M1CaBaffle2')
        self.assertAlmostEqual(CCOB_DISTANCE_TO_FOCAL_PLANE, distance, delta=1e-6)


if __name__ == '__main__':
    unittest.main()
