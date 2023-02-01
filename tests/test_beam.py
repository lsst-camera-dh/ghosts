""" Unit test for the beam module
"""
import unittest

from ghosts import beam
from ghosts.beam_configs import BEAM_CONFIG_3


class BeamTestCase(unittest.TestCase):
    """ Test class for the ghosts.beam module"""
    def test_point_beam_to_target(self):
        """ Verify that the geom is able to point the beam to a target correctly,
        this actually tests various functions
        """
        new_beam = beam.point_beam_to_target(BEAM_CONFIG_3, target_x=0.15, target_y=-0.05)
        self.assertAlmostEqual(new_beam['x_euler'], 15.8582, delta=1e-3)
        self.assertAlmostEqual(new_beam['y_euler'], 11.4700, delta=1e-3)


if __name__ == '__main__':
    unittest.main()
