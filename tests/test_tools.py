""" Unit test for the tools module
"""
import unittest
import numpy as np

from ghosts.tools import get_ranges

class ToolsTestCase(unittest.TestCase):
    """ Test class for the ghosts.tools module"""
    def test_get_ranges(self):
        """ Verify we get the correct xmin, xmax, ymin, ymax"""
        x = np.array([1, 2, 3])
        y = np.array([4, 5, 6])
        my_range = get_ranges(x, y, dr=1)
        self.assertEqual(my_range, (1, 3, 4, 6))  # add assertion here


if __name__ == '__main__':
    unittest.main()
