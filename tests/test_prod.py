""" Unit test for the code efficiency
"""
import unittest

import os
from scripts import run_production


class ProdTestCase(unittest.TestCase):
    """ Test class for the ghosts module"""
    def test_prod(self):
        """ Verify that a small production can be run and save to disk
        """
        run_production.run(4, 'test')
        self.assertTrue(os.path.exists('test.parquet'))


if __name__ == '__main__':
    unittest.main()
