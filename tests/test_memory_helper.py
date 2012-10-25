#!/usr/bin/env python

import unittest
import doctest

import numpy

from cogbox.memory import memory_helper 


class Test(unittest.TestCase):
    """Unit tests for the Sparse Distributed Memory"""


    def test_doctests(self):
        """Run sdm doctests"""
        doctest.testmod(memory_helper)

if __name__ == "__main__":
    unittest.main()
