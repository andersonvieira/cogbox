#!/usr/bin/env python

import unittest
import doctest

import numpy

from cogbox.memory import hopfield, helper 


class Test(unittest.TestCase):
    """Unit tests for the Sparse Distributed Memory"""

    def test_doctests(self):
        """Run sdm doctests"""
        doctest.testmod(hopfield)

    def test_retrieve(self):
        """
        Test if a pattern is retrieved correctly after being stored
        """
        memory = hopfield.HopfieldMemory()
        memory.store(numpy.array([helper.EXAMPLE_PATTERNS['X']]))
        self.assertTrue(numpy.array_equal(
            memory.retrieve(helper.EXAMPLE_PATTERNS['X']), 
            helper.EXAMPLE_PATTERNS['X']))

    def test_double_retrieve(self):
        """
        Test if two patterns are retrieved correctly after being stored
        """
        memory = hopfield.HopfieldMemory()
        memory.store(numpy.vstack((helper.EXAMPLE_PATTERNS['X'],
                                   helper.EXAMPLE_PATTERNS['S'])))

        self.assertTrue(numpy.array_equal(
            memory.retrieve(helper.EXAMPLE_PATTERNS['X']), 
            helper.EXAMPLE_PATTERNS['X']))
        self.assertTrue(numpy.array_equal(
            memory.retrieve(helper.EXAMPLE_PATTERNS['S']), 
            helper.EXAMPLE_PATTERNS['S']))

if __name__ == "__main__":
    unittest.main()
