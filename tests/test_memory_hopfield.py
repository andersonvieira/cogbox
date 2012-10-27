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
        memory.store(numpy.array([helper.example_patterns['X']]))
        self.assertTrue(numpy.array_equal(
            memory.retrieve(helper.example_patterns['X']), 
            helper.example_patterns['X']))

    def test_double_retrieve(self):
        """
        Test if two patterns are retrieved correctly after being stored
        """
        memory = hopfield.HopfieldMemory()
        memory.store(numpy.vstack((helper.example_patterns['X'],
                                   helper.example_patterns['S'])))

        self.assertTrue(numpy.array_equal(
            memory.retrieve(helper.example_patterns['X']), 
            helper.example_patterns['X']))
        self.assertTrue(numpy.array_equal(
            memory.retrieve(helper.example_patterns['S']), 
            helper.example_patterns['S']))

if __name__ == "__main__":
    unittest.main()
