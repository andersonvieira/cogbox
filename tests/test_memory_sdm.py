#!/usr/bin/env python

import unittest
import doctest

import numpy

from cogbox.memory import sdm, helper 


class Test(unittest.TestCase):
    """Unit tests for the Sparse Distributed Memory"""

    def test_doctests(self):
        """Run sdm doctests"""
        doctest.testmod(sdm)

    def test_retrieve(self):
        """
        Test if a pattern is retrieved correctly after being stored
        """
        memory = sdm.SparseDistributedMemory(10000, 256, 256, 101)
        memory.store(helper.example_patterns['X'], 
                     helper.example_patterns['X'])
        self.assertTrue(numpy.array_equal(
            memory.retrieve(helper.example_patterns['X']), 
            helper.example_patterns['X']))

    def test_double_retrieve(self):
        """
        Test if two patterns are retrieved correctly after being stored
        """
        memory = sdm.SparseDistributedMemory(10000, 256, 256, 101)
        memory.store(helper.example_patterns['X'], 
                     helper.example_patterns['X'])
        memory.store(helper.example_patterns['S'], 
                     helper.example_patterns['S'])
        self.assertTrue(numpy.array_equal(
            memory.retrieve(helper.example_patterns['X']), 
            helper.example_patterns['X']))
        self.assertTrue(numpy.array_equal(
            memory.retrieve(helper.example_patterns['S']), 
            helper.example_patterns['S']))

    def test_different_address_and_word(self):
        """
        Test if a pattern is retrieved correctly after being stored
        when the pattern and the address are different
        """
        memory = sdm.SparseDistributedMemory(10000, 256, 256, 101)
        memory.store(helper.example_patterns['S'], 
                     helper.example_patterns['X'])
        self.assertTrue(numpy.array_equal(
            memory.retrieve(helper.example_patterns['S']), 
            helper.example_patterns['X']))

if __name__ == "__main__":
    unittest.main()
