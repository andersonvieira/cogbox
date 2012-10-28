#!/usr/bin/env python

import unittest
import doctest

import numpy

import cogbox.memory.helper as helper


class Test(unittest.TestCase):
    """Unit tests for the Memory Helper"""


    def test_doctests(self):
        """Run memory helper doctests"""
        doctest.testmod(helper)

if __name__ == "__main__":
    unittest.main()
