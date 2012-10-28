#!/usr/bin/env python

import unittest
import doctest

import cogbox.behavior.behavior_network as bn

class Test(unittest.TestCase):
    """Unit tests for the Behavior Network"""


    def test_doctests(self):
        """Run behavior network doctests"""
        doctest.testmod(bn)

if __name__ == "__main__":
    unittest.main()
