#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
This module implements functions that are used by the memory models
"""

from __future__ import division
import numpy

__author__ = "Anderson Vieira"


def convert(array):
    """
    Convert an array from {0, 1} to {-1, 1}
    :param array: an array in {0, 1}
    :type array: array
    :returns: 2 * array - 1
    :rtype: array

    :Example:
    >>> import numpy
    >>> import helper
    >>> helper.convert(numpy.array([0, 1, 0, 0, 1]))
    array([-1,  1, -1, -1,  1])
    """
    return 2 * array - 1


def add_noise(array, prob):
    """
    Add uniform noise to a array in {0, 1} according to a given
    probability.

    :param array: an array in {0, 1}
    :param probability: a value between 0 and 1
    :type array: array
    :type probability: int, float
    :rtype: array

    :Example:
    >>> import numpy
    >>> import helper
    >>> helper.add_noise(numpy.array([0, 1, 0, 0, 1]), 0)
    array([0, 1, 0, 0, 1])

    :Example:
    >>> import numpy
    >>> import helper
    >>> helper.add_noise(numpy.array([0, 1, 0, 0, 1]), 1)
    array([1, 0, 1, 1, 0])
    """
    return numpy.bitwise_xor(array,
                             numpy.random.uniform(0, 1, len(array)) < prob)


def bitify(array):
    """
    Convert an array from {-1, 1} to {0, 1}
    :param array: an array in {-1, 1}
    :type array: array
    :returns: numpy.clip(array, 0, 1)
    :rtype: array

    :Example:
    >>> import numpy
    >>> import helper
    >>> helper.bitify(numpy.array([-1, 1, -1, -1, 1]))
    array([0, 1, 0, 0, 1])
    """
    return numpy.clip(array, 0, 1)

EXAMPLE_PATTERNS = dict()
EXAMPLE_PATTERNS['X'] = numpy.array(
    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
     1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
     0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0,
     0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0,
     0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0,
     0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0,
     0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0,
     0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0,
     0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0,
     1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
     1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1])

EXAMPLE_PATTERNS['S'] = numpy.array(
    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
     0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
     0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
     0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0,
     0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
     0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0])


if __name__ == "__main__":
    import doctest
    doctest.testmod()
