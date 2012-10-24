#!/usr/bin/env python

# -*- coding: utf-8 -*-

from __future__ import division
import sys

import numpy

class SparseDistributedMemory(object):
    """
    Attributes:
        :memory_size: number of hard locations
        :hard_addresses: storage
        :activation_probability: probability of activation
        :activation_radius: radius of activation
        :activation_threshold: activation threshold on similarity
        :counter_range: range of the counter in the contents matrix
        :address_length: size of each address
        :word_length: size of the bit word
    """

    def __init__(self, memory_size, address_length, word_length,
            activation_radius):
        self.memory_size = memory_size
        self.address_length = address_length
        self.word_length = word_length
        self.activation_radius = activation_radius
        self.activation_threshold = address_length - 2 * activation_radius
        self.counter_range = ones(self.word_length) * 15
        self.activation_radius = activation_radius
        
        self.hard_address = SparaseDistributedMemory.convert(
                numpy.random.randint(0, 2, (memory_size, address_length)))
        self.contents = numpy.zeros((memory_size, word_length), dtype=int16)


    @staticmethod
    def convert(array):
        """
        Convert an array from {0, 1} to {-1, 1} 
        :param array: an array in {0, 1}
        :type array: array
        :returns: 2 * array - 1
        :rtype: array
        
        :Example:
        >>> import numpy
        >>> import sdm
        >>> sdm.SparseDistributedMemory.convert(numpy.array([0, 1, 0, 0, 1]))
        array([-1,  1, -1, -1,  1])
        """
        return 2 * array - 1

    
    @staticmethod
    def add_noise(word, prob):
        """
        Add uniform noise to a word in {0, 1} according to a given probability.

        :param array: a word in {0, 1}
        :param probability: a value between 0 and 1
        :type array: array
        :type probability: int, float
        :rtype: array
        
        :Example:
        >>> import numpy
        >>> import sdm
        >>> sdm.SparseDistributedMemory.add_noise(numpy.array([0, 1, 0, 0, 1]), 0)
        array([0, 1, 0, 0, 1])

        :Example:
        >>> import numpy
        >>> import sdm
        >>> sdm.SparseDistributedMemory.add_noise(numpy.array([0, 1, 0, 0, 1]), 1)
        array([1, 0, 1, 1, 0])
        """
        return numpy.bitwise_xor(word, 
                numpy.random.uniform(0, 1, len(word)) < prob)


def __main__(argv):
    pass


if __name__ == '__main__':
    import doctest
    doctest.testmod()
    sys.exit(__main__(sys.argv[1:]))
