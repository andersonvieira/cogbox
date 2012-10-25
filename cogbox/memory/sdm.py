#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
This module implements a binary associative memory according to
Kanerva's Sparse Distributed Memory (1988). The two main components of
the memory are the address matrix, where each line corresponds to an
address, and the content matrix.

Given an address, the memory will answer by activating certain
positions in the content matrix. When storing a word in the content
matrix, the word array will be added to the activated positions. When
retrieving a word from the content matrix, all activated positions are
summed and the result, after being collapsed into a {0, 1} array is
returned.
"""

from __future__ import division
import numpy


class SparseDistributedMemory(object):
    """
    Attributes:
    :hard_addresses: storage
    :activation_threshold: activation threshold on similarity
    :counter_range: range of the counter in the contents matrix
    :contents: memory content
    """

    def __init__(self, memory_size, address_length, word_length,
                 activation_radius):
        """
        :param memory_size: number of hard locations
        :param address_length: size of each address
        :param word_length: size of the bit word
        :param activation_radius: radius of activatio
        :type memory_size: int
        :type address_length: int
        :type word_length: int
        :type activation_radius: int
        """
        self.hard_addresses = SparseDistributedMemory.convert(
            numpy.random.randint(0, 2, (memory_size, address_length)))
        self.activation_threshold = address_length - 2 * activation_radius
        self.counter_range = numpy.ones(word_length) * 15
        self.content = numpy.zeros((memory_size, word_length),
                                   dtype=numpy.int16)

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
        Add uniform noise to a word in {0, 1} according to a given
        probability.

        :param word: an array in {0, 1}
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

    def _active_locations(self, address):
        """
        Return a list containing the indices of the activated locations.

        :param address: array of bits in {0, 1}
        :type address: array
        :rtype: array
        """
        return (numpy.inner(self.hard_addresses,
                            SparseDistributedMemory.convert(address)) >=
                self.activation_threshold)

    def store(self, address, word):
        """
        Store a word in a given address in the memory.

        :param address: array of bits in {0, 1}
        :param word: array of bits in {0, 1}
        :type address: array
        :type word: array
        """
        active = self._active_locations(address)
        self.content[active] = numpy.clip(
            self.content[active] + SparseDistributedMemory.convert(word),
            -self.counter_range,
            self.counter_range)

    def retrieve(self, address):
        """
        Retrieve a word stored in the memory given an address.

        :param address: array of bits in {0, 1}
        :type address: array
        :rtype: array
        """
        return numpy.clip(
            sum(self.content[self._active_locations(address)]), 0, 1)

    def train(self, address, word, repeat=10, error=0.2):
        """
        Add random noise to a word several times and store the different
        versions in the memory.

        :param address: array of bits {0, 1}
        :param word: array of bits {0, 1}
        """
        for _ in xrange(repeat):
            self.store(SparseDistributedMemory.add_noise(address, error),
                       SparseDistributedMemory.add_noise(word, error))

    def remember(self, address):
        """
        Retrieve a word from the memory and use it as the address for
        the following retrieval. Repeat until the pattern converges.

        :param address: array of bits {0, 1}
        :type address: array
        :rtype: array
        """
        word = self.retrieve(address)
        previous_word = numpy.zeros(len(word))
        while word != previous_word:
            previous_word = word
            word = self.retrieve(previous_word)
        return word


if __name__ == "__main__":
    import doctest
    doctest.testmod()
