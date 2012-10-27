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

import cogbox.memory.helper as helper

__author__ = "Anderson Vieira"

class SparseDistributedMemory(object):
    """
    Public methods:
    store -- Stores a pattern in the memory according to a given address
    retrieve -- Given an address, retrieve the corresponding pattern
    
    Instance variables:
    hard_addresses -- addresses to access the memory
    activation_threshold -- activation threshold based on similarity
    counter_range -- range of the counter in the contents matrix
    content -- memory content
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
        self.hard_addresses = helper.convert(
            numpy.random.randint(0, 2, (memory_size, address_length)))
        self.activation_threshold = address_length - 2 * activation_radius
        self.counter_range = numpy.ones(word_length) * 15
        self.content = numpy.zeros((memory_size, word_length),
                                   dtype=numpy.int16)

    def _active_locations(self, address):
        """
        Return a list containing the indices of the activated locations.

        :param address: array of bits in {0, 1}
        :type address: numpy.array
        :rtype: numpy.array
        """
        return (numpy.inner(self.hard_addresses,
                            helper.convert(address)) >=
                self.activation_threshold)

    def store(self, address, word):
        """
        Store a word in a given address in the memory.

        :param address: array of bits in {0, 1}
        :param word: array of bits in {0, 1}
        :type address: numpy.array
        :type word: numpy.array
        """
        active = self._active_locations(address)
        self.content[active] = numpy.clip(
            self.content[active] + helper.convert(word),
            -self.counter_range,
            self.counter_range)

    def retrieve(self, address):
        """
        Retrieve a word stored in the memory given an address.

        :param address: array of bits in {0, 1}
        :type address: numpy.array
        :rtype: numpy.array
        """
        return numpy.clip(
            sum(self.content[self._active_locations(address)]), 0, 1)
