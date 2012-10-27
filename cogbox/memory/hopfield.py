#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
This module implements a binary associative memory according to
Hopfield (1982).
"""

from __future__ import division
import numpy

import cogbox.memory.helper as helper

__author__ = "Anderson Vieira"


class HopfieldMemory(object):
    """
    Stores a set of patterns in the memory and allows later retrieval
    with partial information

    Public methods:
    store -- Stores a set of patterns in the memory
    retrieve -- Given a probe vector, retrieve the corresponding pattern

    Instance variables:
    content -- matrix where the patterns are stored
    """
    def __init__(self):
        self.content = None

    def store(self, words):
        """
        Store a stack of words in the memory.
        Importat: Everytime this method is called it clears the content of 
        the memory and add the new content based on the words parameter.

        :param words: a matrix in which each line is a word in {0, 1}
        :type words: numpy.array
        """
        data = helper.convert(words)
        size, word_length = words.shape
        self.content = ((1 / word_length) * numpy.dot(data.T, data) -
                        (size / word_length) * numpy.identity(word_length))

    def retrieve(self, probe):
        """
        Retrieve a word stored in the memory given a probe vector.

        :param probe: array of bits in {0, 1}
        :type probe: numpy.array
        :rtype: numpy.array
        """
        assert len(probe) == len(self.content)
        word = numpy.array(helper.convert(probe))
        word_length = len(word)
        for i in numpy.random.permutation(xrange(word_length)):
            temp = 0
            while temp - word[i] != 0:
                temp = word[i]
                word[i] = numpy.sign(numpy.dot(self.content[i], word))
        return helper.bitify(word)
