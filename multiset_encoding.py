import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from copy import deepcopy
from collections import Counter
from pybloomfilter import BloomFilter
from util import mapping_dictionary
from base_filters import Filter, BloomFilter2d

def encode_multiset(matrix, df, col_i, col_j):
    """
    Encodes a matrix with multiset encoding
    Columns i and j can be either categorical or numerical
    Picks the elements in column i who collectively form the
    largest disjoint multiset in column j
    Encodes the corresponding elements in j
    :param matrix: Matrix as python array
    :param df: pandas dataframe
    :param i: first column
    :param j: second column
    :return: Encoded matrix, encoding scheme i, encoding scheme j, factor, cutoff
    Cutoff is upper bound for encoding for column i
    """
    mapping_dictionary_i = mapping_dictionary(matrix, col_i, col_j)
    col_j_complete_set = set([row[col_j] for row in matrix])
    mapping_list = sorted(mapping_dictionary_i, key=lambda x: len(mapping_dictionary_i[x]), reverse=True)
    col_j_set = set([])
    col_i_encoded_list = []
    factor = 0
    encoding_scheme_i = {}
    encoding_scheme_j = {}
    print (mapping_list[:10])
    for element in mapping_list:
        test_set = set(mapping_dictionary_i[element])
        if not test_set.intersection(col_j_set):
            col_i_encoded_list.append(element)
            col_j_set = col_j_set.union(test_set)
            factor = max(factor, len(col_j_set))

    for counter, element in enumerate(col_i_encoded_list):
        encoding_scheme_i[element] = counter
        for i, corresponding in enumerate(set(mapping_dictionary_i[element])):
            encoding_scheme_j[corresponding] = counter * factor + i

    cutoff = factor * len(col_i_encoded_list)
    for remaining in set(mapping_dictionary_i) - set(col_i_encoded_list):
        encoding_scheme_i[remaining] = cutoff
        cutoff += 1
    for remaining in col_j_complete_set:
        encoding_scheme_j[remaining] = cutoff
        cutoff += 1

    new_matrix = []
    for row in matrix:
        new_matrix.append([encoding_scheme_i[row[col_i]], encoding_scheme_j[row[col_j]]])
    meaningful_count = 0
    for element in mapping_dictionary_i:
        if element in col_i_encoded_list:
            meaningful_count += len(mapping_dictionary_i[element])
    print ("Fraction of elements encoded meaningfully: {}".format(meaningful_count / len(matrix)))
    return new_matrix, encoding_scheme_i, encoding_scheme_j, factor, len(col_i_encoded_list)


class DisjointSetFilter2d(Filter):
    """
    A filter that works by filtering out elements
    that don't match the disjoint set encoding scheme
    The logic is as follows:
    strictly below a certain cutoff, the first element of each query
    must be equal to the floor of the second element divided
    by factor
    """
    def __init__(self, factor, cutoff):
        super().__init__()
        self.factor = factor
        self.cutoff = cutoff

    def __contains__(self, item):
        if item[0] < self.cutoff and item[0] != item[1] // self.factor:
            return False
        return True

class DisjointSetBloomFilter2d(Filter):
    """
    Wrapper class for disjoint set filter and bloom filter
    """
    def __init__(self, factor, cutoff, capacity, error_rate):
        super().__init__()
        self.disjoint_filter = DisjointSetFilter2d(factor, cutoff)
        self.bloom_filter = BloomFilter2d(capacity, error_rate)

    def __contains__(self, item):
        if item in self.bloom_filter and item in self.disjoint_filter:
            return True
        return False

    def build_filter(self, matrix):
        self.bloom_filter.build_filter(matrix)