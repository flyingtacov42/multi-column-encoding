import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from copy import deepcopy
from collections import Counter
from pybloomfilter import BloomFilter
import sys
from util import mapping_dictionary
from base_filters import Filter, BloomFilter2d

def encode_median_regression(matrix, df, i, j):
    """
    Encodes column i and column j of the matrix with the following scheme:
    Requirements: columns i must be categorical
    column j must be numerical
    ex) Name vs Age
    For each unique value in column i, find the median
    of the values in column j that they map to
    Then, encode the values in column i with the values 0
    through (# of unique elements in col i - 1) in acsending order
    of the median values
    :param matrix: input matrix from a dataset
    :param df: pandas dataframe
    :param i: column i, the categorical column
    :param j: column j, the numerical column
    :return: encoded matrix (with only ith and jth column), encoding scheme
    """
    print("Encoding {} vs {} with median regression".format(df.columns[i], df.columns[j]))
    unique_columns_i = mapping_dictionary(matrix, i, j)

    elements_to_median = [(x, np.median(unique_columns_i[x])) for x in unique_columns_i]
    elements_to_median.sort(key=lambda x: x[1])
    encoding_scheme = {elements_to_median[i][0]: i for i in range(len(elements_to_median))}

    new_matrix = []
    for row in matrix:
        new_matrix.append([encoding_scheme[row[i]], row[j]])

    return new_matrix, encoding_scheme


def encode_categorical_vs_categorical(matrix, df, i, j):
    """
    Encodes two categorical columns of a matrix against each other
    Assigns the most frequent entry of each column to be 0
    Then, 1 is assigned to the entry that has the greatest intersection
    with entry 0, 2 for the next most, etc.
    :param matrix: database in matrix format
    :param df: pandas dataframe
    :param i: first categorical column index
    :param j: second categorical column index
    :return: encoded matrix (only ith and jth columns), encoding scheme i, encoding scheme j
    """
    print("Encoding {} vs {} with categorical encoding".format(df.columns[i], df.columns[j]))
    encoding_scheme_i = encode_categorical_vs_categorical_half(matrix, i, j)
    encoding_scheme_j = encode_categorical_vs_categorical_half(matrix, j, i)
    new_matrix = []
    for row in matrix:
        new_matrix.append([encoding_scheme_i[row[i]], encoding_scheme_j[row[j]]])
    return new_matrix, encoding_scheme_i, encoding_scheme_j


def encode_categorical_vs_categorical_half(matrix, i, j):
    """
    half of the encode categorical vs categorical function
    (this is a helper function)
    """
    counter = Counter([row[i] for row in matrix])
    encoding_scheme = {}
    unique_columns = mapping_dictionary(matrix, i, j)
    most_common = counter.most_common(1)[0][0]
    elements_left = set(counter)
    common_with_0_dict = {}
    most_common_counter = Counter(unique_columns[most_common])
    for i, element in enumerate(unique_columns):
        if i % 100 == 0:
            print (i, len(unique_columns))
        ele_counter = Counter(unique_columns[element])
        common_with_0_dict[element] = sum((most_common_counter & ele_counter).values())
    encoding_scheme = sorted(common_with_0_dict, key=lambda x: common_with_0_dict[x])
    return {encoding_scheme[i]: i for i in range(len(encoding_scheme))}


def encode_random(matrix, df, i, j):
    """
    Encodes column i and column j of the matrix with the following scheme:
    Requirements: columns i must be categorical
    column j must be numerical
    ex) Name vs Age
    Encodes column i and j randomly
    :param matrix: data matrix
    :param df: pandas dataframe
    :param i: categorical column
    :param j: numerical column
    :return: encoded matrix (with only ith and jth column), encoding scheme i, encoding scheme j
    """
    print("Encoding {} vs {} with random encoding".format(df.columns[i], df.columns[j]))
    unique_entries_i = list(set([row[i] for row in matrix]))  # Get unique elements from column i
    encoding_scheme_i = {unique_entries_i[i]: i for i in range(len(unique_entries_i))}
    unique_entries_j = list(set([row[j] for row in matrix]))  # Get unique elements from column j
    encoding_scheme_j = {unique_entries_j[i]: i for i in range(len(unique_entries_j))}

    new_matrix = []
    for row in matrix:
        new_matrix.append([encoding_scheme_i[row[i]], encoding_scheme_j[row[j]]])

    return new_matrix, encoding_scheme_i, encoding_scheme_j


class CorrelFilter(Filter):
    """
    A filter that is used in the correlation scheme defined above
    Creates a grid to keep track of what combinations of
    Categorical data and numerical data are possible
    Used after you encode data
    """

    def __init__(self, col_1_bins, col_2_bins):
        super().__init__()
        self.col_1_bins = col_1_bins
        self.col_2_bins = col_2_bins
        self.grid = np.zeros((col_1_bins, col_2_bins))
        self.categorical_max = 0
        self.categorical_min = 0
        self.numerical_max = 0
        self.numerical_min = 0

    def build_filter(self, matrix):
        """
        Builds the filter, setting all grid cells with
        at least 1 element in the cell to 1
        :param matrix: database in matrix format
        Encoded categorical column should be in column 0
        Numerical column should be in column 1
        :return:
        """
        array = np.array(matrix)
        categorical = array[:, 0]
        numerical = array[:, 1]
        self.categorical_max = np.max(categorical)
        self.categorical_min = np.min(categorical)
        self.numerical_max = np.max(numerical)
        self.numerical_min = np.min(numerical)
        for cat_entry, num_entry in zip(categorical, numerical):
            cat_index, num_index = self.find_index(cat_entry, num_entry)
            self.grid[cat_index][num_index] = 1

    def __contains__(self, item):
        """
        Returns true if the corresponding self.grid cell is 1
        and false otherwise
        Use if (item) in correl_filter
        May give false positive results but never false negatives
        :param item: a tuple of categorical data and numerical data
        :return: True if self.grid is 1 in the item's grid
        """
        index = self.find_index(item[0], item[1])
        if not index:
            return False
        return self.grid[index[0]][index[1]] == 1

    def find_index(self, cat_data, num_data):
        """
        Finds the index in the grid, using the number of bins,
        and the max and mins for the categorical and numerical
        data attributes
        :param cat_data: categorical data entry
        :param num_data: numerical data entry
        :return: cat_index, num_index
        If data is not within the bounds for either data point, return None
        """
        cat_index = int(
            (cat_data - self.categorical_min) / (self.categorical_max - self.categorical_min) * self.col_1_bins)
        num_index = int((num_data - self.numerical_min) / (self.numerical_max - self.numerical_min) * self.col_2_bins)
        if (cat_data == self.categorical_max):
            cat_index = self.col_1_bins - 1
        if (num_data == self.numerical_max):
            num_index = self.col_2_bins - 1
        if cat_index < 0 or cat_index >= self.col_1_bins or num_index < 0 or num_index >= self.col_2_bins:
            return None
        return cat_index, num_index





class CorrelBloomFilter2d(Filter):
    """
    Wrapper class that combines a Correl_filter and a Bloom Filter
    """

    def __init__(self, col_1_bins, col_2_bins, capacity, error_rate):
        super().__init__()
        self.correl_filter = CorrelFilter(col_1_bins, col_2_bins)
        self.bloom_filter_2d = BloomFilter2d(capacity, error_rate)

    def build_filter(self, matrix):
        self.correl_filter.build_filter(matrix)
        self.bloom_filter_2d.build_filter(matrix)

    def __contains__(self, item):
        if item in self.correl_filter and item in self.bloom_filter_2d:
            return True
        return False

def plot_median_regression_encoding(matrix_correlated, matrix_random, file_prefix, show=False):
    """
    Plots the correlated matrix on one plot
    And the random matrix on another
    x axis is categorical encoding
    y axis is numerical
    :param matrix_correlated: correlated matrix
    :param matrix_random: random matrix
    :param show: show the graph or not
    :return: none
    """
    encoded_correl_x = [row[0] for row in matrix_correlated]
    encoded_correl_y = [row[1] for row in matrix_correlated]
    plt.scatter(encoded_correl_x, encoded_correl_y)
    plt.xlabel("Encoded Name")
    plt.ylabel("Age")
    plt.title("Correlated")
    plt.savefig(file_prefix + "_correl.png")
    if show:
        plt.show()
    plt.clf()

    encoded_random_x = [row[0] for row in matrix_random]
    encoded_random_y = [row[1] for row in matrix_random]
    plt.scatter(encoded_random_x, encoded_random_y)
    plt.xlabel("Encoded Name")
    plt.ylabel("Age")
    plt.title("Random")
    plt.savefig(file_prefix + "_random.png")
    if show:
        plt.show()
    plt.clf()
