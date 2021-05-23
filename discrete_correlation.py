import math
import sys
from collections import Counter
from itertools import repeat, chain

import matplotlib.pyplot as plt
import numpy as np
import seaborn
from pybloomfilter import BloomFilter
from tqdm import tqdm

import encode_median_regression
import multiset_encoding
import test_filter_fpr
import util

seaborn.set_style('whitegrid')

def analyze_multiplicity(matrix, i, j):
    """
    Collects statistics for each element in two given columns of a database
    Represented as a matrix
    Statistics: For each column, what is the average number of x when counting
    over all 1 to x mappings
    For each element in each column, what is the number of unique elements it corresponds
    to
    ex) Take the database
    1 1
    1 2
    1 3
    2 2
    4 4
    avg_1_to_x = 5/3
    avg_x_to_1 = 5/4
    unique_mappings_count_i: {1: 3, 2: 1, 4: 1}
    unique_mappings_count_j: {1: 1, 2: 2, 3: 1, 4: 1}
    :param matrix: database as a matrix
    :param i: first column
    :param j: second column
    :return: avg_1_to_x, avg_x_to_1, unique_mappings_count_i, unique_mappings_count_j, unique_mappings_frac_i, unique_mappings_frac_j
    """
    unique_mappings_i = {}
    unique_mappings_j = {}
    for row in matrix:
        if row[i] not in unique_mappings_i:
            unique_mappings_i[row[i]] = set([row[j]])
        else:
            unique_mappings_i[row[i]].add(row[j])
        if row[j] not in unique_mappings_j:
            unique_mappings_j[row[j]] = set([row[i]])
        else:
            unique_mappings_j[row[j]].add(row[i])
    avg_1_to_x = len(matrix) / len(unique_mappings_i)
    avg_x_to_1 = len(matrix) / len(unique_mappings_j)
    unique_mappings_count_i = {x: len(unique_mappings_i[x]) for x in unique_mappings_i}
    unique_mappings_count_j = {x: len(unique_mappings_j[x]) for x in unique_mappings_j}
    unique_mappings_frac_i = {x: len(unique_mappings_i[x]) / len(unique_mappings_j) for x in unique_mappings_i}
    unique_mappings_frac_j = {x: len(unique_mappings_j[x]) / len(unique_mappings_i) for x in unique_mappings_j}
    return avg_1_to_x, avg_x_to_1, unique_mappings_count_i, unique_mappings_count_j, unique_mappings_frac_i, unique_mappings_frac_j


def plot_multiplicity(unique_mappings_count_i, unique_mappings_count_j, output_file_base, i_col_name, j_col_name,
                      show=False):
    i_hist_list = list(unique_mappings_count_i.values())
    j_hist_list = list(unique_mappings_count_j.values())
    plt.hist(i_hist_list)
    plt.title("Histogram for i")
    plt.savefig(output_file_base + "_" + i_col_name + "_to_" + j_col_name)
    if show:
        plt.show()
    plt.clf()

    plt.hist(j_hist_list)
    plt.title("Histogram for j")
    plt.savefig(output_file_base + "_" + j_col_name + "_to_" + i_col_name)
    if show:
        plt.show()
    plt.clf()


def encode_multiplicity(matrix, df):
    """
    Encodes all columns of the matrix with the following schema
    The most common element is 0, the next most common is 1, etc.
    :param matrix: input matrix
    :param df: input dataframe
    :return: nothing, matrix is modified in place
    """
    for i in range(0, len(matrix[0])):

        print("col type", df.columns[i], type(matrix[0][i]), matrix[0][i])
        df[df.columns[i]] = df[df.columns[i]].fillna(-1)

        for j in range(0, len(matrix)):
            if isinstance(matrix[j][i], float):
                if math.isnan(matrix[j][i]):
                    matrix[j][i] = -1

        if True or type("string") == type(matrix[0][i]):
            map_dict = {}
            # print(matrix[i])
            temp_col = []
            for t in range(0, len(matrix)):
                map_dict[matrix[t][i]] = 0
                temp_col.append(matrix[t][i])

            print("sorting by freq")
            new_list = list(chain.from_iterable(repeat(i, c) for i, c in Counter(temp_col).most_common()))
            print("sorting by freq done")

            count = 0
            map_dict[new_list[0]] = 0
            for t in range(0, len(new_list)):
                if t == 0:
                    continue
                if new_list[t] != new_list[t - 1]:
                    count += 1
                map_dict[new_list[t]] = count

            print("cardinality of col", i, "is:", count)

            df[df.columns[i]] = df[df.columns[i]].map(map_dict)
            for j in range(0, len(matrix)):
                # print(matrix[j][j],map_dict[matrix[i][j]])
                matrix[j][i] = map_dict[matrix[j][i]]
    return


def plot_fpr_size(correl_fpr_list, correl_size_list, normal_fpr_list, normal_size_list, show=False):
    print(correl_fpr_list)
    print(correl_size_list)
    print(normal_fpr_list)
    print(normal_size_list)
    plt.plot(correl_size_list, correl_fpr_list, marker='x', markerfacecolor='black', markersize=8, color='red',
             linestyle='-', linewidth=2, label='Correl')
    plt.plot(normal_size_list, normal_fpr_list, marker='x', markerfacecolor='black', markersize=8, color='orange',
             linestyle='--', linewidth=2, label='Normal')
    plt.yscale('log')
    plt.legend(loc='upper right')
    plt.xlabel('Size')
    plt.ylabel('False Positive Rate')
    plt.xlim(left=0)
    plt.tight_layout()
    # if only_last:
    #   plt.savefig("plbf_gauss_low_last_3.png")
    # else:
    #   plt.savefig("plbf_gauss_low_3.png")
    plt.savefig("discrete_correl.png")
    if show:
        plt.show()
    plt.clf()


def transform_nans_to_str(matrix):
    """
    Transforms all float nan values to string "nan"
    This is done so that nans are seen as equal by encoding scheme
    :param matrix: input matrix
    :return: new matrix with all nans -> "nan"
    """
    new_matrix = []
    for row in matrix:
        new_row = []
        for ele in row:
            if ele != ele:  # only nans aren't equal to themselves
                new_row.append("nan")
            else:
                new_row.append(ele)
        new_matrix.append(new_row)
    return new_matrix


def test_filter(file_name, acceptance_list, dataset_name, show=False):
    """
    Benchmarks an encoding scheme
    :param file_name: input file
    :param acceptance_list: list of columns to filter
    :param dataset_name: name of dataset ex) dmv
    :param show: if true, shows plots
    :return: nothing
    """
    matrix, df = util.read_and_create_matrix(file_name, acceptance_list)
    matrix = matrix[:100000]
    matrix = transform_nans_to_str(matrix)
    print("After Nans have been removed, length is ", len(matrix))

    output = []
    for i in range(len(matrix[0])):
        for j in range(len(matrix[0])):
            if i == j:
                continue
            matrix_random, _, _ = encode_median_regression.encode_random(matrix, df, i, j)

            matrix_multiset, encoding_scheme_i, encoding_scheme_j, factor, cutoff = multiset_encoding.encode_multiset(
                matrix, df, i, j)
            # print (matrix_multiset[:10])
            print(factor, cutoff)

            # random.shuffle(matrix_correl)
            # random.shuffle(matrix_random)

            block_size_list = [2 ** x for x in range(10, 15)]
            grid_size_list = [10, 20, 30, 40, 50]
            correl_fpr_list = []
            bloom_fpr_list = []
            two_col_bloom_fpr_list = []
            correl_size_list = []
            two_col_bloom_size_list = []
            for block_size in block_size_list:
                print("Block Size:", block_size)
                correl_fpr = test_filter_fpr.analyze_fpr_2d(matrix_multiset, multiset_encoding.DisjointSetBloomFilter2d,
                                                        factor, cutoff, block_size, 0.00001, block_size=block_size,
                                                        test_amount=2 ** 11, show=False)
                bloom_fpr = test_filter_fpr.analyze_fpr_2d(matrix_random, encode_median_regression.BloomFilter2d,
                                                       block_size, 0.00001, block_size=block_size, test_amount=2 ** 11,
                                                       show=False)
                two_col_bloom_fpr = test_filter_fpr.analyze_fpr_2d(matrix_random,
                                                               multiset_encoding.TwoColumnBloomFilter,
                                                               block_size, 0.00001, block_size=block_size,
                                                               test_amount=2 ** 11,
                                                               show=False)
                correl_fpr_list.append(correl_fpr)
                bloom_fpr_list.append(bloom_fpr)
                two_col_bloom_fpr_list.append(two_col_bloom_fpr)
                print("Multiset + Bloom fpr:", correl_fpr)
                print("Bloom only fpr:", bloom_fpr)
                print("Two column bloom filter fpr:", two_col_bloom_fpr)

            plt.plot(block_size_list, correl_fpr_list, label="Multiset Encoding + Bloom")
            plt.plot(block_size_list, bloom_fpr_list, label="Bloom")
            plt.plot(block_size_list, two_col_bloom_fpr_list, label="Two column bloom")
            plt.xlabel("Block size")
            plt.xscale("log")
            plt.ylabel("False positive Rate")
            plt.title("{} Data ({} vs {})".format(dataset_name, df.columns[i], df.columns[j]))
            plt.legend()
            plt.savefig("{}_plots/multiset_{}_{}.png".format(dataset_name, df.columns[i], df.columns[j]))
            if show:
                plt.show()
            plt.clf()
            improvement_ratio = []
            for c, b in zip(correl_fpr_list, bloom_fpr_list):
                if b != 0:
                    improvement_ratio.append(1 - c / b)

            # sometimes it breaks and both fprs are always 0
            if len(improvement_ratio) == 0:
                continue
            improvement_ratio = sum(improvement_ratio) / len(improvement_ratio)
            if improvement_ratio >= 1:
                improvement_ratio = -1
            else:
                improvement_ratio = 1 / (1 - improvement_ratio)
            print("Improvement ratio is: {}".format(improvement_ratio))
            output.append([df.columns[i], df.columns[j], improvement_ratio])
            # plt.show()
    output.sort(key=lambda x: x[2], reverse=True)
    output = [x[0] + " " + x[1] + " " + str(x[2]) + "\n" for x in output]
    with open("{}_plots/{}_encoding.txt".format(dataset_name, dataset_name), "w") as fout:
        fout.writelines(output)

    return


# DMV
# file_name = "dmv.csv"
# file_name = "census1881.csv"
# file_name = "imdb.csv"
file_name = "all_stocks_5yr.csv"
# file_name="dmv_tiny.csv"
# acceptance_list=["col0", "col1"]
# acceptance_list=["State","Zip"]
# acceptance_list = ["kind", "title"]
# acceptance_list = ["title", "imdb_index", "kind", "production_year", "season_nr", "espisode_nr", "series_year",
#                    "company_name", "country_code"]
# acceptance_list = ["Date", "Time", "Location", "Code"]
# acceptance_list = ["First Name", "Last Name", "Age", "Sect", "Province", "Occupation", "Village"]
# acceptance_list = ["high", "low"]
# acceptance_list = ["season_nr", "imdb_index"]
acceptance_list = ["date","open","high","low","close","volume","Name"]
# acceptance_list = ["First Name", "Age"]
# acceptance_list = ["Age", "Occupation"]
# acceptance_list=["Record Type","Registration Class"]
# acceptance_list=["City","Zip","County","State","Color","Model Year"]
# acceptance_list = ["State", "Zip"]
# acceptance_list=["City","Zip","County","State","Color"]
# query_cols=[["City","Color"],["City"],["Color"],["Body Type"],["State","Color"],["City","Color","State"],["Color","Body Type"]]
# target_fpr=0.01

# benchamrk_bloom_filter_real(file_name,acceptance_list)
test_filter(file_name, acceptance_list, "stock")
