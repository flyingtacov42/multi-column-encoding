import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# from sklearn.svm import SVR
# from sklearn.svm import SVC
# from sklearn.ensemble import RandomForestRegressor 
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import confusion_matrix
# from statistics import mode
import statistics
import pickle
import sys
import random
from collections import Counter
import scipy.stats as ss
import copy

import math
import csv
from tqdm import tqdm

from collections import Counter
from itertools import repeat, chain

from functools import cmp_to_key
from pybloomfilter import BloomFilter

import seaborn, time

import encode_median_regression
import multiset_encoding
import test_filter_fpr

seaborn.set_style('whitegrid')


# from sklearn.linear_model import LinearRegression

# from pomegranate import BayesianNetwork

# analyze datasets and find 1 to 1 mappings
# see what would be a good mapping

def unique_rows(a):
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)] * a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))


# def return_size(start,end,matrix,cola

class discrete_correl:
    def __init__(self):
        self.exception_list_0 = []  # Many to many mappings
        self.exception_list_1 = []  # Not used
        self.exception_list_not_one = []  # Not used
        self.factor_0_to_1 = 0


def encode_discrete(matrix, df, i, j):
    a = np.array(matrix)
    temp_matrix = unique_rows(a)

    print("Sanity Check unique rows before", len(temp_matrix))

    col_name = list(df.columns)
    correl_data_struct = discrete_correl()

    print("\n")
    print("columns", col_name[i], col_name[j])

    prim_sec_map = {}
    sec_prim_map = {}

    for t in range(0, len(matrix)):
        prim_sec_map[matrix[t][i]] = set([])
        sec_prim_map[matrix[t][j]] = set([])

    for t in range(0, len(matrix)):
        prim_sec_map[matrix[t][i]].add(matrix[t][j])
        sec_prim_map[matrix[t][j]].add(matrix[t][i])

    factor = 0
    factor_list = []
    temp_exception_list_0 = set([])
    temp_exception_list_not_one = set([])

    for t in range(0, len(matrix)):
        if not (len(prim_sec_map[matrix[t][i]]) == 1 and len(sec_prim_map[matrix[t][j]]) == 1):
            temp_exception_list_not_one.add(matrix[t][i])

        if len(prim_sec_map[matrix[t][i]]) == 1:
            temp = 0
            val = next(iter(prim_sec_map[matrix[t][i]]))
            factor = max(len(sec_prim_map[val]), factor)
            factor_list.append(len(sec_prim_map[val]))
        else:
            temp_exception_list_0.add(matrix[t][i])

    print(len(temp_exception_list_0))

    # factor = 1000
    correl_data_struct.factor_0_to_1 = factor

    encoding_map = {}
    count_map = {}
    count_exception = 0
    max_col_1 = 0

    for t in range(0, len(matrix)):
        max_col_1 = max(max_col_1, matrix[t][j])

    for t in range(0, len(matrix)):

        if matrix[t][i] in encoding_map:
            continue

        if matrix[t][i] in temp_exception_list_0:
            encoding_map[matrix[t][i]] = math.floor(factor * (max_col_1 + 4) + count_exception)
            count_exception += 1
        else:
            if matrix[t][j] in count_map:
                encoding_map[matrix[t][i]] = math.floor(matrix[t][j] * factor + count_map[matrix[t][j]])
                count_map[matrix[t][j]] += 1
            else:
                count_map[matrix[t][j]] = 0
                encoding_map[matrix[t][i]] = math.floor(matrix[t][j] * factor + count_map[matrix[t][j]])
                count_map[matrix[t][j]] += 1

    one_one_val = 0
    for key in count_map.keys():
        if len(sec_prim_map[key]) == 1:
            one_one_val += 1

    for t in range(0, len(matrix)):
        matrix[t][i] = encoding_map[matrix[t][i]]

    for t in temp_exception_list_0:
        correl_data_struct.exception_list_0.append(encoding_map[t])

    for t in temp_exception_list_not_one:
        correl_data_struct.exception_list_not_one.append(encoding_map[t])

    print("length of prim_sec_map", len(prim_sec_map.keys()))

    print("one one mappings are:", one_one_val, "proportion:", one_one_val * 1.00 / len(prim_sec_map.keys()))

    print("many to many vals:", len(correl_data_struct.exception_list_0) * 1.00 / len(prim_sec_map.keys()))
    print("one to many vals:",
          (len(correl_data_struct.exception_list_not_one) - len(correl_data_struct.exception_list_0)) * 1.00 / len(
              prim_sec_map.keys()))
    print("one to one vals:", 1.00 - (len(correl_data_struct.exception_list_not_one) * 1.00 / len(prim_sec_map.keys())))

    a = np.array(matrix)
    temp_matrix = unique_rows(a)

    print("Sanity Check unique rows after", len(temp_matrix))
    print("\n\n")

    return correl_data_struct


def analyse_fpr(matrix, df, i, j, correl_data_struct, target_fpr, block_size):
    num_blocks = math.floor(len(matrix) / block_size)

    print("num blocks:", num_blocks)

    many_many_elements = set(correl_data_struct.exception_list_0)
    one_many_elements = set(correl_data_struct.exception_list_not_one)

    size_correl = 0.0
    size_normal = 0.0

    block_bloom_list_0_normal = []
    block_bloom_list_0_correl = []
    block_bloom_list_1 = []

    block_set_0 = []
    block_set_1 = []

    for t in range(0, num_blocks):
        block_set_0.append(set([]))
        block_set_1.append(set([]))

    for t in range(0, int(block_size * num_blocks)):
        ind = math.floor(t / block_size)
        block_set_0[ind].add(matrix[t][i])
        block_set_1[ind].add(matrix[t][j])

    for t in range(0, num_blocks):

        count_to_add = 0

        for item in block_set_0[t]:
            if item in one_many_elements:
                count_to_add += 1

        block_bloom_list_0_correl.append(BloomFilter(count_to_add, target_fpr))
        block_bloom_list_0_normal.append(BloomFilter(len(block_set_0[t]), target_fpr))
        block_bloom_list_1.append(BloomFilter(len(block_set_1[t]), target_fpr))

        for item in block_set_0[t]:
            block_bloom_list_0_normal[-1].add(item)
            if item in one_many_elements:
                block_bloom_list_0_correl[-1].add(item)

        # print("perecentage used:",count_to_add*1.00/len(block_set_0[t]))

        for item in block_set_1[t]:
            block_bloom_list_1[-1].add(item)

        size_normal += 1.44 * math.log(1.00 / target_fpr, 2) * len(block_set_0[t])
        size_correl += 1.44 * math.log(1.00 / target_fpr, 2) * count_to_add

    print("Size Ratio:", size_correl * 1.00 / size_normal)
    # correl_bf=BloomFilter(len(correl_data_struct.exception_list_0), 0.01)
    # for item in correl_data_struct.exception_list_0:
    #   correl_bf.add(item)
    #   # print(item)

    # correl_bf_not_one=BloomFilter(len(correl_data_struct.exception_list_not_one), 0.01)
    # for item in correl_data_struct.exception_list_not_one:
    #   correl_bf_not_one.add(item)

    # size_correl=size_normal
    # size_correl+=1.44*math.log(1.00/0.01,2)*len(correl_data_struct.exception_list_0)
    # size_correl+=1.44*math.log(1.00/0.01,2)*len(correl_data_struct.exception_list_not_one)

    num_queries_per_block = 1000

    total_negatives = 0
    total_false_positives_normal = 0
    total_false_positives_correl = 0

    for curr_block in tqdm(range(0, num_blocks)):
        rand_list = np.random.uniform(0, 1.0, num_queries_per_block)

        for t in range(0, num_queries_per_block):
            ind = math.floor(rand_list[t] * num_blocks * block_size)

            # If true positive, continue
            if matrix[ind][i] in block_set_0[curr_block]:
                if matrix[ind][i] not in many_many_elements:
                    val = math.floor(matrix[ind][i] / correl_data_struct.factor_0_to_1)
                    # This will give an error if the factor is too small
                    if val not in block_bloom_list_1[curr_block] or val not in block_set_1[curr_block]:
                        print("ERROR", val, matrix[ind][i], matrix[ind][j])
                        sys.exit(1)
                continue

            total_negatives += 1

            if matrix[ind][i] in block_bloom_list_0_normal[curr_block]:
                total_false_positives_normal += 1

            if matrix[ind][i] in many_many_elements:
                if matrix[ind][i] in block_bloom_list_0_correl[curr_block]:
                    total_false_positives_correl += 1
            else:
                val = math.floor(matrix[ind][i] / correl_data_struct.factor_0_to_1)
                if matrix[ind][i] in one_many_elements:
                    if matrix[ind][i] in block_bloom_list_0_correl[curr_block] and val in block_bloom_list_1[
                        curr_block]:
                        total_false_positives_correl += 1
                else:
                    if val in block_bloom_list_1[curr_block]:
                        total_false_positives_correl += 1

    fpr_correl = total_false_positives_correl * 1.00 / total_negatives
    fpr_normal = total_false_positives_normal * 1.00 / total_negatives
    print("Normal False positive rate:", fpr_normal)
    print("Correl False positive rate:", fpr_correl)

    print("\n\n")

    return fpr_correl, size_correl, fpr_normal, size_normal


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


def read_and_create_matrix(file_name, acceptance_list):
    df = pd.read_csv(file_name, encoding="ISO8859")
    col_list = list(df.columns)
    for i in range(0, len(col_list)):
        if col_list[i] in acceptance_list:
            continue
        df = df.drop(col_list[i], 1)

    print(col_list)
    print('df stuff', df.columns)
    print(df)
    matrix = df.values.tolist()
    return matrix, df


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


def delete_rows_with_nans(matrix):
    """
    Deletes all rows with nans from the matrix
    :param matrix: input matrix database
    :return: new matrix with all rows with any nans deleted
    """
    new_matrix = []
    for row in matrix:
        nans = False
        for entry in row:
            if isinstance(entry, (int, float)):
                if math.isnan(entry):
                    nans = True
        if not nans:
            new_matrix.append(row)
    return new_matrix


def benchmark_vortex(file_name, acceptance_list, output_file_name, show=False):
    """
    Benchmarks an encoding scheme
    :param file_name: input file
    :param acceptance_list: list of columns to filter
    :param output_file_name: file prefix to plot to
    :param show: if true, shows plots
    :return: nothing
    """
    matrix, df = read_and_create_matrix(file_name, acceptance_list)
    matrix = matrix[:100000]
    matrix = transform_nans_to_str(matrix)
    print("After Nans have been removed, length is ", len(matrix))

    output = []
    for i in range(len(matrix[0])):
        for j in range(len(matrix[0])):
            if i == j:
                continue
            matrix_random, _, _ = encode_median_regression.encode_random(matrix, df, i, j)

            matrix_multiset, _, _, factor, cutoff = multiset_encoding.encode_multiset(matrix, df, i, j)
            # print (matrix_multiset[:10])
            print(factor, cutoff)

            # random.shuffle(matrix_correl)
            # random.shuffle(matrix_random)

            block_size_list = [2 ** x for x in range(10, 15)]
            grid_size_list = [10, 20, 30, 40, 50]
            correl_fpr_list = []
            bloom_fpr_list = []
            for block_size in block_size_list:
                print("Block Size:", block_size)
                correl_fpr = test_filter_fpr.analyze_fpr_2d(matrix_multiset, multiset_encoding.DisjointSetBloomFilter2d,
                                                            factor, cutoff, block_size, 0.1, block_size=block_size,
                                                            test_amount=2 ** 11, show=False)
                bloom_fpr = test_filter_fpr.analyze_fpr_2d(matrix_random, encode_median_regression.BloomFilter2d,
                                                           block_size, 0.1, block_size=block_size, test_amount=2 ** 11,
                                                           show=False)
                correl_fpr_list.append(correl_fpr)
                bloom_fpr_list.append(bloom_fpr)
                print("Multiset + Bloom fpr:", correl_fpr)
                print("Bloom only fpr:", bloom_fpr)

            plt.plot(block_size_list, correl_fpr_list, label="Multiset Encoding + Bloom")
            plt.plot(block_size_list, bloom_fpr_list, label="Bloom")
            plt.xlabel("Block size")
            plt.xscale("log")
            plt.ylabel("False positive Rate")
            plt.title("Imdb Data ({} vs {})".format(df.columns[i], df.columns[j]))
            plt.legend()
            # plt.savefig("imdb_plots/multiset_{}_{}.png".format(df.columns[i], df.columns[j]))
            plt.show()
            plt.clf()
            improvement_ratio = []
            for c, b in zip(correl_fpr_list, bloom_fpr_list):
                # If both c and b are << 0.01, then their results matter less
                if (b != 0):
                    improvement_ratio.append(1 - c / b)

            # sometimes it breaks and both fprs are always 0
            if (len(improvement_ratio) == 0):
                continue
            improvement_ratio = sum(improvement_ratio) / len(improvement_ratio)
            if improvement_ratio >= 1:
                improvement_ratio = -1
            else:
                improvement_ratio = 1 / (1 - improvement_ratio)
            print ("Improvement ratio is: {}".format(improvement_ratio))
            output.append([df.columns[i], df.columns[j], improvement_ratio])
            # plt.show()
    output.sort(key=lambda x: x[2], reverse=True)
    output = [x[0] + " " + x[1] + " " + str(x[2]) + "\n" for x in output]
    # with open("imdb_plots/imdb_encoding.txt", "w") as fout:
    #     fout.writelines(output)

    return


# DMV
file_name = "imdb.csv"
# file_name = "census1881.csv"
# file_name="dmv_tiny.csv"
# acceptance_list=["col0", "col1"]
# acceptance_list=["State","Zip"]
acceptance_list = ["kind", "title"]
# acceptance_list = ["title","imdb_index","kind","production_year","season_nr","espisode_nr","series_year","company_name","country_code"]
# acceptance_list = ["Date", "Time", "Location", "Code"]
# acceptance_list = ["First Name", "Last Name", "Age", "Sect", "Province", "Occupation", "Village"]
# acceptance_list = ["high", "low"]
# acceptance_list = ["season_nr", "imdb_index"]
# acceptance_list = ["date","open","high","low","close","volume","Name"]
# acceptance_list = ["First Name", "Age"]
# acceptance_list = ["Age", "Occupation"]
# acceptance_list=["Record Type","Registration Class"]
# acceptance_list=["City","Zip","County","State","Color","Model Year"]
# acceptance_list = ["City", "Zip"]
# acceptance_list=["City","Zip","County","State","Color"]
# query_cols=[["City","Color"],["City"],["Color"],["Body Type"],["State","Color"],["City","Color","State"],["Color","Body Type"]]
# target_fpr=0.01

# benchamrk_bloom_filter_real(file_name,acceptance_list)
benchmark_vortex(file_name, acceptance_list, "mappings/stock_data")
