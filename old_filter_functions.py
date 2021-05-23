import math
import sys

import numpy as np
from pybloomfilter import BloomFilter
from tqdm import tqdm


def unique_rows(a):
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)] * a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))

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