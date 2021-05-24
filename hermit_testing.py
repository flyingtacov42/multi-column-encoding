import TRS_tree
import util
import encode_median_regression
import math
import min_max_map
import bisect
import random
import matplotlib.pyplot as plt


def build_hermit(encoded_matrix, host_col, target_col, outlier_ratio=0.1, error_bound=0.5, verbose=False):
    """
    Builds a TRS tree on the encoded matrix

    :param encoded_matrix: database in matrix form
    :param host_col: host column index
    :param target_col: target column index
    :return: a TRS tree
    """
    trs_tree = TRS_tree.TRSTree(4, outlier_ratio, error_bound)
    trs_tree.build_tree(encoded_matrix, host_col, target_col, verbose=verbose)
    return trs_tree


def test_efficiency_hermit(trs_tree, encoded_matrix, host_index, target_index):
    """
    Tests the false positive rate of hermit
    when queried with point queries on the categorical column
    The TRS tree creates a range query on the host column
    Then, the efficiency is defined as
    e = # of rows where row[target_col] = point_query / total rows in range query
    Also tests the size of the query on the host column
    size = (size of range query on host column + number of outliers) / len(database)

    Performs this analysis for every unique value in target_col
    :param trs_tree: TRS tree, used to get range query
    :param encoded_matrix: matrix
    :param host_index: host index
    :param target_index: target index
    :return: average false positive rate
    """
    # sort by host col so that we can binary search to find the submatrix later
    sorted_matrix = sorted(encoded_matrix, key=lambda x: x[host_index])
    target_col = [row[target_index] for row in encoded_matrix]
    target_col_set = set(target_col)
    avg_fpr = 0
    avg_size = 0
    avg_outlier_frac = 0
    for i, element in enumerate(target_col_set):
        host_low, host_high, outliers = trs_tree.get_host_range(element, element)
        submatrix = find_submatrix_in_range(sorted_matrix, host_index, host_low, host_high)
        count = 0
        for row in submatrix:
            if row[target_index] == element:
                count += 1
        range_count = count
        if element in outliers:
            count += len(outliers[element])
        avg_fpr += count / (sum([len(x) for x in outliers.values()]) + len(submatrix))
        avg_outlier_frac += sum([len(x) for x in outliers.values()]) / (sum([len(x) for x in outliers.values()]) +
                                                                        len(submatrix))
        avg_size += (len(submatrix) + sum([len(x) for x in outliers.values()])) / len(encoded_matrix)
        # print ("Fraction of entries from range query: {}".format(len(submatrix) /
        #                                                 (sum([len(x) for x in outliers.values()]) + len(submatrix))))
        # if len(submatrix) > 0:
        #     print ("Fraction of correct from just range query: {}".format(range_count / len(submatrix)))
        # print ("Submatrix: ", submatrix)
        # print ("Element: ", element)
        # print ("Count in range query: {}".format(range_count))
        # print ("Actual count: {}".format(target_col.count(element)))
    print ("Average outlier fraction: {}".format(avg_outlier_frac / len(target_col_set)))
    return avg_fpr / len(target_col_set), avg_size / len(target_col_set)

def build_min_max_map(matrix, host_index, target_index):
    """
    Builds a min max map and returns it
    :param matrix: database in matrix form, should not be encoded
    :param host_index: host index
    :param target_index: host index
    :return: a min max map object
    """
    mmm = min_max_map.MinMaxMap()
    mmm.build(matrix, host_index, target_index)
    return mmm

def test_efficiency_mmm(mmm, matrix, host_index, target_index):
    """
    Tests the fpr of the min max map
    :param mmm: min max map
    :param matrix: unencoded database in matrix form
    :param host_index: host index
    :param target_index: target index
    :return: average fpr across all unique elements in target column, also average size of range query
    in relationship to size of database
    """
    sorted_matrix = sorted(matrix, key=lambda x: x[host_index])
    target_col = [row[target_index] for row in matrix]
    target_col_set = set(target_col)
    avg_fpr = 0
    avg_size = 0
    for element in target_col_set:
        host_low, host_high = mmm.get_bounds(element)
        submatrix = find_submatrix_in_range(sorted_matrix, host_index, host_low, host_high)
        count = 0
        for row in submatrix:
            if row[target_index] == element:
                count += 1
        avg_fpr += count / (len(submatrix))
        avg_size += len(submatrix) / len(matrix)
    return avg_fpr / len(target_col_set), avg_size / len(target_col_set)

def find_submatrix_in_range(matrix, host_col, low_cutoff, high_cutoff):
    """
    Gets the rows where the host column (1st column) lies between
    low cutoff and high cutoff
    :param submatrix: matrix with only host column and target column
    MUST be sorted in ascending order by target column entries
    :param host_col: index of host column
    :param low_cutoff: lower bound
    :param high_cutoff: upper bound
    :return: rows in matrix between lower and upper bound
    """
    host_col = [row[host_col] for row in matrix]
    low_index = bisect.bisect_left(host_col, low_cutoff)
    high_index = bisect.bisect_right(host_col, high_cutoff)
    return matrix[low_index:high_index]


if __name__ == "__main__":
    col_names = ["Name", "Open"]
    matrix, df = util.read_and_create_matrix("RealisticTabularDataSets-master/census-income/census-income_srt.csv",
                                             col_names)
    random.shuffle(matrix)
    matrix = util.delete_rows_with_nans(matrix)
    matrix = util.delete_rows_with_zeros(matrix)
    matrix = matrix[:100000]
    print(len(matrix))
    print(matrix[:10])
    for i, col in enumerate(col_names):
        if i == 5:
            break
        encoded_matrix, encoding_scheme = encode_median_regression.encode_median_regression(matrix, df, i, 5)
        # Alternate encoding scheme
        # encoded_matrix, encoding_scheme = encode_median_regression.encode_categorical_to_median_numerical(matrix, df, 1, 0, epsilon=0.01)
        print (encoding_scheme)
        occs = [row[0] for row in encoded_matrix]
        wages = [row[1] for row in encoded_matrix]
        plt.scatter(occs, wages)
        plt.show()
        error_bounds = [2, 10, 100, 1000, 10000, 100000, 1000000]
        efficiencies = []
        host_sizes = []
        sizes = []
        for e in error_bounds:
            print ("Error bound = {}".format(e))
            trs_tree = build_hermit(encoded_matrix, 1, 0, outlier_ratio=0.1, error_bound=e, verbose=False)
            efficiency, host_query_size = test_efficiency_hermit(trs_tree, encoded_matrix, 1, 0)
            size = trs_tree.get_size()
            print ("TRS tree size:", size)
            sizes.append(size)
            print("TRS tree efficiency:", efficiency)
            efficiencies.append(efficiency)
            print ("TRS tree fraction of host column queried:", host_query_size)
            host_sizes.append(host_query_size)

        plt.plot(error_bounds, efficiencies)
        plt.title("Error_bound vs Efficiency (Name vs Open)")
        plt.xlabel("Error bounds")
        plt.ylabel("Efficiency")
        plt.xscale("log")
        plt.savefig("census_income_plots/trs_tree_efficiency_{}_vs_{}.png".format(col, "salary"))
        plt.clf()
        plt.plot(error_bounds, sizes)
        plt.title("Error_bound vs Size (Name vs Open)")
        plt.xlabel("Error bounds")
        plt.ylabel("Size")
        plt.xscale("log")
        plt.savefig("census_income_plots/trs_tree_size_{}_vs_{}.png".format(col, "salary"))
        plt.clf()
        plt.plot(error_bounds, host_sizes)
        plt.title("Error_bound vs Fraction of Host column queried (Name vs Open)")
        plt.xlabel("Error bounds")
        plt.ylabel("Fraction of database queried")
        plt.xscale("log")
        plt.yscale("log")
        plt.savefig("census_income_plots/trs_tree_host_frac_{}_vs_{}.png".format(col, "salary"))
        plt.clf()

        with open("census_income_plots/hermit_performance_{}_vs_{}.txt".format(col, "salary"), "w") as f:
            mmm = build_min_max_map(matrix, 5, i)
            efficiency, host_size = test_efficiency_mmm(mmm, matrix, 5, i)
            print("Min max map efficiency: ", efficiency)
            print("Min max map fraction of host column queried: ", host_size)
            size = mmm.get_size()
            print("Min max map size: ", size)
            f.write("Error bound, Efficiency, Fraction of database queried, Size\n")
            for error, efficiency, fraction, size in zip(error_bounds, efficiencies, host_sizes, sizes):
                f.write("{} {} {} {} \n".format(error, efficiency, fraction, size))
            f.write("Min max map:\n")
            f.write("efficiency: {}\n".format(efficiency))
            f.write("Fraction of database queried: {}\n".format(host_size))
            f.write("Size: {}\n".format(size))
