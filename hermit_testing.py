import TRS_tree
import util
import encode_median_regression
import math
import min_max_map
import bisect


def test_hermit(matrix, trs_tree):
    pass


def build_hermit(encoded_matrix, host_col, target_col, outlier_ratio=0.1, error_bound=0.5, verbose=False):
    """
    Builds a TRS tree on the encoded matrix
    Assumes the host column is column 0 and target column is column 1
    :param encoded_matrix: database in matrix form
    :return: a TRS tree
    """
    trs_tree = TRS_tree.TRSTree(4, outlier_ratio, error_bound)
    trs_tree.build_tree(encoded_matrix, host_col, target_col, verbose=verbose)
    return trs_tree


def test_fpr_hermit(trs_tree, encoded_matrix, host_index, target_index):
    """
    Tests the false positive rate of hermit
    when queried with point queries on the categorical column
    The TRS tree creates a range query on the host column
    Then, the fpr is defined as
    fpr = # of rows where row[target_col] = point_query / total rows in range query

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
    for element in target_col_set:
        host_low, host_high, outliers = trs_tree.get_host_range(element, element)
        submatrix = find_submatrix_in_range(sorted_matrix, host_index, host_low, host_high)
        count = 0
        for row in submatrix:
            if row[target_index] == element:
                count += 1
        count += len(outliers[element])
        avg_fpr += count / (sum([len(x) for x in outliers.values()]) + len(submatrix))
    return avg_fpr / len(target_col_set)

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

def test_fpr_mmm(mmm, matrix, host_index, target_index):
    """
    Tests the fpr of the min max map
    :param mmm: min max map
    :param matrix: unencoded database in matrix form
    :param host_index: host index
    :param target_index: target index
    :return: average fpr across all unique elements in target column
    """
    sorted_matrix = sorted(encoded_matrix, key=lambda x: x[host_index])
    target_col = [row[target_index] for row in encoded_matrix]
    target_col_set = set(target_col)
    avg_fpr = 0
    for element in target_col_set:
        host_low, host_high = mmm.get_bounds(element)
        submatrix = find_submatrix_in_range(sorted_matrix, host_index, host_low, host_high)
        count = 0
        for row in submatrix:
            if row[target_index] == element:
                count += 1
        avg_fpr += count / (len(submatrix))
    return avg_fpr / len(target_col_set)

def find_submatrix_in_range(encoded_matrix, host_col, low_cutoff, high_cutoff):
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
    host_col = [row[host_col] for row in encoded_matrix]
    low_index = bisect.bisect_left(host_col, low_cutoff)
    high_index = bisect.bisect_right(host_col, high_cutoff)
    return encoded_matrix[low_index:high_index]


if __name__ == "__main__":
    matrix, df = util.read_and_create_matrix("all_stocks_5yr.csv", ["open", "Name"])
    matrix = util.delete_rows_with_nans(matrix)
    print (matrix[:10])
    encoded_matrix, encoding_scheme = encode_median_regression.encode_median_regression(matrix, df, 1, 0)
    print (encoding_scheme)
    print (encoded_matrix[0])
    encoded_matrix = util.delete_rows_with_nans(encoded_matrix)
    trs_tree = build_hermit(encoded_matrix, 1, 0, outlier_ratio=0.1, error_bound=100000, verbose=True)
    print("Number of nodes: ", trs_tree.get_num_nodes())
    low, high, outliers = trs_tree.get_host_range(0, 505)
    outliers_length = sum([len(x) for x in outliers.values()])
    print("number of outliers: ", outliers_length)
    print("total rows: ", len(encoded_matrix))
    print("fraction encoded successfully: ", 1 - outliers_length / len(encoded_matrix))
    print("Calculated low and high: {}, {}".format(low, high))
    low = min([row[1] for row in encoded_matrix])
    high = max([row[1] for row in encoded_matrix])
    outliers_list = []
    for x in outliers.values():
        outliers_list += x
    print ("Min and max of outliers: {}, {}".format(min(outliers_list), max(outliers_list)))
    print("Actual low and high: {}, {}".format(low, high))
