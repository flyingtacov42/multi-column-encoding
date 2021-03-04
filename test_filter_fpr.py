import random

def analyze_fpr_2d_1_round(filter_block, test_block, filter_2d, test_amount=2 ** 13, show=True, verbose=0):
    """
    1 round of finding the false positive ratio
    This is usually a helper function for
    analyze_fpr_2_dim_bloom_correl
    :return: false positive rate (fpr)
    """
    if not test_block:
        print("Block size larger than matrix size")
        return None
    filter_2d.build_filter(filter_block)

    try:
        sample_items = random.sample(test_block, test_amount)
    except ValueError:
        print("Test amount is greater than filter block length")
        return None

    filter_set = set((tuple(x) for x in filter_block))
    # print("number of intersections: {}".format(len(filter_set & test_set)))
    # print(len(filter_set))
    false_pos_count = 0
    negative_count = 0
    for item in sample_items:
        if tuple(item) in filter_set:
            continue
        if item in filter_2d:
            false_pos_count += 1
        negative_count += 1
    if verbose > 0:
        print("False positive rate is: {}".format(false_pos_count / negative_count))
        print("True positive count: {}".format(test_amount - negative_count))
    if show:
        filter_2d.plot()

    if negative_count == 0:
        return 0
    return false_pos_count / negative_count


def analyze_fpr_2d(matrix, filter_2d_class, *args, block_size=2 ** 13, test_amount=2 ** 13, show=True):
    """
    Analyzes the false positive rate of correlation filters
    This function splits the matrix into equally sized blocks
    and finds the fpr from each block
    The function then randomly chooses rows from the other blocks
    to test against the correlation filter for the chosen block
    :param show: if true, shows all plots
    :param test_amount: amount of rows to test per iteration
    :param filter_2d_class: A filter class
    :param matrix: n rows by 2 columns database in matrix form
    The first column is encoded categorical and second is numerical
    :param block_size: number of rows to build the database with
    :param args: Arguments to build the filter
    :return: average false positive rate
    """
    cutoff = block_size
    fpr_list = []
    while cutoff < len(matrix):
        filter_block = matrix[cutoff - block_size:cutoff]
        test_block = matrix[:cutoff - block_size] + matrix[cutoff:]
        filter_2d = filter_2d_class(*args)
        fpr_list.append(analyze_fpr_2d_1_round(filter_block, test_block, filter_2d, test_amount, show))
        cutoff += block_size
    return sum(fpr_list) / len(fpr_list)