def mapping_dictionary(matrix, i, j):
    """
    Creates a mapping dictionary of matrix from i to j
    The dictionary will have the following format
    d[element in col i] = list(all elements in column j that
                              this element maps to)
    :param matrix: database as matrix
    :param i: column i
    :param j: column j
    :return: dictionary
    """
    unique_columns_i = {}
    for row in matrix:
        if row[i] in unique_columns_i:
            unique_columns_i[row[i]].append(row[j])
        else:
            unique_columns_i[row[i]] = [row[j]]
    return unique_columns_i

