import pandas as pd
import math

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

def delete_rows_with_zeros(matrix):
    """
    Deletes all rows with zeros from the matrix
    :param matrix: input matrix database
    :return: new matrix with all rows with 0s deleted
    """
    new_matrix = []
    for row in matrix:
        zeros = False
        for entry in row:
            if entry == 0:
                zeros = True
        if not zeros:
            new_matrix.append(row)
    return new_matrix

