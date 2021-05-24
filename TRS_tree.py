import bisect
from scipy.stats import linregress
import random
import math
import numpy as np

class TRSTree:
    """
    Creates a TRS tree as described in HERMIT
    """

    def __init__(self, node_fanout, outlier_ratio, error_bound):
        """
        Constructor
        :param node_fanout: number of child nodes per node
        :param outlier_ratio: Maximum fraction of points that can
        be outliers
        :param error_bound: Margin of error for linear correlation
        """
        self.min = 0
        self.max = 0
        self.node_fanout = node_fanout
        self.outlier_ratio = outlier_ratio
        self.error_bound = error_bound
        self.root = None

    def build_tree(self, matrix, host_index, target_index, verbose=False):
        """
        Constructs the TRS tree
        :param matrix: database in matrix form
        :param df: pandas dataframe
        :param host_index: the host column (the one you already know the index of)
        :param target_index: target column
        :return: Nothing
        """
        host_col = [row[host_index] for row in matrix]
        target_col = [row[target_index] for row in matrix]
        self.root = TRSNode(min(target_col), max(target_col), [], {})
        submatrix = [[row[host_index], row[target_index]] for row in matrix]
        submatrix = sorted(submatrix, key=lambda x: x[1])
        queue = [self.root]
        while queue:
            node = queue.pop(0)
            low_cutoff, high_cutoff = node.low_cutoff, node.high_cutoff
            bounded_matrix = self._get_rows_in_range(submatrix, low_cutoff, high_cutoff)
            alpha, beta = node.create_regression(bounded_matrix)
            outliers = node.find_outliers(bounded_matrix, self.error_bound)
            if verbose:
                print("Node with low cutoff = {} and high cutoff = {}".format(low_cutoff, high_cutoff))
                print("Regression: (alpha, beta) = {}, {}".format(alpha, beta))
                # print("Outliers: {}".format(outliers))
            if not node.validate(bounded_matrix, self.outlier_ratio):
                child_nodes = node.split(self.node_fanout)
                for child in child_nodes:
                    queue.append(child)

    @staticmethod
    def _get_rows_in_range(submatrix, low_cutoff, high_cutoff):
        """
        Gets the rows where the target column (2nd column) lies between
        low cutoff and high cutoff
        :param submatrix: matrix with only host column and target column
        MUST be sorted in ascending order by target column entries
        :param low_cutoff: lower bound
        :param high_cutoff: upper bound
        :return: rows in matrix between lower and upper bound
        """
        target_col = [row[1] for row in submatrix]
        low_index = bisect.bisect_left(target_col, low_cutoff)
        high_index = bisect.bisect_right(target_col, high_cutoff)
        return submatrix[low_index:high_index]

    def get_host_range(self, low_cutoff, high_cutoff):
        """
        Gets the host range from the target range
        Outliers in the host range are also included
        :param low_cutoff: lower bound on target
        :param high_cutoff: upper bound on target
        :return: a host range, plus a dictionary of any outliers
        """
        host_low_cutoff, host_high_cutoff = math.inf, -math.inf
        leaf_nodes = self._get_all_leaf_nodes()
        outliers = {}
        for leaf in leaf_nodes:
            node_low, node_high, node_outliers = leaf.get_host_range(low_cutoff, high_cutoff)
            if node_low is not None:
                host_low_cutoff = min(node_low, host_low_cutoff)
                host_high_cutoff = max(node_high, host_high_cutoff)
                for entry in node_outliers:
                    if entry in outliers:
                        outliers[entry] = list(set(outliers[entry] + node_outliers[entry]))
                    else:
                        outliers[entry] = node_outliers[entry]
        return host_low_cutoff, host_high_cutoff, outliers

    def _get_all_leaf_nodes(self):
        """
        Gets all leaf nodes
        :return: a list of all leaf nodes
        """
        leaf_nodes = []
        queue = [self.root]
        while queue:
            node = queue.pop(0)
            children = node.get_children()
            if children:
                queue += children
            else:
                leaf_nodes.append(node)
        return leaf_nodes

    def get_all_nodes(self):
        """
        Gets a list of all nodes
        :return: a list of all nodes
        """
        all_nodes = []
        queue = [self.root]
        while queue:
            node = queue.pop(0)
            all_nodes.append(node)
            children = node.get_children()
            queue = queue + children
        return all_nodes

    def get_num_nodes(self):
        return len(self.get_all_nodes())

    def get_size(self):
        """
        Gets the "size" of the trs tree
        Size is defined as the number of "pointers" in the tree
        Size is defined as the number of nodes + size of each node
        :return:
        """
        size = 0
        for node in self.get_all_nodes():
            size += 1
            size += node.get_size()
        return size

class TRSNode:
    """
    A node in the TRS tree. See Hermit paper for details

    This class covers both the leaf node and internal node case
    In the case of a leaf node, the children will be empty
    and the outliers may be nonempty
    In the case of an "internal" (non-leaf) node, the children
    will be nonempty and the outliers will be empty
    """

    def __init__(self, low_cutoff, high_cutoff, children, outliers):
        """
        Constructor
        :param low_cutoff: minimum value for the target column
        :param high_cutoff: max value for the target column
        :param children: child nodes of this node
        :param outliers: a dictionary from target -> list of host entries
        """
        self.low_cutoff = low_cutoff
        self.high_cutoff = high_cutoff
        self.children = children
        self.outliers = outliers
        self.alpha = 0
        self.beta = 0
        self.regression_flag = False # Flag to detect nans in regression

    def create_regression(self, database_rows):
        """
        Creates the regression and calculates alpha and beta
        The regression is defined as:
        n = beta*m + alpha +- epsilon,
        where n is the host column entry and
        m is the target column entry
        :param database_rows: rows of database that fall within min and max value
        Col 0 is host col and col 1 is target col
        :return: alpha and beta
        """
        host_col = [row[0] for row in database_rows]
        target_col = [row[1] for row in database_rows]
        if len(database_rows) > 1:
            slope, intercept, r, p, se = linregress(target_col, host_col)
            self.alpha = intercept
            self.beta = slope
            # if nan intercept and slope
            # This can occur if all of target_col is the same number
            if np.isnan(intercept):
                self.alpha = 0
                if target_col[0] != 0:
                    self.beta = np.mean(host_col) / (target_col[0])
                else:
                    self.beta = 0
                self.regression_flag = True
        elif len(database_rows) == 1:
            self.alpha = database_rows[0][0]
            self.beta = 0
        else:
            self.alpha = 0
            self.beta = 0
        return self.alpha, self.beta

    def find_outliers(self, database_rows, error_bound):
        """
        Finds all outliers in the node's range and puts them into a set
        of outliers
        :param database_rows: rows of the database in node's range
        :param error_bound: error allowed for the regression
        :return: outliers set
        """
        if len(database_rows) == 0:
            self.epsilon = 0
            return self.outliers
        self.epsilon = self.beta * (self.high_cutoff - self.low_cutoff) * error_bound / (2 * len(database_rows))
        for host_entry, target_entry in database_rows:
            if abs(self._target_to_host(target_entry) - host_entry) > abs(self.epsilon):
                if target_entry in self.outliers:
                    self.outliers[target_entry].append(host_entry)
                else:
                    self.outliers[target_entry] = [host_entry]
        return self.outliers

    def _target_to_host(self, target_value):
        return target_value * self.beta + self.alpha

    def validate(self, database_rows, outlier_ratio):
        """
        Checks if the node is correct as is or needs splitting
        :param database_rows: rows of the database to check against
        :param outlier_ratio: maximum fraction of outliers allowed
        :return: true if fraction of outliers < outlier_ratio
        """
        # print ("validating")
        if len(database_rows) == 0:
            return True
        elif self._outlier_length() / len(database_rows) <= outlier_ratio:
            return True
        elif self.regression_flag:
            return True
        return False

    def split(self, num_children):
        """
        Splits the node into children. Erases all outliers
        :param num_children: number of children of this node
        :return: a list of the child nodes
        """
        self.outliers = {}
        for i in range(num_children):
            low_cutoff = i / num_children * (self.high_cutoff - self.low_cutoff) + self.low_cutoff
            high_cutoff = (i + 1) / num_children * (self.high_cutoff - self.low_cutoff) + self.low_cutoff
            self.children.append(TRSNode(low_cutoff, high_cutoff, [], {}))
        return self.children

    def get_host_range(self, low_cutoff, high_cutoff):
        """
        Gets the host range from the node

        Host range is interpolated from the relationship
        between low_cutoff and the node's own low cutoff
        and the high cutoff and the node's own high cutoff

        Epsilon is also included

        If low_cutoff and high_cutoff are both outside the range
        of this node, returns none for both host ranges
        :param low_cutoff: lower bound from tree
        :param high_cutoff: high bound from tree
        :param epsilon: epsilon
        :return: host range, with all outliers in the range
        """
        assert(low_cutoff <= high_cutoff)
        if high_cutoff < self.low_cutoff or low_cutoff > self.high_cutoff:
            return None, None, {}
        if low_cutoff < self.low_cutoff:
            low_cutoff = self.low_cutoff
        if high_cutoff > self.high_cutoff:
            high_cutoff = self.high_cutoff
        host_low_cutoff = self._target_to_host(low_cutoff)
        host_high_cutoff = self._target_to_host(high_cutoff)
        if host_low_cutoff > host_high_cutoff:
            host_low_cutoff, host_high_cutoff = host_high_cutoff, host_low_cutoff
        host_low_cutoff -= abs(self.epsilon)
        host_high_cutoff += abs(self.epsilon)
        outliers_in_range = {}
        for outlier in self.outliers:
            if outlier >= self.low_cutoff and outlier <= self.high_cutoff:
                outliers_in_range[outlier] = self.outliers[outlier]
        return host_low_cutoff, host_high_cutoff, outliers_in_range

    def is_leaf(self):
        """
        :return: true if the node is a leaf node, e.g. no children
        """
        return len(self.children) == 0

    def get_children(self):
        return self.children

    def _outlier_length(self):
        """
        Gets the total number of outliers in self's own dictionary
        :return:
        """
        return sum([len(x) for x in self.outliers.values()])

    def get_size(self):
        """
        Gets the size of this node
        Size is defined in terms of the number of pointers
        Each field is defined as 1 pointer
        Each key in outliers is defined as 1 pointer
        Each list in outliers list has size equal to its length
        :return: size of node
        """
        size = 6 # Number of fields, includes alpha, beta, low_cutoff, high_cutoff,
        # epsilon, recursion_flag
        size += len(self.children)
        size += self._outlier_length()
        return size


