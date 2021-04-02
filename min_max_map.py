class MinMaxMap:
    """
    A simple map that correlates a categorical target column
    with a numerical host column
    The class stores the min and max of the data in the numerical
    column that each categorical entry corresponds to
    """
    def __init__(self):
        """
        Creates an empty min max map
        """
        self.min_map = {}
        self.max_map = {}

    def build(self, matrix, host_index, target_index):
        """
        Builds the map from a database
        :param matrix: database in matrix form
        :param host_index: index of host column
        :param target_index: index of target column
        :return: none
        """
        for i, row in enumerate(matrix):
            host = row[host_index]
            target = row[target_index]
            if target in self.min_map:
                if host < self.min_map[target]:
                    self.min_map[target] = host
                if host > self.max_map[target]:
                    self.max_map[target] = host
            else:
                self.min_map[target] = host
                self.max_map[target] = host

    def get_bounds(self, target_element):
        """
        Gets the min and max of the target element
        If target element doesn't exist, raises an error
        :param target_element: element in the target column
        :return: min, max
        """
        return self.min_map[target_element], self.max_map[target_element]

    def get_size(self):
        """
        Gets the size of the min max map
        Size is defined as the number of "pointers", which is 3 * size of min map
        :return: size
        """
        return len(self.min_map) * 3
