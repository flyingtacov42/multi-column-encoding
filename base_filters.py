from pybloomfilter import BloomFilter


class Filter:
    """
    Base class for all Filter-type objects
    """

    def __init__(self):
        pass

    def build_filter(self, matrix):
        pass

    def __contains__(self, item):
        return False

class BloomFilter2d(Filter):
    """
    Wrapper class that contains 2 bloom filters,
    1 for each column of data
    """

    def __init__(self, capacity, error_rate):
        super().__init__()
        self.bloom_filter_1 = BloomFilter(capacity, error_rate)
        self.bloom_filter_2 = BloomFilter(capacity, error_rate)

    def build_filter(self, matrix):
        for row in matrix:
            self.bloom_filter_1.add(row[0])
            self.bloom_filter_2.add(row[1])

    def __contains__(self, item):
        if item[0] in self.bloom_filter_1 and item[1] in self.bloom_filter_2:
            return True
        return False