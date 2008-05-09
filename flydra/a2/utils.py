import numpy

class FastFinder(object):
    """allows fast searching by use of a cached, sorted copy of the original data

    See tests.TestUtils.test_fast_finder() for an example.
    """
    def __init__(self,values1d):
        values1d = numpy.atleast_1d( values1d )
        assert len(values1d.shape)==1, 'only 1D arrays supported'
        self.idxs = numpy.argsort( values1d )
        self.sorted = values1d[ self.idxs ]
    def get_idxs_of_equal(self,testval):
        """performs fast search on sorted data"""
        testval = numpy.asarray(testval)
        assert len( testval.shape)==0, 'can only find equality of a scalar'

        left_idxs = self.sorted.searchsorted( testval, side='left' )
        right_idxs = self.sorted.searchsorted( testval, side='right' )

        this_idxs = self.idxs[left_idxs:right_idxs]
        return this_idxs
    def get_first_idx_of_assumed_equal(self,testval):
        """performs fast search on sorted data"""
        testval = numpy.asarray(testval)
        assert len( testval.shape)==0, 'can only find equality of a scalar'

        left_idx = self.sorted.searchsorted( testval, side='left' )

        this_idx = self.idxs[left_idx]
        return this_idx
