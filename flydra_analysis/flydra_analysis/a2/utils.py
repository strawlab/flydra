from __future__ import absolute_import
import numpy as np
import time, warnings

import flydra_fastfinder_help

from flydra_analysis.a2.missing_value_error import MissingValueError

class FastFinder(object):
    """fast search by use of a cached, sorted copy of the original data

    Parameters
    ----------
    values1d : 1D array
      The input data which is sorted and stored for indexing
    """

    def __init__(self, values1d):
        values1d = np.atleast_1d(values1d)
        assert values1d.ndim == 1, "only 1D arrays supported"
        self.idxs = np.argsort(values1d)
        self.sorted = values1d[self.idxs]
        self.values = values1d

    def get_idxs_of_equal(self, testval):
        """performs fast search for scalar

        Parameters
        ----------
        testval : scalar
          The value to find the indices of

        Returns
        -------
        result : array
          The indices into the original values1d array

        Examples
        --------

        >>> a = np.array([ 1, 2, 3, 3, 2, 1, 2.3 ])
        >>> af = FastFinder(a)
        >>> bs = [ 0, 1, 2, 1.1 ]
        >>> for b in bs:
        ...     sorted(af.get_idxs_of_equal(b))
        ...
        []
        [0, 5]
        [1, 4]
        []

        """
        testval = np.asarray(testval)
        assert len(testval.shape) == 0, "can only find equality of a scalar"

        left_idx = self.sorted.searchsorted(testval, side="left")
        right_idx = self.sorted.searchsorted(testval, side="right")

        this_idxs = self.idxs[left_idx:right_idx]
        return this_idxs

    def get_idx_of_equal(self, testvals, missing_ok=False):

        # XXX should test dtype of testvals and self.values and call
        # appropriate helper function.

        return flydra_fastfinder_help.get_first_idx_double(
            self.values.astype(np.float64),
            np.asanyarray(testvals).astype(np.float64),
            missing_ok=missing_ok,
        )

    def get_idx_of_equal_slow(self, testvals):
        """performs fast search for vector

        Fails unless there is one and only one equal value in the
        searched data.

        Parameters
        ----------
        testval : scalar
          The value to find the indices of

        Returns
        -------
        result : array
          The indices into the original values1d array

        Examples
        --------

        >>> a = np.array([ 10, 0, 2, 3, 3, 2.1, 1, 2.3 ])
        >>> af = FastFinder(a)
        >>> bs = [ 0, 1, 2 ]
        >>> af.get_idx_of_equal(bs).tolist()
        [1, 6, 2]

        """
        warnings.warn("slow implementation of get_idx_of_equal()")
        tmp_result = [self.get_idxs_of_equal(t) for t in testvals]
        for i, t in enumerate(tmp_result):
            assert len(t) == 1
        result = [t[0] for t in tmp_result]
        return np.array(result)

    def get_idxs_in_range(self, low, high):
        """performs fast search on sorted data

        Parameters
        ----------
        low : scalar
          The low value to find the indices of
        high : scalar
          This high value to find the indices of

        Returns
        -------
        result : array
          The indices into the original values1d array. May not be sorted.

        Examples
        --------

        >>> a = np.array([ 1, 2, 3, 3, 2, 1, 2.3 ])
        >>> af = FastFinder(a)
        >>> bs = [ (0,1.1), (0,1), (0,2), (1.1,5), (2,2.1), (2,3) ]
        >>> for low,high in bs:
        ...     list = af.get_idxs_in_range(low,high).tolist()
        ...     list.sort()
        ...     list
        [0, 5]
        [0, 5]
        [0, 1, 4, 5]
        [1, 2, 3, 4, 6]
        [1, 4]
        [1, 2, 3, 4, 6]

        """
        low = np.asarray(low)
        assert len(low.shape) == 0, "can only find equality of a scalar"

        high = np.asarray(high)
        assert len(high.shape) == 0, "can only find equality of a scalar"

        low_idx = self.sorted.searchsorted(low, side="left")
        high_idx = self.sorted.searchsorted(high, side="right")

        this_idxs = self.idxs[low_idx:high_idx]
        return this_idxs


def iter_contig_chunk_idxs(arr):
    if len(arr) == 0:
        return
    # ADS print 'arr',arr
    diff = arr[1:] - arr[:-1]
    # ADS print 'diff',diff
    non_one = diff != 1
    # ADS print 'non_one',non_one

    non_one = np.ma.array(non_one).filled()
    # ADS print 'non_one',non_one

    idxs = np.nonzero(non_one)[0]
    # ADS print 'idxs',idxs
    prev_idx = 0
    for idx in idxs:
        next_idx = idx + 1
        # ADS print 'idx, prev_idx, next_idx',idx, prev_idx, next_idx
        data_chunk = np.ma.array(arr[prev_idx:next_idx])
        # ADS print 'data_chunk',data_chunk
        if not np.any(data_chunk.mask):
            yield (prev_idx, next_idx)
        else:
            # ADS print 'skipped!'
            pass
        prev_idx = next_idx
    yield (prev_idx, len(arr))


def get_contig_chunk_idxs(arr):
    """get indices of contiguous chunks

    Parameters
    ----------
    arr : 1D array

    Returns
    -------
    list_of_startstops : list of 2-tuples
        A list of tuples, where each tuple is (start_idx,stop_idx) of arr

    Examples
    --------

    >>> a = np.array([ 1, 2, 3, 4, 10, 11, 12, -1, -2, -3, -2, -1, 0, 1])
    >>> list_of_start_stops = get_contig_chunk_idxs(a)
    >>> list_of_start_stops
    [(0, 4), (4, 7), (7, 8), (8, 9), (9, 14)]
    >>> for start,stop in list_of_start_stops:
    ...     print(a[start:stop])
    ...
    [1 2 3 4]
    [10 11 12]
    [-1]
    [-2]
    [-3 -2 -1  0  1]

    """
    return [start_stop for start_stop in iter_contig_chunk_idxs(arr)]


def test_get_contig_chunk_idxs_1():
    input = np.array([1, 2, 3, 4, 5, 11, 12, 13, 14, 15, 16, 0, 5, -1], dtype=float)
    expected = [
        (0, 5),
        (5, 11),
        (11, 12),
        (12, 13),
        (13, 14),
    ]
    actual = get_contig_chunk_idxs(input)
    assert len(expected) == len(actual)
    for i in range(len(expected)):
        start, stop = expected[i]
        assert (start, stop) == actual[i]


def test_get_contig_chunk_idxs_2():
    input = np.array(
        [np.nan, np.nan, 3, 4, 5, np.nan, 12, 13, 14, 15, 16, 0, 5, -1], dtype=float
    )
    input = np.ma.masked_where(np.isnan(input), input)
    expected = [
        (2, 5),
        (6, 11),
        (11, 12),
        (12, 13),
        (13, 14),
    ]
    actual = get_contig_chunk_idxs(input)
    assert len(expected) == len(actual)
    for i in range(len(expected)):
        start, stop = expected[i]
        assert (start, stop) == actual[i]


def test_get_contig_chunk_idxs_empty():
    input = np.array([])
    expected = []
    actual = get_contig_chunk_idxs(input)
    assert len(expected) == len(actual)
    for i in range(len(expected)):
        start, stop = expected[i]
        assert (start, stop) == actual[i]


def test_fast_finder():
    a = np.array([1, 2, 3, 3, 2, 1, 2.3])
    bs = [0, 1, 2, 1.1]
    af = FastFinder(a)
    for b in bs:
        idxs1 = sorted(af.get_idxs_of_equal(b))
        idxs2 = sorted(np.nonzero(a == b)[0])
        assert len(idxs1) == len(idxs2)
        assert np.allclose(idxs1, idxs2)


def iter_non_overlapping_chunk_start_stops(
    arr, min_chunk_size=10000, size_increment=10, status_fd=None
):

    # This is a relatively dumb implementation that I think could be
    # massively sped up.

    start = 0
    cur_stop = start + min_chunk_size

    while 1:
        if status_fd is not None:
            tstart = time.time()

            status_fd.write("Computing non-overlapping chunks...")
            status_fd.flush()

        if cur_stop >= len(arr):
            cur_stop = len(arr)
            yield (start, cur_stop)
            return

        while 1:
            # inner loop - keep incrementing cur_stop until condition passes
            arr_pre = arr[start:cur_stop]
            arr_post = arr[cur_stop:]
            premax = arr_pre.max()
            postmin = arr_post.min()
            if premax < postmin:
                # condition passed
                break
            cur_stop += size_increment
            if cur_stop >= len(arr):
                # end of array reached - pass by definition
                cur_stop = len(arr)
                break

        # If we are here, the condition passed by definition.
        if status_fd is not None:
            status_fd.write(
                "did %d rows in %.1f sec.\n"
                % (cur_stop - start, (time.time() - tstart))
            )
            status_fd.flush()
        yield (start, cur_stop)
        if cur_stop >= len(arr):
            break
        start = cur_stop
        cur_stop = start + min_chunk_size


def test_iter_non_overlapping_chunk_start_stops():

    a = np.array(
        [1, 1, 1, 2, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 4, 4, 6, 6, 4, 6]
    )

    b = [(0, 8), (8, 12), (12, 25)]

    curmax = -np.inf
    thislen = 0
    for i, bi in enumerate(
        iter_non_overlapping_chunk_start_stops(a, min_chunk_size=3, size_increment=1)
    ):
        this_chunk = a[bi[0] : bi[1]]
        # print '%d this_chunk: %s'%(i,this_chunk)
        assert b[i] == bi
        thismax = np.max(this_chunk)
        assert thismax > curmax
        curmax = thismax
        thislen += 1
    assert len(b) == thislen


def test_iter_non_overlapping_chunk_start_stops2():
    a = np.arange(2365964, dtype=np.uint64)

    for start, stop in iter_non_overlapping_chunk_start_stops(
        a, min_chunk_size=2000000, size_increment=1000
    ):
        # print 'start,stop',start,stop
        assert stop > start


def test_get_idx_of_equal():
    a = np.array([10, 0, 2, 3, 3, 2.1, 1, 2.3])
    af = FastFinder(a)
    bs = [0, 1, 2]
    actual = af.get_idx_of_equal(bs)
    expected = np.array([1, 6, 2])
    assert np.allclose(actual, expected)


def test_get_idx_of_equal_ints():
    a = np.array([10, 0, 2, 3, 3, 2, 1, -2])
    af = FastFinder(a)
    bs = [0, 1, 2, -2]
    actual = af.get_idx_of_equal(bs)
    expected = np.array([1, 6, 2, 7])
    assert np.allclose(actual, expected)


def test_get_idx_of_equal_missing():
    a = np.array([10, 0, 2, 3, 3, 2.1, 1, 2.3])
    af = FastFinder(a)
    bs = [0, 1, 2, 3.4]
    try:
        af.get_idx_of_equal(bs)
    except MissingValueError:
        # expected error
        pass
    else:
        raise
    actual = af.get_idx_of_equal(bs, missing_ok=True)
    expected = np.array([1, 6, 2, -1])
    assert np.allclose(actual, expected)
