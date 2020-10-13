from time import perf_counter
import pytest

import numpy as np
import lttbc

ARRAY_SIZE = 1000
THRESHOLD = 100
LARGE_ARRAY = 1000000
LARGE_THRESHOLD = 10000


def test_downsample_uint64():
    """Test the base down sampling of the module"""
    x = np.arange(ARRAY_SIZE, dtype='int32')
    y = np.random.randint(1000, size=ARRAY_SIZE, dtype='uint64')
    nx, ny = lttbc.downsample(x, y, THRESHOLD)
    assert len(nx) == THRESHOLD
    assert len(ny) == THRESHOLD
    assert nx.dtype == np.double
    assert ny.dtype == np.double


def test_downsample_bool():
    """Test the down sampling with boolean types"""
    x = np.arange(ARRAY_SIZE, dtype='int32')
    y = np.array([True] * ARRAY_SIZE, dtype=np.bool)
    nx, ny = lttbc.downsample(x, y, THRESHOLD)
    assert len(nx) == THRESHOLD
    assert len(ny) == THRESHOLD
    assert nx.dtype == np.double
    assert ny.dtype == np.double
    test_array = np.array([1.0] * THRESHOLD, dtype=np.float)
    test_array_bool = np.array([1.0] * THRESHOLD, dtype=np.bool)
    np.testing.assert_array_almost_equal(ny, test_array)
    np.testing.assert_array_almost_equal(ny, test_array_bool)


def test_inf():
    """Test the down sampling with inf types"""
    x = np.arange(ARRAY_SIZE, dtype='int32')
    y = np.array([np.inf] * ARRAY_SIZE, dtype=np.float)
    nx, ny = lttbc.downsample(x, y, THRESHOLD)
    assert len(nx) == THRESHOLD
    assert len(ny) == THRESHOLD
    assert nx.dtype == np.double
    assert ny.dtype == np.double
    test_array = np.array([0.0] * THRESHOLD, dtype=np.float)
    np.testing.assert_array_almost_equal(ny, test_array)


def test_nan():
    """Test the down sampling with NaN types"""
    x = np.arange(ARRAY_SIZE, dtype='int32')
    y = np.array([np.nan] * ARRAY_SIZE, dtype=np.float)
    nx, ny = lttbc.downsample(x, y, THRESHOLD)
    assert len(nx) == THRESHOLD
    assert len(ny) == THRESHOLD
    assert nx.dtype == np.double
    assert ny.dtype == np.double
    test_array = np.array([0.0] * THRESHOLD, dtype=np.float)
    np.testing.assert_array_almost_equal(ny, test_array)


def test_array_size():
    """Test the input failure for different dimensions of arrays"""
    x = np.arange(ARRAY_SIZE)
    y = np.random.randint(1000, size=ARRAY_SIZE - 1, dtype='uint64')
    with pytest.raises(Exception):
        assert lttbc.downsample(x, y, ARRAY_SIZE)


def test_benchmark():
    """Basic skeletton benchmark test for the down sample algorithm"""
    x = np.arange(LARGE_ARRAY, dtype=np.int32)
    y = np.arange(LARGE_ARRAY, dtype=np.float)

    def sample():
        nx, ny = lttbc.downsample(x, y, LARGE_THRESHOLD)
        return nx, ny

    t_start = perf_counter()
    nx, ny = sample()
    elapsed = perf_counter() - t_start

    assert len(nx) == LARGE_THRESHOLD
    assert len(ny) == LARGE_THRESHOLD
    assert nx.dtype == np.double
    assert ny.dtype == np.double

    assert elapsed < 0.1


def test_negative_threshold():
    """Test if a negative threshold provides problems"""
    x = np.arange(ARRAY_SIZE, dtype='int32')
    y = np.array([1] * ARRAY_SIZE, dtype=np.float)
    nx, ny = lttbc.downsample(x, y, -THRESHOLD)
    assert len(nx) == ARRAY_SIZE
    assert len(ny) == ARRAY_SIZE
    assert nx.dtype == np.double
    assert ny.dtype == np.double
    test_array = np.array([1.0] * ARRAY_SIZE, dtype=np.float)
    np.testing.assert_array_almost_equal(ny, test_array)


def test_array_mix():
    """Test mix of problematic input 'inf' and 'nan'"""

    x = np.arange(20, dtype='int32')
    y = np.array([0.0, 1.0, 2.0, np.nan, 4.0, 5.0, 6.0, np.nan, np.inf,
                  np.inf, 10.0, np.nan, 12.0, -np.inf, 14.0, 15.0, 16.0, 17.0,
                  np.nan, 19.0], dtype=np.float)
    nx, ny = lttbc.downsample(x, y, 10)
    assert len(nx) == 10
    assert len(ny) == 10
    assert nx.dtype == np.double
    assert ny.dtype == np.double
    test_array = np.array(
        [0., 0., 4., 4., 4., 10., -np.inf, -np.inf, -np.inf, 19.],
        dtype=np.float)
    np.testing.assert_array_almost_equal(ny, test_array)


def test_singe_nan():
    """Test single 'nan' input for down sampling"""
    x = np.arange(20, dtype='int32')
    y = np.array([0.0, 1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0,
                  9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0,
                  18.0, 19.0], dtype=np.float)
    nx, ny = lttbc.downsample(x, y, 10)
    assert len(nx) == 10
    assert len(ny) == 10
    assert nx.dtype == np.double
    assert ny.dtype == np.double
    test_array = np.array(
        [0., 0., 4., 5., 7., 10., 12., 14., 16., 19.],
        dtype=np.float)
    np.testing.assert_array_almost_equal(ny, test_array)


def test_singe_inf():
    """Test single 'inf' input for down sampling

    XXX: Apparently infinite values provide a crappy result...
    """
    x = np.arange(20, dtype='int32')
    y = np.array([0.0, 1.0, 2.0, np.inf, 4.0, 5.0, 6.0, 7.0, 8.0,
                  9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0,
                  18.0, 19.0], dtype=np.float)
    nx, ny = lttbc.downsample(x, y, 10)
    assert len(nx) == 10
    assert len(ny) == 10
    assert nx.dtype == np.double
    assert ny.dtype == np.double
    test_array = np.array(
        [0., 1., np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 19.],
        dtype=np.float)
    np.testing.assert_array_almost_equal(ny, test_array)
    assert False
