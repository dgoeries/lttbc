from time import perf_counter

import numpy as np
import lttbc

ARRAY_SIZE = 1000
THRESHOLD = 100


def test_downsample_uint64():
    """Test the base down sampling of the module"""
    x = np.arange(ARRAY_SIZE, dtype='int32')
    y = np.random.randint(1000, size=ARRAY_SIZE, dtype='uint64')
    nx, ny = lttbc.downsample(x, y, THRESHOLD)
    assert len(nx) == THRESHOLD
    assert len(ny) == THRESHOLD
    assert nx.dtype == np.float
    assert ny.dtype == np.float


def test_downsample_bool():
    """Test the down sampling with boolean types"""
    x = np.arange(ARRAY_SIZE, dtype='int32')
    y = np.array([True] * ARRAY_SIZE, dtype=np.bool)
    nx, ny = lttbc.downsample(x, y, THRESHOLD)
    assert len(nx) == THRESHOLD
    assert len(ny) == THRESHOLD
    assert nx.dtype == np.float
    assert ny.dtype == np.float
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
    assert nx.dtype == np.float
    assert ny.dtype == np.float
    test_array = np.array([0.0] * THRESHOLD, dtype=np.float)
    np.testing.assert_array_almost_equal(ny, test_array)


def test_nan():
    """Test the down sampling with NaN types"""
    x = np.arange(ARRAY_SIZE, dtype='int32')
    y = np.array([np.nan] * ARRAY_SIZE, dtype=np.float)
    nx, ny = lttbc.downsample(x, y, THRESHOLD)
    assert len(nx) == THRESHOLD
    assert len(ny) == THRESHOLD
    assert nx.dtype == np.float
    assert ny.dtype == np.float
    test_array = np.array([0.0] * THRESHOLD, dtype=np.float)
    np.testing.assert_array_almost_equal(ny, test_array)


def test_array_size():
    """Test the input failure for different dimensions of arrays"""
    fail = False
    x = np.arange(ARRAY_SIZE)
    y = np.random.randint(1000, size=ARRAY_SIZE - 1, dtype='uint64')
    try:
        lttbc.downsample(x, y, ARRAY_SIZE)
    except Exception:
        fail = True

    assert fail


def test_benchmark():
    """Basic skeletton benchmark test for the down sample algorithm"""
    LARGE_ARRAY = 1000000
    SAMPLE_SIZE = 10000
    x = np.arange(LARGE_ARRAY, dtype=np.int32)
    y = np.arange(LARGE_ARRAY, dtype=np.float)

    def sample():
        nx, ny = lttbc.downsample(x, y, SAMPLE_SIZE)
        return nx, ny

    t_start = perf_counter()
    sample()
    elapsed = perf_counter() - t_start
    assert elapsed < 0.1
