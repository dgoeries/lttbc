import sys
from time import perf_counter

import numpy as np
import pytest

import lttbc

ARRAY_SIZE = 1000
THRESHOLD = 100
LARGE_ARRAY = 1000000
LARGE_THRESHOLD = 10000


def test_input_wrong_x_y():
    """Test the down sampling with wrong input types for x or/and y"""
    x = 1
    y = np.array([True] * ARRAY_SIZE, dtype=bool)
    with pytest.raises(TypeError):
        lttbc.downsample(x, y, THRESHOLD)

    x = np.array([True] * ARRAY_SIZE, dtype=bool)
    y = 4
    with pytest.raises(TypeError):
        lttbc.downsample(x, y, THRESHOLD)

    x = "wrong"
    y = np.array([True] * ARRAY_SIZE, dtype=bool)
    with pytest.raises(TypeError):
        lttbc.downsample(x, y, THRESHOLD)

    x = np.array([True] * ARRAY_SIZE, dtype=bool)
    y = "wrong"
    with pytest.raises(TypeError):
        lttbc.downsample(x, y, THRESHOLD)

    x = 1
    y = "wrong"
    with pytest.raises(TypeError):
        lttbc.downsample(x, y, THRESHOLD)


def test_single_dimension_validation():
    """Test that the downsample algorithm rejects arrays with multiple dims"""
    x = np.array([[0., 0.], [1., 0.8], [0.9, 0.8], [0.9, 0.7], [0.9, 0.6],
                  [0.8, 0.5], [0.8, 0.5], [0.7, 0.5], [0.1, 0.], [0., 0.]],
                 dtype=np.double)
    assert x.shape == (10, 2)
    assert x.ndim == 2

    y = np.array([True] * ARRAY_SIZE, dtype=bool)
    with pytest.raises(ValueError):
        lttbc.downsample(x, y, THRESHOLD)


def test_negative_threshold():
    """Test if a negative threshold provides problems"""
    x = np.arange(ARRAY_SIZE, dtype=np.int32)
    y = np.random.randint(1000, size=ARRAY_SIZE, dtype=np.uint64)
    assert sys.getrefcount(x) == 2
    assert sys.getrefcount(y) == 2
    nx, ny = lttbc.downsample(x, y, -THRESHOLD)
    assert len(nx) == ARRAY_SIZE
    assert len(ny) == ARRAY_SIZE
    assert nx.dtype == np.double
    assert ny.dtype == np.double
    assert sys.getrefcount(x) == 2
    assert sys.getrefcount(y) == 2
    assert sys.getrefcount(nx) == 2
    assert sys.getrefcount(ny) == 2

    np.testing.assert_array_almost_equal(ny, y)


def test_threshold_larger():
    """Test if a larger threshold provides problems"""
    x = np.arange(ARRAY_SIZE, dtype=np.int32)
    y = np.random.randint(1000, size=ARRAY_SIZE, dtype=np.uint64)
    assert sys.getrefcount(x) == 2
    assert sys.getrefcount(y) == 2
    # Will return the arrays!
    nx, ny = lttbc.downsample(x, y, ARRAY_SIZE + 1)
    assert len(nx) == ARRAY_SIZE
    assert len(ny) == ARRAY_SIZE
    assert nx.dtype == np.double
    assert ny.dtype == np.double
    assert sys.getrefcount(x) == 2
    assert sys.getrefcount(y) == 2
    assert sys.getrefcount(nx) == 2
    assert sys.getrefcount(ny) == 2

    # NOTE: Known feature, we return double arrays ...
    np.testing.assert_array_almost_equal(nx, x)
    np.testing.assert_array_almost_equal(ny, y)


def test_input_list():
    """Test the down sampling with lists types"""
    x = list(range(ARRAY_SIZE))
    y = [True] * ARRAY_SIZE
    assert sys.getrefcount(x) == 2
    assert sys.getrefcount(y) == 2
    nx, ny = lttbc.downsample(x, y, THRESHOLD)
    assert len(nx) == THRESHOLD
    assert len(ny) == THRESHOLD
    assert nx.dtype == np.double
    assert ny.dtype == np.double
    assert sys.getrefcount(x) == 2
    assert sys.getrefcount(y) == 2
    assert sys.getrefcount(nx) == 2
    assert sys.getrefcount(ny) == 2
    test_array = np.array([1.0] * THRESHOLD, dtype=np.float64)
    test_array_bool = np.array([1.0] * THRESHOLD, dtype=bool)
    np.testing.assert_array_almost_equal(ny, test_array)
    np.testing.assert_array_almost_equal(ny, test_array_bool)


def test_input_list_array():
    """Test the down sampling with mixed types"""
    x = list(range(ARRAY_SIZE))
    y = np.array([True] * ARRAY_SIZE, dtype=bool)
    assert sys.getrefcount(x) == 2
    assert sys.getrefcount(y) == 2
    nx, ny = lttbc.downsample(x, y, THRESHOLD)
    assert len(nx) == THRESHOLD
    assert len(ny) == THRESHOLD
    assert nx.dtype == np.double
    assert ny.dtype == np.double
    assert sys.getrefcount(x) == 2
    assert sys.getrefcount(y) == 2
    assert sys.getrefcount(nx) == 2
    assert sys.getrefcount(ny) == 2
    test_array = np.array([1.0] * THRESHOLD, dtype=np.float64)
    test_array_bool = np.array([1.0] * THRESHOLD, dtype=bool)
    np.testing.assert_array_almost_equal(ny, test_array)
    np.testing.assert_array_almost_equal(ny, test_array_bool)


def test_array_size():
    """Test the input failure for different dimensions of arrays"""
    x = np.arange(ARRAY_SIZE)
    y = np.random.randint(1000, size=ARRAY_SIZE - 1, dtype=np.uint64)
    assert sys.getrefcount(x) == 2
    assert sys.getrefcount(y) == 2
    with pytest.raises(ValueError):
        assert lttbc.downsample(x, y, ARRAY_SIZE)
    assert sys.getrefcount(x) == 2
    assert sys.getrefcount(y) == 2


def test_downsample_uint64():
    """Test the base down sampling of the module"""
    x = np.arange(ARRAY_SIZE, dtype=np.int32)
    y = np.random.randint(1000, size=ARRAY_SIZE, dtype=np.uint64)
    assert sys.getrefcount(x) == 2
    assert sys.getrefcount(y) == 2
    nx, ny = lttbc.downsample(x, y, THRESHOLD)
    assert len(nx) == THRESHOLD
    assert len(ny) == THRESHOLD
    assert nx.dtype == np.double
    assert ny.dtype == np.double
    assert sys.getrefcount(x) == 2
    assert sys.getrefcount(y) == 2
    assert sys.getrefcount(nx) == 2
    assert sys.getrefcount(ny) == 2


def test_downsample_bool():
    """Test the down sampling with boolean types"""
    x = np.arange(ARRAY_SIZE, dtype=np.int32)
    y = np.array([True] * ARRAY_SIZE, dtype=bool)
    assert sys.getrefcount(x) == 2
    assert sys.getrefcount(y) == 2
    nx, ny = lttbc.downsample(x, y, THRESHOLD)
    assert len(nx) == THRESHOLD
    assert len(ny) == THRESHOLD
    assert nx.dtype == np.double
    assert ny.dtype == np.double
    assert sys.getrefcount(x) == 2
    assert sys.getrefcount(y) == 2
    assert sys.getrefcount(nx) == 2
    assert sys.getrefcount(ny) == 2
    test_array = np.array([1.0] * THRESHOLD, dtype=np.float64)
    test_array_bool = np.array([1.0] * THRESHOLD, dtype=bool)
    np.testing.assert_array_almost_equal(ny, test_array)
    np.testing.assert_array_almost_equal(ny, test_array_bool)


def test_inf():
    """Test the down sampling with inf types"""
    x = np.arange(ARRAY_SIZE, dtype=np.int32)
    y = np.array([np.inf] * ARRAY_SIZE, dtype=np.float64)
    assert sys.getrefcount(x) == 2
    assert sys.getrefcount(y) == 2
    nx, ny = lttbc.downsample(x, y, THRESHOLD)
    assert len(nx) == THRESHOLD
    assert len(ny) == THRESHOLD
    assert nx.dtype == np.double
    assert ny.dtype == np.double
    assert sys.getrefcount(x) == 2
    assert sys.getrefcount(y) == 2
    assert sys.getrefcount(nx) == 2
    assert sys.getrefcount(ny) == 2
    test_array = np.array([0.0] * THRESHOLD, dtype=np.float64)
    np.testing.assert_array_almost_equal(ny, test_array)


def test_nan():
    """Test the down sampling with NaN types"""
    x = np.arange(ARRAY_SIZE, dtype=np.int32)
    y = np.array([np.nan] * ARRAY_SIZE, dtype=np.float64)
    assert sys.getrefcount(x) == 2
    assert sys.getrefcount(y) == 2
    nx, ny = lttbc.downsample(x, y, THRESHOLD)
    assert len(nx) == THRESHOLD
    assert len(ny) == THRESHOLD
    assert nx.dtype == np.double
    assert ny.dtype == np.double
    assert sys.getrefcount(x) == 2
    assert sys.getrefcount(y) == 2
    assert sys.getrefcount(nx) == 2
    assert sys.getrefcount(ny) == 2
    test_array = np.array([0.0] * THRESHOLD, dtype=np.float64)
    np.testing.assert_array_almost_equal(ny, test_array)


def test_benchmark():
    """Basic skeletton benchmark test for the down sample algorithm"""
    x = np.arange(LARGE_ARRAY, dtype=np.int32)
    y = np.arange(LARGE_ARRAY, dtype=np.float64)
    assert sys.getrefcount(x) == 2
    assert sys.getrefcount(y) == 2

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
    assert sys.getrefcount(x) == 2
    assert sys.getrefcount(y) == 2
    assert sys.getrefcount(nx) == 2
    assert sys.getrefcount(ny) == 2

    assert elapsed < 0.1


def test_array_mix_inf_nan():
    """Test mix of problematic input 'inf' and 'nan'"""

    x = np.arange(20, dtype=np.int32)
    y = np.array([0.0, 1.0, 2.0, np.nan, 4.0, 5.0, 6.0, np.nan, np.inf,
                  np.inf, 10.0, np.nan, 12.0, -np.inf, 14.0, 15.0, 16.0, 17.0,
                  np.nan, 19.0], dtype=np.float64)
    assert sys.getrefcount(x) == 2
    assert sys.getrefcount(y) == 2
    nx, ny = lttbc.downsample(x, y, 10)
    assert len(nx) == 10
    assert len(ny) == 10
    assert nx.dtype == np.double
    assert ny.dtype == np.double
    assert sys.getrefcount(x) == 2
    assert sys.getrefcount(y) == 2
    assert sys.getrefcount(nx) == 2
    assert sys.getrefcount(ny) == 2
    test_array = np.array(
        [0., 0., 4., 4., 4., 10., -np.inf, -np.inf, -np.inf, 19.],
        dtype=np.float64)
    np.testing.assert_array_almost_equal(ny, test_array)


def test_single_nan():
    """Test single 'nan' input for down sampling"""
    x = np.arange(20, dtype=np.int32)
    y = np.array([0.0, 1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0,
                  9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0,
                  18.0, 19.0], dtype=np.float64)
    assert sys.getrefcount(x) == 2
    assert sys.getrefcount(y) == 2
    nx, ny = lttbc.downsample(x, y, 10)
    assert len(nx) == 10
    assert len(ny) == 10
    assert nx.dtype == np.double
    assert ny.dtype == np.double
    assert sys.getrefcount(x) == 2
    assert sys.getrefcount(y) == 2
    assert sys.getrefcount(nx) == 2
    assert sys.getrefcount(ny) == 2
    test_array = np.array(
        [0., 0., 4., 5., 7., 10., 12., 14., 16., 19.],
        dtype=np.float64)
    np.testing.assert_array_almost_equal(ny, test_array)


def test_single_inf():
    """Test single 'inf' input for down sampling

    XXX: Apparently infinite values provide a crappy result...
    """
    x = np.arange(20, dtype=np.int32)
    y = np.array([0.0, 1.0, 2.0, np.inf, 4.0, 5.0, 6.0, 7.0, 8.0,
                  9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0,
                  18.0, 19.0], dtype=np.float64)
    assert sys.getrefcount(x) == 2
    assert sys.getrefcount(y) == 2
    nx, ny = lttbc.downsample(x, y, 10)
    assert len(nx) == 10
    assert len(ny) == 10
    assert nx.dtype == np.double
    assert ny.dtype == np.double
    assert sys.getrefcount(x) == 2
    assert sys.getrefcount(y) == 2
    assert sys.getrefcount(nx) == 2
    assert sys.getrefcount(ny) == 2
    test_array = np.array(
        [0., 1., np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 19.],
        dtype=np.float64)
    np.testing.assert_array_almost_equal(ny, test_array)
