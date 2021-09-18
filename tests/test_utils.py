"""
Test utils module
"""
import pytest
import numpy as np
from dimsm.utils import reshape_var, unravel_index, ravel_multi_index


@pytest.mark.parametrize("x", [np.random.randn(3, 2)])
@pytest.mark.parametrize("dim_index", [0, 1])
def test_reshape_var(x, dim_index):
    y = reshape_var(x, x.shape, dim_index)
    if dim_index == 0:
        assert np.allclose(x, y.T)
    else:
        assert np.allclose(x, y)


@pytest.mark.parametrize("x", [np.random.randn(3, 2)])
def test_reshape_var_reverse(x):
    y = reshape_var(x, x.shape, 0)
    z = reshape_var(y, x.shape, 0, reverse=True)

    assert np.allclose(x.ravel(), z)


@pytest.mark.parametrize("index", [15, 36, 27])
@pytest.mark.parametrize("dims", [(3, 4, 5), (4, 5, 6)])
def test_unravel_index(index, dims):
    assert np.allclose(unravel_index(index, dims),
                       np.unravel_index(index, dims))


@pytest.mark.parametrize("multi_index", [(1, 3, 1), (2, 1, 1)])
@pytest.mark.parametrize("dims", [(3, 4, 5), (4, 5, 6)])
def test_ravel_multi_index(multi_index, dims):
    assert np.isclose(ravel_multi_index(multi_index, dims),
                      np.ravel_multi_index(multi_index, dims))
