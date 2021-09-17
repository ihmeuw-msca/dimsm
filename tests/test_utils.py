"""
Test utils module
"""
import pytest
import numpy as np
from dimsm.utils import reshape_var


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
