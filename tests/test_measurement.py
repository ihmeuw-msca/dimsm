"""
Test measurement module
"""
import numpy as np
import pandas as pd
from scipy.sparse import diags
import pytest

from dimsm.dimension import Dimension
from dimsm.measurement import Measurement


def ad_jacobian(fun, x, out_shape=(), eps=1e-10):
    c = x + 0j
    g = np.zeros((*out_shape, *x.shape))
    if len(out_shape) == 0:
        for i in np.ndindex(x.shape):
            c[i] += eps*1j
            g[i] = fun(c).imag/eps
            c[i] -= eps*1j
    else:
        for j in np.ndindex(out_shape):
            for i in np.ndindex(x.shape):
                c[i] += eps*1j
                g[j][i] = fun(c)[j].imag/eps
                c[i] -= eps*1j
    return g


@pytest.fixture
def dims():
    return [Dimension("age", [20, 40, 60, 80, 100]),
            Dimension("year", [1990, 1995, 2000, 2005])]


@pytest.fixture
def data():
    return pd.DataFrame({
        "age": [25.0, 30.0, 35.0, 25.0, 30.0, 35.0],
        "year": [1991, 1991, 1991, 1993, 1993, 1993],
        "value": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    })


@pytest.fixture
def col_value():
    return "value"


@pytest.fixture
def imat():
    return diags(np.ones(6))


@pytest.fixture
def measurement(data, col_value, imat):
    return Measurement(data, col_value, imat)


def test_size(measurement, data):
    assert measurement.size == data.shape[0]


@pytest.mark.parametrize("data", [np.ones((5, 3))])
def test_data_error(data, col_value, imat):
    with pytest.raises(TypeError):
        Measurement(data, col_value, imat)


@pytest.mark.parametrize("col_value", ["random"])
def test_col_value_error(data, col_value, imat):
    with pytest.raises(ValueError):
        Measurement(data, col_value, imat)


@pytest.mark.parametrize("imat", [-1.0, 0.0,
                                  np.ones(10), -np.ones(6),
                                  np.ones((6, 3, 2)), np.ones((6, 3)),
                                  np.random.randn(6, 6),
                                  -np.diag(np.ones(6))])
def test_imat_error(data, col_value, imat):
    with pytest.raises(ValueError):
        Measurement(data, col_value, imat)


def test_update_dim(measurement, dims):
    measurement.update_dim(dims)
    assert measurement.mat.shape == (6, 20)
    assert np.allclose(np.sum(measurement.mat.toarray(), axis=1), 1.0)


def test_objective(measurement, dims):
    measurement.update_dim(dims)
    x = np.zeros(20)
    assert np.isclose(measurement.objective(x), 3.0)


def test_gradient(measurement, dims):
    measurement.update_dim(dims)
    x = np.zeros(20)
    my_gradient = measurement.gradient(x)
    tr_gradient = ad_jacobian(measurement.objective, x)
    assert np.allclose(my_gradient, tr_gradient)


def test_hessian(measurement, dims):
    measurement.update_dim(dims)
    x = np.zeros(20)
    my_hessian = measurement.hessian().toarray()
    tr_hessian = ad_jacobian(measurement.gradient, x, out_shape=(x.size,))
    assert np.allclose(my_hessian, tr_hessian)
