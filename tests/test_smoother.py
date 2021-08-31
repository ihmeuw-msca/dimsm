"""
Test smoother module
"""
import numpy as np
import pandas as pd
import pytest
from dimsm.dimension import Dimension
from dimsm.measurement import Measurement
from dimsm.prior import GaussianPrior, UniformPrior
from dimsm.process import Process
from dimsm.smoother import Smoother


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
    return [Dimension("age", [25, 30, 35, 40, 45, 50]),
            Dimension("year", [1990, 1995, 2000, 2005])]


@pytest.fixture
def meas():
    data = pd.DataFrame({
        "age": [25.0, 30.0, 35.0, 25.0, 30.0, 35.0],
        "year": [1990, 1990, 1990, 1995, 1995, 1995],
        "value": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    })
    return Measurement(data)


@pytest.fixture
def prcs():
    return {"age": Process(order=0),
            "year": Process(order=1)}


@pytest.fixture
def gpriors():
    return {"state": [GaussianPrior(mean=0.0, imat=1.0)],
            "year": [GaussianPrior(mean=0.0, imat=1.0)]}


@pytest.fixture
def upriors():
    return {"state": [UniformPrior(lb=-1.0, ub=1.0)],
            "year": [UniformPrior(lb=-1.0, ub=1.0)]}


@pytest.fixture
def smoother(dims, meas, prcs, gpriors, upriors):
    return Smoother(dims, meas, prcs, gpriors, upriors)


def test_smoother_dim_names(smoother):
    assert smoother.dim_names == ["age", "year"]


def test_smoother_var_shape(smoother):
    assert smoother.var_shape == (6, 4)


def test_smoother_prc_names(smoother):
    assert smoother.prc_names == ["age", "year"]


def test_smoother_num_vars(smoother):
    assert smoother.num_vars == 2


def test_smoother_num_dims(smoother):
    assert smoother.num_dims == 2


def test_smoother_var_indices(smoother):
    var_indices = smoother.var_indices
    assert all(name in var_indices for name in ["state", "age", "year"])
    assert var_indices["state"] == [0]
    assert var_indices["age"] == []
    assert var_indices["year"] == [1]


def test_gradient(smoother):
    x = np.zeros(smoother.num_vars*smoother.var_size)
    my_gradient = smoother.gradient(x)
    tr_gradient = ad_jacobian(smoother.objective, x)
    assert np.allclose(my_gradient, tr_gradient)


def test_hessian(smoother):
    x = np.zeros(smoother.num_vars*smoother.var_size)
    my_hessian = smoother.hessian().toarray()
    tr_hessian = ad_jacobian(smoother.gradient, x, out_shape=(x.size,))
    assert np.allclose(my_hessian, tr_hessian)
