"""
Test process module
"""
from functools import partial
from numpy.core.fromnumeric import var
import pytest
import numpy as np
from dimsm.process import default_gen_mat, default_gen_vmat, Process
from dimsm.dimension import Dimension


def ad_jacobian(fun, x, shape, eps=1e-10):
    n = len(x)
    c = x + 0j
    g = np.zeros(shape)
    for i in np.ndindex(shape):
        c[i] += eps*1j
        g[i] = fun(c).imag/eps
        c[i] -= eps*1j
    return g


@pytest.fixture
def dim():
    return Dimension(name="age", grid=[20, 40, 60, 80])


@pytest.fixture
def order():
    return 1


@pytest.fixture
def gen_mat():
    return None


@pytest.fixture
def gen_vmat():
    return None


@pytest.fixture
def process(order, gen_mat, gen_vmat):
    return Process(order, gen_mat, gen_vmat)


@pytest.mark.parametrize("dt", [0.1, 0.5, 1.0])
def test_default_gen_mat(dt):
    mat = default_gen_mat(dt, 2)
    assert np.allclose(mat, np.array([[1.0, dt], [0.0, 1.0]]))


@pytest.mark.parametrize("dt", [0.1, 0.5, 1.0])
def test_default_gen_vmat(dt):
    vmat = default_gen_vmat(dt, 2)
    assert np.allclose(vmat, np.array([[dt**3/3, dt**2/2], [dt**2/2, dt]]))


@pytest.mark.parametrize("dt", [0.1, 0.5, 1.0])
def test_process_gen(process, dt):
    mat = process.gen_mat(dt)
    vmat = process.gen_vmat(dt)
    assert np.allclose(mat, np.array([[1.0, dt], [0.0, 1.0]]))
    assert np.allclose(vmat, np.array([[dt**3/3, dt**2/2], [dt**2/2, dt]]))


@pytest.mark.parametrize("order", [1.0, -1])
def test_process_order_error(order, gen_mat, gen_vmat):
    with pytest.raises((TypeError, ValueError)):
        Process(order, gen_mat, gen_vmat)


@pytest.mark.parametrize("gen_mat", [1.0])
def test_process_gen_mat_error(order, gen_mat, gen_vmat):
    with pytest.raises(TypeError):
        Process(order, gen_mat, gen_vmat)


@pytest.mark.parametrize("gen_vmat", [1.0])
def test_process_gen_vmat_error(order, gen_mat, gen_vmat):
    with pytest.raises(TypeError):
        Process(order, gen_mat, gen_vmat)


def test_process_gradient(process, dim):
    process.update_dim(dim)
    var_shape = (5, dim.size)
    dim_index = 1
    x = np.ones((process.order + 1, np.prod(var_shape)))

    my_gradient = process.gradient(x, var_shape, dim_index)
    tr_gradient = ad_jacobian(
        partial(process.objective, var_shape=var_shape, dim_index=dim_index),
        x,
        x.shape
    )

    assert np.allclose(my_gradient, tr_gradient)
