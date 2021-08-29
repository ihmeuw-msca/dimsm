"""
Test prior module
"""
import pytest
import numpy as np
from dimsm.prior import GaussianPrior, UniformPrior, extend_info


def ad_jacobian(fun, x, shape, eps=1e-10):
    n = len(x)
    c = x + 0j
    g = np.zeros(shape)
    for i in np.ndindex(shape):
        c[i] += eps*1j
        g[i] = fun(c).imag/eps
        c[i] -= eps*1j
    return g


@pytest.mark.parametrize(("info", "size"),
                         [(np.ones((2, 2, 2)), 2),
                          (np.ones(3), 4),
                          (np.ones(3), 1)])
def test_extend_info_error(info, size):
    with pytest.raises(ValueError):
        extend_info(info, size)


@pytest.mark.parametrize("info", [np.ones(1), np.ones((1, 1))])
@pytest.mark.parametrize("size", [1, 2, 3])
def test_extend_info(info, size):
    extended_info = extend_info(info, size)
    assert extended_info.shape[0] == size
    if info.ndim == 1:
        assert np.allclose(extended_info, info[0])
    else:
        assert np.allclose(extended_info.diagonal(), info[0, 0])


@pytest.mark.parametrize("mean", [1.0, np.ones(5)])
@pytest.mark.parametrize("imat", [1.0, np.ones(5), np.identity(5)])
@pytest.mark.parametrize("mat", [np.ones((5, 3))])
@pytest.mark.parametrize("size", [None, 5])
def test_gprior(mean, imat, mat, size):
    gprior = GaussianPrior(mean, imat, mat, size)
    assert gprior.size == 5


@pytest.mark.parametrize("mean", [1.0])
@pytest.mark.parametrize("imat", [1.0])
@pytest.mark.parametrize("size", [5])
def test_gprior_update_size(mean, imat, size):
    gprior = GaussianPrior(mean, imat)
    gprior.update_size(size)
    assert gprior.size == size
    assert gprior.mean.shape[0] == size
    assert gprior.imat.shape[0] == size


@pytest.mark.parametrize("mean", [1.0])
@pytest.mark.parametrize("imat", [1.0])
@pytest.mark.parametrize("size", [5])
@pytest.mark.parametrize("x", [np.arange(5)])
def test_gprior_objective(mean, imat, size, x):
    gprior = GaussianPrior(mean, imat, size=size)
    my_obj = gprior.objective(x)
    tr_obj = 0.5*np.sum((x - mean)**2)
    assert np.isclose(my_obj, tr_obj)


@pytest.mark.parametrize("mean", [1.0])
@pytest.mark.parametrize("imat", [1.0])
@pytest.mark.parametrize("size", [5])
@pytest.mark.parametrize("x", [np.arange(5)])
def test_gprior_gradient(mean, imat, size, x):
    gprior = GaussianPrior(mean, imat, size=size)
    my_gradient = gprior.gradient(x)
    tr_gradient = ad_jacobian(gprior.objective, x, x.shape)
    assert np.allclose(my_gradient, tr_gradient)


@pytest.mark.parametrize("lb", [0.0, np.zeros(5)])
@pytest.mark.parametrize("ub", [1.0, np.ones(5)])
@pytest.mark.parametrize("mat", [np.ones((5, 3))])
@pytest.mark.parametrize("size", [None, 5])
def test_uprior(lb, ub, mat, size):
    uprior = UniformPrior(lb, ub, mat, size)
    assert uprior.size == 5


@pytest.mark.parametrize("lb", [0.0])
@pytest.mark.parametrize("ub", [1.0])
@pytest.mark.parametrize("size", [5])
def test_uprior_update_size(lb, ub, size):
    uprior = UniformPrior(lb, ub)
    uprior.update_size(size)
    assert uprior.size == size
    assert len(uprior.lb) == size
    assert len(uprior.ub) == size
