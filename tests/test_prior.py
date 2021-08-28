"""
Test prior module
"""
import pytest
import numpy as np
from dimsm.prior import GaussianPrior, UniformPrior, extend_info


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
    assert len(extended_info) == size
    if info.ndim == 1:
        assert np.allclose(extended_info, info[0])
    else:
        assert np.allclose(np.diag(extended_info), info[0, 0])


@pytest.mark.parametrize("mean", [1.0, np.ones(5)])
@pytest.mark.parametrize("vmat", [1.0, np.ones(5), np.identity(5)])
@pytest.mark.parametrize("mat", [np.ones((5, 3))])
@pytest.mark.parametrize("size", [None, 5])
def test_gprior(mean, vmat, mat, size):
    gprior = GaussianPrior(mean, vmat, mat, size)
    assert gprior.size == 5


@pytest.mark.parametrize("mean", [1.0])
@pytest.mark.parametrize("vmat", [1.0])
@pytest.mark.parametrize("size", [5])
def test_gprior_update_size(mean, vmat, size):
    gprior = GaussianPrior(mean, vmat)
    gprior.update_size(size)
    assert gprior.size == size
    assert len(gprior.mean) == size
    assert len(gprior.vmat) == size


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
