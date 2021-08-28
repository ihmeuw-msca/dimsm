"""
Test process module
"""
import pytest
import numpy as np
from dimsm.process import default_gen_mat, default_gen_vmat, Process


@pytest.fixture
def name():
    return "age"


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
