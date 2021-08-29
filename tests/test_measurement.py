"""
Test measurement module
"""
import numpy as np
import pandas as pd
import pytest
from dimsm.dimension import Dimension
from dimsm.measurement import Measurement


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
def vmat():
    return np.identity(6)


@pytest.fixture
def measurement(data, col_value, vmat):
    return Measurement(data, col_value, vmat)


def test_size(measurement, data):
    assert measurement.size == data.shape[0]


@pytest.mark.parametrize("data", [np.ones((5, 3))])
def test_data_error(data, col_value, vmat):
    with pytest.raises(TypeError):
        Measurement(data, col_value, vmat)


@pytest.mark.parametrize("col_value", ["random"])
def test_col_value_error(data, col_value, vmat):
    with pytest.raises(ValueError):
        Measurement(data, col_value, vmat)


@pytest.mark.parametrize("vmat", [-1.0, 0.0,
                                  np.ones(10), -np.ones(6),
                                  np.ones((6, 3, 2)), np.ones((6, 3)),
                                  np.random.randn(6, 6),
                                  -np.diag(np.ones(6))])
def test_vmat_error(data, col_value, vmat):
    with pytest.raises(ValueError):
        Measurement(data, col_value, vmat)


def test_update_mat(measurement, dims):
    measurement.update_mat(dims)
    assert measurement.mat.shape == (6, 20)
    assert np.allclose(np.sum(measurement.mat.toarray(), axis=1), 1.0)
