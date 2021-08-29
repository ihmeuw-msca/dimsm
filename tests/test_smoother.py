"""
Test smoother module
"""
import pandas as pd
import pytest
from dimsm.dimension import Dimension
from dimsm.measurement import Measurement
from dimsm.prior import GaussianPrior, UniformPrior
from dimsm.process import Process
from dimsm.smoother import Smoother


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
    return {"state": GaussianPrior(mean=0.0, vmat=1.0),
            "year[0]": GaussianPrior(mean=0.0, vmat=1.0)}


@pytest.fixture
def upriors():
    return {"state": UniformPrior(lb=-1.0, ub=1.0),
            "year[0]": UniformPrior(lb=-1.0, ub=1.0)}


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