"""
Test dimension module
"""
import numpy as np
import pytest
from dimsm.dimension import Dimension


@pytest.fixture
def name():
    return "age"


@pytest.fixture
def grid():
    return [25, 30, 35, 40, 60]


@pytest.fixture
def dim_age(name, grid):
    return Dimension(name, grid)


def test_name(dim_age, name):
    assert dim_age.name == name


@pytest.mark.parametrize("name", [1, 1.0, True])
def test_name_error(name, grid):
    with pytest.raises(TypeError):
        Dimension(name, grid)


def test_grid(dim_age, grid):
    assert np.allclose(dim_age.grid, grid)
    assert dim_age.grid.dtype == float


@pytest.mark.parametrize("grid", [[], [1], [1, 1]])
def test_grid_error(name, grid):
    with pytest.raises(ValueError):
        Dimension(name, grid)


def test_size(dim_age, grid):
    assert dim_age.size == len(grid)
