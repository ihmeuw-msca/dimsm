"""
Test solver module
"""
import pytest
import numpy as np
from scipy.sparse import csc_matrix, diags
from scipy.optimize import LinearConstraint

from dimsm.solver import IPSolver


@pytest.fixture
def size():
    return 100


@pytest.fixture
def h_mat(size):
    return csc_matrix(diags(np.ones(size)))


@pytest.fixture
def g_vec(size):
    np.random.seed(123)
    return np.random.randn(size)*5.0


@pytest.fixture
def linear_constraints(size):
    mat = csc_matrix(diags(np.ones(size)))
    lb = -np.ones(size)
    ub = np.ones(size)
    return LinearConstraint(A=mat, lb=lb, ub=ub)


def test_simple_linear_solve(h_mat, g_vec):
    solver = IPSolver(h_mat, g_vec)
    p = solver.minimize()
    assert np.allclose(p, -g_vec)


def test_solve_with_constraints(h_mat, g_vec, linear_constraints):
    solver = IPSolver(h_mat, g_vec, linear_constraints)
    p = solver.minimize()
    assert np.allclose(p, np.maximum(-1.0, np.minimum(1.0, -g_vec)))
