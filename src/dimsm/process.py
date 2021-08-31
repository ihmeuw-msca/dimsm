"""
Process
=======

Process class that contains the process matrix and its (co)variance matrix.
"""
from functools import partial
from operator import attrgetter
from typing import Callable, Optional, Tuple

import numpy as np
from scipy.sparse import block_diag, csr_matrix, diags

from dimsm.dimension import Dimension


def default_gen_mat(dt: float, size: int) -> np.ndarray:
    """Default process matrix generator.

    Parameters
    ----------
    dt : float
        Dimension variable difference.
    size : int
        Size of the process matrix, equals to number of rows and columns.

    Returns
    -------
    np.ndarray
        Process matrix.
    """
    mat = np.identity(size)
    for i in range(1, size):
        np.fill_diagonal(mat[:, i:], dt**i/np.math.factorial(i))
    return mat


def default_gen_vmat(dt: float, size: int, sigma: float = 1.0) -> np.ndarray:
    """Default process (co)variance matrix generator.

    Parameters
    ----------
    dt : float
        Dimension variable difference.
    size : int
        Size of the process matrix, equals to number of rows and columns.
    sigma : float, optional
        Noise level, by default 1.0

    Returns
    -------
    np.ndarray
        Process (co)variance matrix.
    """
    mat = np.zeros((size, size))
    for i in range(size):
        for j in range(i, size):
            mat[i, j] = dt**(i + j + 1)
            mat[i, j] /= (i + j + 1)*np.math.factorial(i)*np.math.factorial(j)
            mat[j, i] = mat[i, j]
    return np.flip(mat)*sigma**2


class Process:
    """Process class that contains the process matrix and its (co)variance
    matrix.

    Parameters
    ----------
    order : int
        Order of smoothness. Must be a non-negative integer.
    gen_mat : Optional[Callable], optional
        Process matrix generator function. This function takes in `dt` as the
        input and returns the process matrix. When it is `None`, it will use the
        default generator `default_gen_mat`. Default to `None`.
    gen_vmat : Optional[Callable], optional
        Process (co)variance matrix generator function. This function takes in
        `dt` as the input and returns the process matrix. When it is `None`, it
        will use the default generator `default_gen_vmat`. Default to `None`.

    Attributes
    ----------
    order : int
        Order of smoothness. Must be a non-negative integer.
    gen_mat : Callable
        Process matrix generator function.
    gen_vmat : Callable
        Process (co)variance matrix generator function.

    Raises
    ------
    TypeError
        Raised when input order is not an integer.
    ValueError
        Raised when input order is negative.
    TypeError
        Raised when input process matrix generator is not callable or `None`.
    TypeError
        Raised when input process (co)variance generator is not callable or
        `None`.

    Methods
    -------
    update_dim(dim)
        Update process matrices and their (co)variance matrices.
    reshape_var(x, var_shape, dim_index, reverse=False)
        Reshape the variable array.
    objective(x, var_shape, dim_index)
        Objective function.
    gradient(x, var_shape, dim_index)
        Gradient function.
    hessian(var_shape, dim_index)
        Hessian function.
    """

    order = property(attrgetter("_order"))
    gen_mat = property(attrgetter("_gen_mat"))
    gen_vmat = property(attrgetter("_gen_vmat"))

    def __init__(self,
                 order: int,
                 gen_mat: Optional[Callable] = None,
                 gen_vmat: Optional[Callable] = None):
        self.order = order
        self.gen_mat = gen_mat
        self.gen_vmat = gen_vmat

        self.mat = None
        self.imat = None

    @order.setter
    def order(self, order: int):
        if not isinstance(order, int):
            raise TypeError(f"{type(self).__name__}.order must be an integer.")
        if order < 0:
            raise ValueError(f"{type(self).__name__}.order must be "
                             "non-negative.")
        self._order = order

    @gen_mat.setter
    def gen_mat(self, gen_mat: Optional[Callable]):
        if gen_mat is None:
            gen_mat = partial(default_gen_mat, size=self.order + 1)
        else:
            if not callable(gen_mat):
                raise TypeError(f"{type(self).__name__}.gen_mat must be "
                                "callable.")
        self._gen_mat = gen_mat

    @gen_vmat.setter
    def gen_vmat(self, gen_vmat: Optional[Callable]):
        if gen_vmat is None:
            gen_vmat = partial(default_gen_vmat, size=self.order + 1)
        else:
            if not callable(gen_vmat):
                raise TypeError(f"{type(self).__name__}.gen_vmat must be "
                                "callable.")
        self._gen_vmat = gen_vmat

    def update_dim(self, dim: Dimension):
        """Update process matrices and their (co)variance matrices.

        Parameters
        ----------
        dim : Dimension
            The corresponding dimenion.
        """
        dts = np.diff(dim.grid)
        self.mat = csr_matrix(block_diag([self.gen_mat(dt) for dt in dts]))
        self.imat = csr_matrix(
            block_diag([np.linalg.inv(self.gen_vmat(dt)) for dt in dts])
        )

    def reshape_var(self,
                    x: np.ndarray,
                    var_shape: Tuple[int],
                    dim_index: int,
                    reverse: bool = False) -> np.ndarray:
        """Reshape the variable array.

        Parameters
        ----------
        x : np.ndarray
            Variable array.
        var_shape : Tuple[int]
            Variable shape corresponding to one layer.
        dim_index : int
            Corresponding dimension index.
        reverse : bool, optional
            If `True` reshape the variable back to origibnal shape, by default
            `False`.

        Returns
        -------
        np.ndarray
            Reshaped variable array.
        """
        other_dim_indices = list(range(len(var_shape)))
        other_dim_indices.remove(dim_index)

        if reverse:
            indices = np.argsort(self.reshape_var(
                np.arange(x.size), var_shape, dim_index, reverse=False
            ).ravel())
            return x.ravel()[indices]
        x = x.reshape(self.order + 1, *var_shape)
        x = x.transpose((*[i + 1 for i in other_dim_indices], dim_index + 1, 0))
        x = x.reshape(int(np.prod([var_shape[i] for i in other_dim_indices])),
                      var_shape[dim_index]*(self.order + 1))
        return x

    def objective(self,
                  x: np.ndarray,
                  var_shape: Tuple[int],
                  dim_index: int) -> float:
        """Objective function.

        Parameters
        ----------
        x : np.ndarray
            Variable array.
        var_shape : Tuple[int]
            Variable shape corresponding to one layer.
        dim_index : int
            Corresponding dimension index.

        Returns
        -------
        float
            Objective value.
        """
        s = self.order + 1
        x = self.reshape_var(x, var_shape, dim_index)
        r = x.T[s:] - self.mat.dot(x.T[:-s])
        t = self.imat.dot(r)
        return 0.5*np.sum(r*t)

    def gradient(self,
                 x: np.ndarray,
                 var_shape: Tuple[int],
                 dim_index: int) -> np.ndarray:
        """Gradient function.

        Parameters
        ----------
        x : np.ndarray
            Variable array.
        var_shape : Tuple[int]
            Variable shape corresponding to one layer.
        dim_index : int
            Corresponding dimension index.

        Returns
        -------
        np.ndarray
            Gradient array.
        """
        s = self.order + 1
        x = self.reshape_var(x, var_shape, dim_index)
        r = x.T[s:] - self.mat.dot(x.T[:-s])
        t = self.imat.dot(r)
        g = np.zeros(x.shape, dtype=x.dtype)

        g.T[s:] += t
        g.T[:-s] -= self.mat.T.dot(t)

        return self.reshape_var(g, var_shape, dim_index, reverse=True)

    def hessian(self,
                var_shape: Tuple[int],
                dim_index: int) -> np.ndarray:
        """Hessian function.

        Parameters
        ----------
        var_shape : Tuple[int]
            Variable shape corresponding to one layer.
        dim_index : int
            Corresponding dimension index.

        Returns
        -------
        np.ndarray
            Hessian matrix.
        """
        s = self.order + 1
        n = var_shape[dim_index]
        k = np.prod(var_shape) // n
        # compute hessian for each column
        mat_m = diags(np.ones((n - 1)*s), shape=((n - 1)*s, n*s))
        mat_p = diags(np.ones((n - 1)*s), shape=((n - 1)*s, n*s), offsets=s)
        mat = mat_p - self.mat.dot(mat_m)
        row_hessian = mat.T.dot(self.imat.dot(mat))
        # create hessian matrix
        hessian = csr_matrix(block_diag([row_hessian]*k))
        # permute hessian into right order
        indices = self.reshape_var(
            np.arange(n*s*k), var_shape, dim_index, reverse=True
        )
        hessian = hessian[indices[:, None], indices]
        return hessian

    def __repr__(self) -> str:
        return f"{type(self).__name__}(order={self.order})"
