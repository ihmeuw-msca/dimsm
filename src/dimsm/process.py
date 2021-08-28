"""
Process
=======

Process class that contains the process matrix and its (co)variance matrix.
"""
from functools import partial
from operator import attrgetter
from typing import Callable, Optional

import numpy as np


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
    vec = np.flip(np.array([dt**i/i for i in range(1, 2*size)]))
    for i in range(size):
        mat[i] = vec[i:i + size]
    return sigma*mat


class Process:
    """Process class that contains the process matrix and its (co)variance
    matrix.

    Parameters
    ----------
    name : str
        Name of the corresponding dimension.
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
    name : str
        Name of the corresponding dimension.
    order : int
        Order of smoothness. Must be a non-negative integer.
    gen_mat : Callable
        Process matrix generator function.
    gen_vmat : Callable
        Process (co)variance matrix generator function.

    Raises
    ------
    TypeError
        Raised when input name is not a string.
    TypeError
        Raised when input order is not an integer.
    ValueError
        Raised when input order is negative.
    TypeError
        Raised when input process matrix generator is not callable or `None`.
    TypeError
        Raised when input process (co)variance generator is not callable or
        `None`.
    """

    name = property(attrgetter("_name"))
    order = property(attrgetter("_order"))
    gen_mat = property(attrgetter("_gen_mat"))
    gen_vmat = property(attrgetter("_gen_vmat"))

    def __init__(self,
                 name: str,
                 order: int,
                 gen_mat: Optional[Callable] = None,
                 gen_vmat: Optional[Callable] = None):
        self.name = name
        self.order = order
        self.gen_mat = gen_mat
        self.gen_vmat = gen_vmat

    @name.setter
    def name(self, name: str):
        if not isinstance(name, str):
            raise TypeError(f"{type(self).__name__}.name has to be a string.")
        self._name = name

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

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name}, order={self.order})"
