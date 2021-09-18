"""
Utility Functions
=================

Utility functions that can be used across different modules.
"""
from typing import Tuple
import numpy as np
from numba import njit


def reshape_var(x: np.ndarray,
                var_shape: Tuple[int],
                dim_index: int,
                reverse: bool = False) -> np.ndarray:
    """Reshape the variable array to expose the selected dimension to last
    dimension.

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
    x = np.asarray(x)
    n = np.prod(var_shape)
    k = x.size // n
    other_dim_indices = np.array([i for i in range(len(var_shape))
                                  if i != dim_index])

    if reverse:
        indices = np.argsort(reshape_var(
            np.arange(x.size), var_shape, dim_index, reverse=False
        ).ravel())
        return x.ravel()[indices]

    x = x.reshape(k, *var_shape)
    x = x.transpose(np.hstack([other_dim_indices + 1, dim_index + 1, 0]))
    x = x.reshape(n // var_shape[dim_index], var_shape[dim_index]*k)
    return x


@njit
def unravel_index(index: int, dims: Tuple[int]) -> np.ndarray:
    """Unravel the index into multi index.

    Parameters
    ----------
    index : int
        Given index.
    dims : Tuple[int]
        Given dimensions.

    Returns
    -------
    np.ndarray
        Returns multi index.
    """
    ndim = len(dims)
    multi_index = np.zeros(ndim, dtype=np.int64)
    sizes = np.zeros(ndim, dtype=np.int64)

    sizes[-1] = 1
    for i in range(ndim - 1):
        sizes[ndim - 2 - i] = sizes[ndim - 1 - i] * dims[ndim - 1 - i]

    for i in range(ndim):
        multi_index[i] = index // sizes[i]
        index -= multi_index[i] * sizes[i]

    return multi_index


@njit
def ravel_multi_index(multi_index: Tuple[int], dims: Tuple[int]) -> int:
    """Ravel multi index into single index.

    Parameters
    ----------
    multi_index : Tuple[int]
        Given multi index.
    dims : Tuple[int]
        Given dimension.

    Returns
    -------
    int
        Returns index when flatten the multiarray.
    """
    ndim = len(dims)
    index = 0
    sizes = np.zeros(ndim, dtype=np.int64)

    sizes[-1] = 1
    for i in range(ndim - 1):
        sizes[ndim - 2 - i] = sizes[ndim - 1 - i] * dims[ndim - 1 - i]

    for i in range(ndim):
        index += multi_index[i]*sizes[i]

    return index
