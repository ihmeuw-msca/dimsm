"""
Utility Functions
=================

Utility functions that can be used across different modules.
"""
from typing import Tuple
import numpy as np


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
