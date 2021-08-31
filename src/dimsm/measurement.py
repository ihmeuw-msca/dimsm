"""
Measurement
===========

Contains table of measurements and the (co)variance matrix.
"""
from typing import List, Union
from operator import attrgetter
from itertools import product

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, diags
from dimsm.dimension import Dimension


class Measurement:
    """Measurement class stores table of measurements and the (co)variance
    matrix.

    Parameters
    ----------
    data : pd.DataFrame
        Data frame that contains measurements and their dimension labels.
    col_value : str, optional
        Column name corresponding to the measurements. Default to be `"value"`.
    imat : Union[float, np.ndarray], optional
        Inverse (co)varianace matrix corresponding to the measurements. It can
        be a scalar, vector, and a matrix. When it is a scalar, the matrix will
        be constructed as a diagonal matrix with the scalar value. When it is a
        vector, the matrix will be constructed as a diagonal matrix with the
        vector as the diagonal. Default to be 1.

    Attributes
    ----------
    data : pd.DataFrame
        Data frame that contains measurements and their dimension labels.
    col_value : str, optional
        Column name corresponding to the measurements. Default to be `"value"`.
    imat : np.ndarray
        Inverse (co)varianace matrix corresponding to the measurements.
    mat : np.ndarray
        Measurement matrix operating on state variable.
    size : int
        Size of the measurement which is the number of rows of `data`.

    Raises
    ------
    TypeError
        Raised when input for `data` is not a data frame.
    ValueError
        Raised when `col_value` not in `data` columns.
    ValueError
        Raised when vector input for `imat` not matching with number of rows
        in `data`.
    ValueError
        Raised when input for `imat` is not a scalar, vector or a matrix.
    ValueError
        Raised when matrix input for `imat` is not squared.
    ValueError
        Raised when input for `imat` matrix does not have positive diagonal.

    Methods
    -------
    update_dim(dims)
        Update the observation linear mapping.
    objective(x)
        Objective function.
    graident(x)
        Gradient function.
    hessian()
        Hessian function.
    """

    data = property(attrgetter("_data"))
    imat = property(attrgetter("_imat"))
    col_value = property(attrgetter("_col_value"))

    def __init__(self,
                 data: pd.DataFrame,
                 col_value: str = "value",
                 imat: Union[float, np.ndarray] = 1.0):
        self.data = data
        self.col_value = col_value
        self.imat = imat
        self.mat = None

    @data.setter
    def data(self, data: pd.DataFrame):
        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"{type(self).__name__}.data has to be a data "
                            "frame.")
        self._data = data

    @col_value.setter
    def col_value(self, col_value: str):
        if col_value not in self.data.columns:
            raise ValueError(f"{type(self).__name__}.col_value not in data "
                             "frame columns.")
        self._col_value = col_value

    @imat.setter
    def imat(self, imat: Union[float, np.ndarray]):
        if np.isscalar(imat):
            imat = np.repeat(imat, self.size)
        if imat.ndim == 1:
            imat = diags(imat)
        if imat.ndim != 2:
            raise ValueError(f"Input for {type(self).__name__}.imat must "
                             "must be a scalar, vector or a matrix.")
        if imat.shape[0] != imat.shape[1]:
            raise ValueError(f"{type(self).__name__}.imat must be a "
                             "squared matrix.")
        if imat.shape[0] != self.size:
            raise ValueError(f"{type(self).__name__}.imat size does not "
                             "match with the data frame.")
        if not all(imat.diagonal() > 0):
            raise ValueError(f"{type(self).__name__}.imat diagonal must be "
                             "positive.")
        self._imat = csr_matrix(imat)

    @property
    def size(self) -> int:
        """Size of the observations."""
        return self.data.shape[0]

    def update_dim(self, dims: List[Dimension]):
        """Update the observation linear mapping.

        Parameters
        ----------
        dims : List[Dimension]
            Dimensions specification.
        """
        var_shape = tuple(dim.size for dim in dims)
        row_indices = []
        col_indices = []
        mat_weights = []

        for i, obs in self.data.iterrows():
            dim_indices = []
            dim_weights = []
            for dim in dims:
                x = obs[dim.name]
                j = dim.grid.searchsorted(x, side="right")
                if j == 0:
                    indices, weights = (0,), (1,)
                elif j == dim.size:
                    indices, weights = (dim.size - 1,), (1,)
                else:
                    p = (dim.grid[j] - x) / (dim.grid[j] - dim.grid[j - 1])
                    indices, weights = (j - 1, j), (p, 1 - p)
                dim_indices.append(indices)
                dim_weights.append(weights)
            indices = product(*dim_indices)
            weights = product(*dim_weights)

            add_col_indices = list(
                map(lambda x: np.ravel_multi_index(x, var_shape), indices)
            )
            col_indices.extend(add_col_indices)
            row_indices.extend([i]*len(add_col_indices))
            mat_weights.extend(list(map(np.prod, weights)))

        self.mat = csr_matrix((mat_weights, (row_indices, col_indices)),
                              shape=(self.size, np.prod(var_shape)))

    def objective(self, x: np.ndarray) -> float:
        """Objective function.

        Parameters
        ----------
        x : np.ndarray
            Variable array.

        Returns
        -------
        float
            Objective value.
        """
        r = self.data[self.col_value].values - self.mat.dot(x.ravel())
        return 0.5*r.dot(self.imat.dot(r))

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Gradient function.

        Parameters
        ----------
        x : np.ndarray
            Variable array.

        Returns
        -------
        np.ndarray
            Gradient vector.
        """
        r = self.data[self.col_value].values - self.mat.dot(x.ravel())
        return -self.mat.T.dot(self.imat.dot(r))

    def hessian(self) -> np.ndarray:
        """Hessian function.

        Returns
        -------
        np.ndarray
            Hessian matrix.
        """
        return self.mat.T.dot(self.imat.dot(self.mat))

    def __repr__(self) -> int:
        return f"{type(self).__name__}(size={self.size})"
