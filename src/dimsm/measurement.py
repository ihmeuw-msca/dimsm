"""
Measurement
===========

Contains table of measurements and the (co)variance matrix.
"""
from typing import Union
from operator import attrgetter

import numpy as np
import pandas as pd


class Measurement:
    """Measurement class stores table of measurements and the (co)variance
    matrix.

    Parameters
    ----------
    data : pd.DataFrame
        Data frame that contains measurements and their dimension labels.
    col_value : str, optional
        Column name corresponding to the measurements. Default to be `"value"`.
    vmat : Union[float, np.ndarray], optional
        (Co)varianace matrix corresponding to the measurements. It can be a
        scalar, vector, and a matrix. When it is a scalar, the matrix will be
        constructed as a diagonal matrix with the scalar value. When it is a
        vector, the matrix will be constructed as a diagonal matrix with the
        vector as the diagonal. Default to be 1.

    Attributes
    ----------
    data : pd.DataFrame
        Data frame that contains measurements and their dimension labels.
    col_value : str, optional
        Column name corresponding to the measurements. Default to be `"value"`.
    vmat : np.ndarray
        (Co)varianace matrix corresponding to the measurements.
    size : int
        Size of the measurement which is the number of rows of `data`.

    Raises
    ------
    TypeError
        Raised when input for `data` is not a data frame.
    ValueError
        Raised when `col_value` not in `data` columns.
    ValueError
        Raised when vector input for `vmat` not matching with number of rows
        in `data`.
    ValueError
        Raised when input for `vmat` is not a scalar, vector or a matrix.
    ValueError
        Raised when matrix input for `vmat` is not squared.
    ValueError
        Raised when `vmat` is not symmetric positive definite.
    """

    data = property(attrgetter("_data"))
    vmat = property(attrgetter("_vmat"))
    col_value = property(attrgetter("_col_value"))

    def __init__(self,
                 data: pd.DataFrame,
                 col_value: str = "value",
                 vmat: Union[float, np.ndarray] = 1.0):
        self.data = data
        self.col_value = col_value
        self.vmat = vmat

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

    @vmat.setter
    def vmat(self, vmat: Union[float, np.ndarray]):
        vmat = np.asarray(vmat).astype(float)
        if vmat.ndim == 0:
            vmat = np.repeat(vmat, self.size)
        if vmat.ndim == 1:
            vmat = np.diag(vmat)
        if vmat.ndim != 2:
            raise ValueError(f"Input for {type(self).__name__}.vmat must "
                             "must be a scalar, vector or a matrix.")
        if vmat.shape[0] != vmat.shape[1]:
            raise ValueError(f"{type(self).__name__}.vmat must be a "
                             "squared matrix.")
        if vmat.shape[0] != self.size:
            raise ValueError(f"{type(self).__name__}.vmat size does not "
                             "match with the data frame.")
        if not (np.allclose(vmat, vmat.T) and
                np.all(np.linalg.eigvals(vmat) > 0.0)):
            raise ValueError(f"{type(self).__name__}.vmat must be a "
                             "symmetric positive definite matrix.")
        self._vmat = vmat

    @property
    def size(self) -> int:
        """Size of the observations."""
        return self.data.shape[0]

    def __repr__(self) -> int:
        return f"{type(self).__name__}(size={self.size})"
