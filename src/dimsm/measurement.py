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
    df : pd.DataFrame
        Data frame that contains measurements and their dimension labels.
    col_value : str, optional
        Column name corresponding to the measurements. Default to be `"value"`.
    varmat : Union[float, np.ndarray], optional
        (Co)varianace matrix corresponding to the measurements. It can be a
        scalar, vector, and a matrix. When it is a scalar, the matrix will be
        constructed as a diagonal matrix with the scalar value. When it is a
        vector, the matrix will be constructed as a diagonal matrix with the
        vector as the diagonal. Default to be 1.

    Attributes
    ----------
    df : pd.DataFrame
        Data frame that contains measurements and their dimension labels.
    col_value : str, optional
        Column name corresponding to the measurements. Default to be `"value"`.
    varmat : np.ndarray
        (Co)varianace matrix corresponding to the measurements.
    """

    varmat = property(attrgetter("_varmat"))

    def __init__(self,
                 df: pd.DataFrame,
                 col_value: str = "value",
                 varmat: Union[float, np.ndarray] = 1.0):
        self.df = df
        self.col_value = col_value
        self.varmat = varmat

    @varmat.setter
    def varmat(self, varmat: Union[float, np.ndarray]) -> None:
        if np.isscalar(varmat):
            varmat = np.diag(np.repeat(varmat, self.df.shape[0]))

        varmat = np.asarray(varmat).astype(float)
        if varmat.ndim == 1:
            if varmat.size != self.df.shape[0]:
                raise ValueError(f"{type(self).__name__}.varmat size does not "
                                 "match with the data frame.")
            varmat = np.diag(varmat)
        else:
            if varmat.ndim != 2:
                raise ValueError(f"Input for {type(self).__name__}.varmat must "
                                 "must be a scalar, vector or a matrix.")
            if varmat.shape[0] != varmat.shape[1]:
                raise ValueError(f"{type(self).__name__}.varmat must be a "
                                 "squared matrix.")
            if not (np.allclose(varmat, varmat.T) and
                    np.all(np.linalg.eigvals(varmat)) > 0.0):
                raise ValueError(f"{type(self).__name__}.varmat must be a "
                                 "symmetric positive definite matrix.")
        self._varmat = varmat

    @property
    def size(self) -> int:
        """Size of the observations."""
        return self.df.shape[0]
