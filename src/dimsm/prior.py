"""
Prior
=====

Prior class contains prior information for state and dimension variable.
"""
from operator import attrgetter
from typing import Union, Optional
import numpy as np


def extend_info(info: np.ndarray, size: int) -> np.ndarray:
    """Extend infomation array.

    Parameters
    ----------
    info : np.ndarray
        Information array, must be a vector or a matrix.
    size : int
        Expected extension size.

    Returns
    -------
    np.ndarray
        Extended information array

    Raises
    ------
    ValueError
        Raised when information array is not a vector or a matrix.
    ValueError
        Raised when information array cannot be extended to the given size.
    """
    if not (info.ndim == 1 or info.ndim == 2):
        raise ValueError("Information array must be a vector or a matrix.")
    if len(info) != 1:
        if len(info) != size:
            raise ValueError("Cannot extend info, size not matching.")
        return info
    if info.ndim == 1:
        return np.repeat(info, size)
    return np.diag(np.repeat(info[0], size))


class GaussianPrior:
    """Gaussian prior class includes prior information that will be incorporate
    into the likelihood as quadratic regularizers.

    Parameters
    ----------
    mean : Union[float, np.ndarray], optional
        Mean of the prior. Default to be 0.
    vmat : Union[float, np.ndarray], optional
        (Co)variance matrix of the prior. Default to be `np.inf`.
    mat : Optional[np.ndarray], optional
        Linear Mapping of the prior. Default to be `None`. When it is `None`,
        prior will be directly applied to the variable, equivalent with when
        `mat` is an identity matrix.
    size : Optional[int], optional
        Size of the prior. Default to be `None`. When it is `None`, size will be
        inferred from other inputs. And when `mat` is not `None`, size will be
        overwritten by the number of rows of `mat`.

    Attributes
    ----------
    mean : np.ndarray
        Mean of the prior.
    vmat : np.ndarray
        (Co)variance matrix of the prior.
    mat : Optional[np.ndarray]
        Linear Mapping of the prior. Default to be `None`. When it is `None`,
        prior will be directly applied to the variable, equivalent with when
        `mat` is an identity matrix.
    size : int
        Size of the prior.

    Raises
    ------
    ValueError
        Raised when input for `vmat` is not a scalar, vector or a matrix.
    ValueError
        Raised when matrix input for `vmat` is not squared.
    ValueError
        Raised when `vmat` is not symmetric positive definite.
    ValueError
        Raised when input for `mat` is not a matrix.

    Methods
    -------
    update_size(size)
        Update the size of the prior.
    """

    mean = property(attrgetter("_mean"))
    vmat = property(attrgetter("_vmat"))
    mat = property(attrgetter("_mat"))

    def __init__(self,
                 mean: Union[float, np.ndarray] = 0.0,
                 vmat: Union[float, np.ndarray] = np.inf,
                 mat: Optional[np.ndarray] = None,
                 size: Optional[int] = None):
        self.mean = mean
        self.vmat = vmat
        self.mat = mat

        if self.mat is not None:
            size = len(self.mat)
        if size is None:
            size = max(map(len, [self.mean, self.vmat]))
        self.update_size(size)

    @mean.setter
    def mean(self, mean: Union[float, np.ndarray]):
        mean = np.asarray(mean)
        if np.isscalar(mean):
            mean = np.array([mean])
        self._mean = mean.ravel()

    @vmat.setter
    def vmat(self, vmat: Union[float, np.ndarray]):
        vmat = np.asarray(vmat).astype(float)
        if np.isscalar(vmat):
            vmat = np.np.repeat(vmat, self.size)
        if vmat.ndim == 1:
            vmat = np.diag(vmat)
        if vmat.ndim != 2:
            raise ValueError(f"Input for {type(self).__name__}.vmat must "
                             "must be a scalar, vector or a matrix.")
        if vmat.shape[0] != vmat.shape[1]:
            raise ValueError(f"{type(self).__name__}.vmat must be a "
                             "squared matrix.")
        if not (np.allclose(vmat, vmat.T) and
                np.all(np.linalg.eigvals(vmat) > 0.0)):
            raise ValueError(f"{type(self).__name__}.vmat must be a "
                             "symmetric positive definite matrix.")
        self._vmat = vmat

    @mat.setter
    def mat(self, mat: Optional[np.ndarray]):
        if mat is not None:
            mat = np.asarray(mat).astype(float)
            if mat.ndim != 2:
                raise ValueError(f"Input for {type(self).__name__}.mat must be "
                                 "a matrix.")
        self._mat = mat

    def update_size(self, size: int):
        """Update the size of the prior.

        Parameters
        ----------
        size : int
            New prior size.

        Raises
        ------
        TypeError
            Raised when `size` is not an integer.
        ValueError
            Raised when `size` is not positive.
        """
        if not isinstance(size, int):
            raise TypeError(f"{type(self).__name__}.size must be an integer.")
        if size <= 0:
            raise ValueError(f"{type(self).__name__}.size must be positive.")
        self.size = size
        self.mean = extend_info(self.mean)
        self.vmat = extend_info(self.vmat)


class UniformPrior:
    """Uniform prior class includes prior information that will be incorporate
    into the likelihood as linear constraints.

    Parameters
    ----------
    lb : Union[float, np.ndarray], optional
        Lower bounds of the prior. Default to be `-np.inf`.
    ub : Union[float, np.ndarray], optional
        Upper bounds of the prior. Default to be `np.inf`.
    mat : Optional[np.ndarray], optional
        Linear Mapping of the prior. Default to be `None`. When it is `None`,
        prior will be directly applied to the variable, equivalent with when
        `mat` is an identity matrix.
    size : Optional[int], optional
        Size of the prior. Default to be `None`. When it is `None`, size will be
        inferred from other inputs. And when `mat` is not `None`, size will be
        overwritten by the number of rows of `mat`.

    Attributes
    ----------
    mean : np.ndarray
        Mean of the prior.
    vmat : np.ndarray
        (Co)variance matrix of the prior.
    mat : Optional[np.ndarray]
        Linear Mapping of the prior. Default to be `None`. When it is `None`,
        prior will be directly applied to the variable, equivalent with when
        `mat` is an identity matrix.
    size : int
        Size of the prior.

    Raises
    ------
    ValueError
        Raised when not all lower bounds are less or equal than upper bounds.
    ValueError
        Raised when input for `mat` is not a matrix.

    Methods
    -------
    update_size(size)
        Update the size of the prior.
    """

    lb = property(attrgetter("_lb"))
    ub = property(attrgetter("_ub"))
    mat = property(attrgetter("_mat"))

    def __init__(self,
                 lb: Union[float, np.ndarray] = -np.inf,
                 ub: Union[float, np.ndarray] = np.inf,
                 mat: Optional[np.ndarray] = None,
                 size: Optional[int] = None):
        self.lb = lb
        self.ub = ub
        self.mat = mat

        if not np.all(self.lb <= self.ub):
            raise ValueError(f"{type(self).__name__}.lb must less or equal "
                             f"than {type(self).__name__}.ub")

        if self.mat is not None:
            size = len(self.mat)
        if size is None:
            size = max(map(len, [self.mean, self.vmat]))
        self.update_size(size)

    @lb.setter
    def lb(self, lb: Union[float, np.ndarray]):
        lb = np.asarray(lb)
        if np.isscalar(lb):
            lb = np.array([lb])
        self._lb = lb.ravel()

    @ub.setter
    def ub(self, ub: Union[float, np.ndarray]):
        ub = np.asarray(ub)
        if np.isscalar(ub):
            ub = np.array([ub])
        self._ub = ub.ravel()

    @mat.setter
    def mat(self, mat: Optional[np.ndarray]):
        if mat is not None:
            mat = np.asarray(mat).astype(float)
            if mat.ndim != 2:
                raise ValueError(f"Input for {type(self).__name__}.mat must be "
                                 "a matrix.")
        self._mat = mat

    def update_size(self, size: int):
        """Update the size of the prior.

        Parameters
        ----------
        size : int
            New prior size.

        Raises
        ------
        TypeError
            Raised when `size` is not an integer.
        ValueError
            Raised when `size` is not positive.
        """
        if not isinstance(size, int):
            raise TypeError(f"{type(self).__name__}.size must be an integer.")
        if size < 0:
            raise ValueError(f"{type(self).__name__}.size must be "
                             "non-negative.")
        self.size = size
        self.lb = extend_info(self.lb)
        self.ub = extend_info(self.ub)
