"""
Prior
=====

Prior class contains prior information for state and dimension variable.
"""
from operator import attrgetter
from typing import Union, Optional
import numpy as np
from scipy.sparse import diags, csr_matrix


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
    if info.shape[0] != 1:
        if info.shape[0] != size:
            raise ValueError("Cannot extend info, size not matching.")
        return info
    if info.ndim == 1:
        return np.repeat(info, size)
    return csr_matrix(diags(np.repeat(info.diagonal(), size)))


class GaussianPrior:
    """Gaussian prior class includes prior information that will be incorporate
    into the likelihood as quadratic regularizers.

    Parameters
    ----------
    mean : Union[float, np.ndarray], optional
        Mean of the prior. Default to be 0.
    imat : Union[float, np.ndarray], optional
        Inverse (co)variance matrix of the prior. Default to be `0.0`.
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
    imat : np.ndarray
        inverse (co)variance matrix of the prior.
    mat : Optional[np.ndarray]
        Linear Mapping of the prior. Default to be `None`. When it is `None`,
        prior will be directly applied to the variable, equivalent with when
        `mat` is an identity matrix.
    size : int
        Size of the prior.

    Raises
    ------
    ValueError
        Raised when input for `imat` is not a scalar, vector or a matrix.
    ValueError
        Raised when matrix input for `imat` is not squared.
    ValueError
        Raised when input for `imat` matrix does not have positive diagonal.
    ValueError
        Raised when input for `mat` is not a matrix.

    Methods
    -------
    update_size(size)
        Update the size of the prior.
    objective(x)
        Objective function.
    graident(x)
        Gradient function.
    hessian()
        Hessian function.
    """

    mean = property(attrgetter("_mean"))
    imat = property(attrgetter("_imat"))
    mat = property(attrgetter("_mat"))

    def __init__(self,
                 mean: Union[float, np.ndarray] = 0.0,
                 imat: Union[float, np.ndarray] = 0.0,
                 mat: Optional[np.ndarray] = None,
                 size: Optional[int] = None):
        self.size = 1
        self.mean = mean
        self.imat = imat
        self.mat = mat

        if size is None:
            size = max(map(lambda x: x.shape[0],
                           [self.mean, self.imat, self.mat]))
        self.update_size(size)

    @mean.setter
    def mean(self, mean: Union[float, np.ndarray]):
        mean = np.asarray(mean)
        if np.isscalar(mean):
            mean = np.array([mean])
        self._mean = mean.ravel()

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
        if not all(imat.diagonal() > 0):
            raise ValueError(f"{type(self).__name__}.imat diagonal must be "
                             "positive.")
        self._imat = csr_matrix(imat)

    @mat.setter
    def mat(self, mat: Optional[np.ndarray]):
        if mat is not None:
            if mat.ndim != 2:
                raise ValueError(f"Input for {type(self).__name__}.mat must be "
                                 "a matrix.")
        else:
            mat = diags(np.ones(self.size))
        self._mat = csr_matrix(mat)

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
        if size != self.size:
            self.size = size
            self.mean = extend_info(self.mean, self.size)
            self.imat = extend_info(self.imat, self.size)
            self.mat = extend_info(self.mat, self.size)

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
        r = self.mean - self.mat.dot(x)
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
        r = self.mean - self.mat.dot(x)
        return -self.mat.T.dot(self.imat.dot(r))

    def hessian(self) -> np.ndarray:
        """Hessian function.

        Returns
        -------
        np.ndarray
            Hessian matrix.
        """
        return self.mat.T.dot(self.imat.dot(self.mat))

    def __repr__(self) -> str:
        return f"{type(self).__name__}(size={self.size})"


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
    imat : np.ndarray
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
        self.size = 1
        self.lb = lb
        self.ub = ub
        self.mat = mat

        if not np.all(self.lb <= self.ub):
            raise ValueError(f"{type(self).__name__}.lb must less or equal "
                             f"than {type(self).__name__}.ub")

        if size is None:
            size = max(map(lambda x: x.shape[0],
                           [self.lb, self.ub, self.mat]))
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
            if mat.ndim != 2:
                raise ValueError(f"Input for {type(self).__name__}.mat must be "
                                 "a matrix.")
        else:
            mat = diags(np.ones(self.size))
        self._mat = csr_matrix(mat)

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
        if size != self.size:
            self.size = size
            self.lb = extend_info(self.lb, self.size)
            self.ub = extend_info(self.ub, self.size)
            self.mat = extend_info(self.mat, self.size)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(size={self.size})"
