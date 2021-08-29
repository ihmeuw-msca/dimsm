"""
Smoother
========

Smoother class gathers all components information, provides interface for
optimization solver and user to extract the results.
"""
from operator import attrgetter
from typing import Dict, List, Optional

import numpy as np

from dimsm.dimension import Dimension
from dimsm.measurement import Measurement
from dimsm.prior import GaussianPrior, UniformPrior
from dimsm.process import Process


class Smoother:
    """Smoother class gathers all components information, provides interface for
    optimization solver and user to extract the results.

    Parameters
    ----------
    dims : List[Dimension]
        List of different dimensions.
    meas : Measurement
        Measurements of the model.
    prcs : Optional[Dict[str, Process]], optional
        Dictionary of processes of the model. It requires dimension name as the
        key and instances of Process as the value. Default to be `None`.
    gpriors : Optional[Dict[str, GaussianPrior]], optional
        Gaussian priors for each variable. It requires variable name as the key
        and instance of Gaussian prior as the value. Default to be `None`.
    upriors : Optional[Dict[str, GaussianPrior]], optional
        Gaussian priors for each variable. It requires variable name as the key
        and instance of Uniform prior as the value. Default to be `None`.


    Attributes
    ----------
    dims : List[Dimension]
        List of different dimensions.
    meas : Measurement
        Measurements of the model.
    prcs : Dict[str, Process]
        Dictionary of processes of the model.
    gpriors : Dict[str, GaussianPrior]
        Gaussian priors for each variable.
    upriors : Dict[str, GaussianPrior]
        Uniform priors for each variable.
    var_shape : Tuple[int]
        Shape of each variable.
    dim_names : List[str]
        Names of the dimensions.
    prc_names : List[str]
        Names of the processes.
    var_size : int
        Size of the variable.
    num_dims : int
        Number of dimensions.
    num_vars : int
        Number of variables.
    var_indices : Dict[str, List[int]]
        Variable indices dictionary with `"state"` or process name as the key
        and a list of integers (indices) as values.

    Methods
    -------
    objective(x)
        Objective function.
    gradient(x)
        Gradient function.
    """

    dims = property(attrgetter("_dims"))
    meas = property(attrgetter("_meas"))
    prcs = property(attrgetter("_prcs"))
    gpriors = property(attrgetter("_gpriors"))
    upriors = property(attrgetter("_upriors"))

    def __init__(self,
                 dims: List[Dimension],
                 meas: Measurement,
                 prcs: Optional[Dict[str, Process]] = None,
                 gpriors: Optional[Dict[str, GaussianPrior]] = None,
                 upriors: Optional[Dict[str, UniformPrior]] = None):
        self.dims = dims
        self.meas = meas
        self.prcs = prcs
        self.gpriors = gpriors
        self.upriors = upriors

    @dims.setter
    def dims(self, dims: List[Dimension]):
        if not all(isinstance(dim, Dimension) for dim in dims):
            raise TypeError(f"{type(self).__name__}.dims must be a list "
                            "of instances of Dimension.")
        self.dim_names = [dim.name for dim in dims]
        self.var_shape = tuple(dim.size for dim in dims)
        self.var_size = np.prod(self.var_shape)
        self.num_dims = len(dims)
        self._dims = list(dims)

    @meas.setter
    def meas(self, meas: Measurement):
        if not isinstance(meas, Measurement):
            raise TypeError(f"{type(self).__name__}.meas must be an "
                            "instance of Measurement.")
        for dim_name in self.dim_names:
            if dim_name not in meas.data.columns:
                raise ValueError(f"{type(self).__name__}.meas must contain "
                                 f"dimension label {dim_name} in the column.")
        meas.update_dim(self.dims)
        self._meas = meas

    @prcs.setter
    def prcs(self, prcs: Optional[Dict[str, Process]]):
        self.num_vars = 1
        self.prc_names = []
        self.var_indices = {"state": [0]}
        self._prcs = {}
        if prcs is not None:
            if not isinstance(prcs, Dict):
                raise TypeError(f"{type(self).__name__}.prcs must be a "
                                "dictionary.")
            for key, value in prcs.items():
                if key not in self.dim_names:
                    raise ValueError(f"{key} not in "
                                     f"{type(self).__name__}.dim_names.")
                if not isinstance(value, Process):
                    raise TypeError(f"{type(self).__name__}.prcs values must "
                                    "be instances of Process.")
                value.update_dim(self.dims[self.dim_names.index(key)])
                self.prc_names.append(key)
                self.num_vars += value.order
            self.prc_names.sort(key=lambda name: self.dim_names.index(name))
            counter = 1
            for prc_name in self.prc_names:
                self.var_indices[prc_name] = list(
                    range(counter, counter + prcs[prc_name].order)
                )
                counter += prcs[prc_name].order
            self._prcs = prcs

    @gpriors.setter
    def gpriors(self, gpriors: Optional[Dict[str, List[GaussianPrior]]]):
        self._gpriors = {}
        if gpriors is not None:
            if not isinstance(gpriors, Dict):
                raise TypeError(f"{type(self).__name__}.gpriors must be a "
                                "dictionary.")

    @upriors.setter
    def upriors(self, upriors: Optional[Dict[str, List[UniformPrior]]]):
        self._upriors = {}
        if upriors is not None:
            if not isinstance(upriors, Dict):
                raise TypeError(f"{type(self).__name__}.gpriors must be a "
                                "dictionary.")

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
        params = x.reshape(self.num_vars, self.var_size)

        # measurement
        value = self.meas.objective(params[self.var_indices["state"]])

        # process
        for name, prc in self.prcs.items():
            indices = [0] + self.var_indices[name]
            value += prc.objective(params[indices],
                                   self.var_shape,
                                   self.dim_names.index(name))

        # gprior
        for name, gpriors in self.gpriors.items():
            for i, gprior in enumerate(gpriors):
                if gprior is not None:
                    index = self.var_indices["name"][i]
                    value += gprior.objective(params[index])

        return value

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
        params = x.reshape(self.num_vars, self.var_size)
        gvalue = np.zeros((self.num_vars, self.var_size))

        # measurement
        gvalue[self.var_indices["state"]] += self.meas.gradient(
            params[self.var_indices["state"]]
        )

        # process
        for name, prc in self.prcs.items():
            indices = [0] + self.var_indices[name]
            gvalue[indices] += prc.gradient(params[indices],
                                            self.var_shape,
                                            self.dim_names.index(name))

        # gprior
        for name, gpriors in self.gpriors.items():
            for i, gprior in enumerate(gpriors):
                if gprior is not None:
                    index = self.var_indices["name"][i]
                    gvalue[index] += gprior.gradient(params[index])

        return gvalue.ravel()

    def __repr__(self) -> str:
        return (f"{type(self).__name__}(\n"
                f"    dim_names={self.dim_names},\n"
                f"    meas={self.meas},\n"
                f"    prcs={self.prcs}\n"
                ")")
