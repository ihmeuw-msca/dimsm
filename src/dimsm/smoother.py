"""
Smoother
========

Smoother class gathers all components information, provides interface for
optimization solver and user to extract the results.
"""
from operator import attrgetter
from typing import Dict, List, Optional
from math import prod

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
    var_names : List[str]
        Names of the variables.
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
        self._measurement = meas

    @prcs.setter
    def prcs(self, prcs: Optional[Dict[str, Process]]):
        self.var_names = ["state"]
        self.prc_names = []
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
                self.prc_names.append(key)
            self.prc_names.sort(key=lambda name: self.dim_names.index(name))
            self.var_names += [f"{prc_name}[{i}]"
                               for prc_name in self.prc_names
                               for i in range(prcs[prc_name].order)]
            self._prcs = prcs
        self._prcs = {}

    @gpriors.setter
    def gpriors(self, gpriors: Optional[Dict[str, GaussianPrior]]):
        if gpriors is not None:
            if not isinstance(gpriors, Dict):
                raise TypeError(f"{type(self).__name__}.gpriors must be a "
                                "dictionary.")
            for key, value in gpriors.items():
                if key not in self.var_names:
                    raise ValueError(f"{key} not in "
                                     f"{type(self).__name__}.var_names.")
                if not isinstance(value, GaussianPrior):
                    raise TypeError(f"{type(self).__name__}.gpriors values "
                                    "must be instances of GaussianPrior.")
                if value.mat is None:
                    value.update_size(prod(self.var_shape))
                else:
                    if value.mat.shape[1] != prod(self.var_shape):
                        raise ValueError(f"gprior for {key} not match variable "
                                         "size.")
        self._gpriors = {}

    @upriors.setter
    def upriors(self, upriors: Optional[Dict[str, UniformPrior]]):
        if upriors is not None:
            if not isinstance(upriors, Dict):
                raise TypeError(f"{type(self).__name__}.gpriors must be a "
                                "dictionary.")
            for key, value in upriors.items():
                if key not in self.var_names:
                    raise KeyError(f"{key} not in "
                                   f"{type(self).__name__}.var_names.")
                if not isinstance(value, UniformPrior):
                    raise TypeError(f"{type(self).__name__}.upriors values "
                                    "must be instances of UniformPrior.")
                if value.mat is None:
                    value.update_size(prod(self.var_shape))
                else:
                    if value.mat.shape[1] != prod(self.var_shape):
                        raise ValueError(f"uprior for {key} not match variable "
                                         "size.")
        self._upriors = {}

    def __repr__(self) -> str:
        return (f"{type(self).__name__}(dims={self.dims}, meas={self.meas}, "
                f"prcs={self.prcs}")
