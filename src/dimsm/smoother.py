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
    gpriors : Optional[Dict[str, Dict[int, GaussianPrior]]], optional
        Gaussian priors for each variable. It requires dimension name as the key
        and another dictionary as the value. For each dimension, it requires the
        index as the key and instance of GaussianPrior as the value. The index
        cannot exceed the order of the corresponding process. Default to be
        `None`.
    upriors : Optional[Dict[str, Dict[int, UniformPrior]]], optional
        Uniform priors for each variable. Same specification as in `gpriors`.
        Default to be `None`.


    Attributes
    ----------
    dims : List[Dimension]
        List of different dimensions.
    meas : Measurement
        Measurements of the model.
    prcs : Dict[str, Process]
        Dictionary of processes of the model.
    gpriors : Dict[str, Dict[int, GaussianPrior]]
        Gaussian priors for each variable.
    upriors : Dict[str, Dict[int, UniformPrior]]
        Uniform priors for each variable.
    dim_names : List[str]
        Names of the dimensions
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
                 gpriors: Optional[Dict[str, Dict[int, GaussianPrior]]] = None,
                 upriors: Optional[Dict[str, Dict[int, UniformPrior]]] = None):
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
    def prcs(self, prcs: Optional[Dict[Process]]):
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
            self._prcs = prcs
        self._prcs = {}

    @gpriors.setter
    def gpriors(self, gpriors: Optional[Dict[str, Dict[int, GaussianPrior]]]):
        if gpriors is not None:
            if not isinstance(gpriors, Dict):
                raise TypeError(f"{type(self).__name__}.gpriors must be a "
                                "dictionary.")
            for key, value in gpriors.items():
                if key not in self.prcs:
                    raise ValueError(f"{key} not in "
                                     f"{type(self).__name__}.prcs.")
                for index, prior in value.items():
                    if not isinstance(prior, GaussianPrior):
                        raise TypeError(f"{type(self).__name__}.gpriors values "
                                        "must be instances of GaussianPrior.")
                    if index > self.prcs[key].order:
                        raise IndexError(f"Index exceed process {key} order.")
                    if prior.mat is None:
                        prior.update_size(np.prod(self.var_shape))
                    else:
                        if prior.mat.shape[1] != np.prod(self.var_shape):
                            raise ValueError(f"gprior for {key} not match "
                                             "variable size.")
        self._gpriors = {}

    @upriors.setter
    def upriors(self, upriors: Optional[Dict[str, Dict[int, UniformPrior]]]):
        if upriors is not None:
            if not isinstance(upriors, Dict):
                raise TypeError(f"{type(self).__name__}.gpriors must be a "
                                "dictionary.")
            for key, value in upriors.items():
                if key not in self.prcs:
                    raise KeyError(f"{key} not in "
                                   f"{type(self).__name__}.prcs.")
                for index, prior in value.items():
                    if not isinstance(prior, UniformPrior):
                        raise TypeError(f"{type(self).__name__}.upriors values "
                                        "must be instances of UniformPrior.")
                    if index > self.prcs[key].order:
                        raise IndexError(f"Index exceed process {key} order.")
                    if prior.mat is None:
                        prior.update_size(np.prod(self.var_shape))
                    else:
                        if prior.mat.shape[1] != np.prod(self.var_shape):
                            raise ValueError(f"uprior for {key} not match "
                                             "variable size.")
        self._upriors = {}

    def __repr__(self) -> str:
        return (f"{type(self).__name__}(dims={self.dims}, meas={self.meas}, "
                f"prcs={self.prcs}")
