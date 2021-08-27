"""
Dimension
=========

Record the name and grid information of the dimension.
"""
from operator import attrgetter
from typing import List
from dataclasses import dataclass

import numpy as np


@dataclass
class Dimension:
    """Dimension class records the name and grid information of the dimension.

    Parameters
    ----------
    name : str
        Name of the dimension. Name need to be a string.
    grid : List[float]
        Grid of the dimension. Grid need to contain at least two unique values.

    Attributes
    ----------
    name : str
        Name of the dimension.
    grid : List[float]
        Grid of the dimension.

    Raises
    ------
    TypeError
        Raised when input name is not a string.
    ValueError
        Raised when input grid doesn't have at least two unique values. This
        means that the dimension is degenerate.
    """

    name = property(attrgetter("_name"))
    grid = property(attrgetter("_grid"))

    name: str
    grid: List[float]

    @name.setter
    def name(self, name: str) -> None:
        if not isinstance(name, str):
            raise TypeError(f"{type(self).__name__}.name has to be a string.")
        self._name = name

    @grid.setter
    def grid(self, grid: List[float]) -> None:
        grid = np.unique(grid).astype(float)
        if grid.size <= 1:
            raise ValueError(f"{type(self).__name__}.grid needs to have at "
                             "least two unique points. Otherwise dimension is "
                             "degenerate.")
        self._grid = grid
