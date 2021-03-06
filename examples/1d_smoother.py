from functools import partial

import numpy as np
import pandas as pd

from dimsm.dimension import Dimension
from dimsm.measurement import Measurement
from dimsm.process import Process, default_gen_vmat
from dimsm.smoother import Smoother


# generate simulate data
x = np.linspace(0, 1, 50)
s = 0.1
y = np.sin(2*np.pi*x) + np.random.randn(x.size)*s

# create dimension(s)
dims = [Dimension("x", grid=np.linspace(0, 1, 30))]

# create measurement data
df = pd.DataFrame({"x": x, "y": y})
meas = Measurement(df, imat=1/s**2)

# create process
order = 1
gen_vmat = partial(default_gen_vmat, size=order + 1, sigma=10.0)
prcs = {"x": Process(order=order, gen_vmat=gen_vmat)}

# create smoother
smoother = Smoother(dims, meas, prcs=prcs)

# fit model
smoother.fit(verbose=True)

# extract state variable on the grid
state = smoother.opt_vars[0]
