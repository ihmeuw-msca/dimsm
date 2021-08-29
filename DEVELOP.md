# Developer Notes and Logs

We document different design ideas in this file.

## Package Description

`dimsm` package is intend to solve smoothing problem across different
dimensions.
For example, we want to smooth the data in age, time dimensions.
There are existing packages and methods to deal with similar issues, for
example Gaussian Process and Kalman Smoother.
But we need the capability to specific flexible priors and constraints (e.g.
cohort constraints).
And that motivates us to create Kalman-based `dimsm` package.
The grid setting is inspired by the package
[`dismod_at`](https://github.com/bradbell/dismod_at).

## Use Cases

The user for this package will mainly be IHME researchers.
And the design of the package and new features to be added will be guided
by IHME projects need.
However the package is fairly general, broader audiences can use this package on
their own projects.

### Age Time Smoothing

One main application for this package is the age-time smoothing.
Imagine we collect data from different time points and across different age
groups.
One way to smooth the data and reduce the error is to introduce correlation
across time and age dimensions.
Kalman Smoother is a non-parametric method to achieve this goal.


## Functionality

Here we specify the input and desired behavior of the problem.

### Inputs

* Dimension settings (grid for age and time)
* Observations and their (co)variance structure across age and time
* Process settings (process matrix and (co)variance matrix)
* Priors (on the states and/or on dimension variables)

### Methods

* Fit current data
* Predict giving a data table
* Create draws giving a data table

## Design and Structure

We describe package classes and their purpose in the following

### Smoother
`Smoother` class is used to gather all information from different components.
And it provides model interface, `fit`, `predict` and `get_draws` functions.

### Dimension
A simple class that contains the name and the grids of the dimension.

### Measurement
A data class contains the measurements and (co)variance matrix.

### Process
Kalman-based process, it carries name of the dimension, process matrix and the
process (co)variance matrix.

### Priors
Priors for the states and the dimension variables.
The priors include Gaussian priors (regularizers) and Uniform priors
(constraints).
