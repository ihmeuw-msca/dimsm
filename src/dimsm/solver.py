"""
Customized Interior Point Solver
================================

Solver class solves large scale sparse least square problem with linear
constraints.
"""
from typing import List, Optional
import numpy as np
from scipy.optimize import LinearConstraint
from scipy.sparse import vstack, csc_matrix
from scipy.sparse.linalg import spsolve


class IPSolver:
    """Interior point solver for large sparse quadratic system with linear
    constraints.

    Parameters
    ----------
    h_mat : csc_matrix
        Quadratic matrix in the objective function.
    g_vec : np.ndarray
        Linear vector in the objective function.
    linear_constraints : Optional[LinearConstraint], optional
        Linear constraints for the problem. Default to be `None`. If it is
        `None`, solver will use a simple linear solve.

    Attributes
    ----------
    h_mat : csc_matrix
        Quadratic matrix in the objective function.
    g_vec : np.ndarray
        Linear vector in the objective function.
    linear_constraints : Optional[LinearConstraint]
        Linear constraints for the problem.
    c_mat : Optional[csc_matrix]
        Constraint matrix, when `linear_constraints` is `None`, `c_mat` will be
        `None` as well.
    c_vec : Optional[csc_matrix]
        Constraint vector, when `linear_constraints` is `None`, `c_mat` will be
        `None` as well.

    Methods
    -------
    get_kkt(p, mu)
        Get the KKT system.
    get_step(p, dp, scale=0.99)
        Get the step size.
    minimize(xtol=1e-8, gtol=1e-8, max_iter=100, mu=1.0, scale_mu=0.1,
             scale_step=0.99, verbose=False)
        Minimize the quadratic objective over linear constraints.
    """

    def __init__(self,
                 h_mat: csc_matrix,
                 g_vec: np.ndarray,
                 linear_constraints: Optional[LinearConstraint] = None):
        self.h_mat = h_mat
        self.g_vec = g_vec
        self.linear_constraints = linear_constraints
        self.c_mat = None
        self.c_vec = None

        if self.linear_constraints is not None:
            mat = csc_matrix(self.linear_constraints.A)
            lb = self.linear_constraints.lb
            ub = self.linear_constraints.ub

            self.c_mat = csc_matrix(vstack([-mat[~np.isneginf(lb)],
                                            mat[~np.isposinf(ub)]]))
            self.c_vec = np.hstack([-lb[~np.isneginf(lb)],
                                    ub[~np.isposinf(ub)]])

    def get_kkt(self,
                p: List[np.ndarray],
                mu: float) -> List[np.ndarray]:
        """Get the KKT system.

        Parameters
        ----------
        p : List[np.ndarray]
            A list a parameters, including x, s, and v, where s is the slackness
            variable and v is the dual variable for the constraints.
        mu : float
            Interior point method barrier variable.

        Returns
        -------
        List[np.ndarray]
            The KKT system with three components.
        """
        return [
            self.c_mat.dot(p[0]) + p[1] - self.c_vec,
            p[1]*p[2] - mu,
            self.h_mat.dot(p[0]) + self.g_vec + self.c_mat.T.dot(p[2])
        ]

    def get_step(self,
                 p: List[np.ndarray],
                 dp: List[np.ndarray],
                 scale: float = 0.99) -> float:
        """Get the step size.

        Parameters
        ----------
        p : List[np.ndarray]
            A list a parameters, including x, s, and v, where s is the slackness
            variable and v is the dual variable for the constraints.
        dp : List[np.ndarray]
            A list of direction for the parameters.
        scale : float, optional
            Shrinkage factor for the step size, by default 0.99.

        Returns
        -------
        float
            The step size in the given direction.
        """
        a = 1.0
        for i in [1, 2]:
            indices = dp[i] < 0.0
            if not any(indices):
                continue
            a = scale*np.minimum(a, np.min(-p[i][indices] / dp[i][indices]))
        return a

    def minimize(self,
                 xtol: float = 1e-8,
                 gtol: float = 1e-8,
                 max_iter: int = 100,
                 mu: float = 1.0,
                 scale_mu: float = 0.1,
                 scale_step: float = 0.99,
                 verbose: bool = False) -> np.ndarray:
        """Minimize the quadratic objective over linear constraints.

        Parameters
        ----------
        xtol : float, optional
            Tolerance for the differences in `x`, by default 1e-8.
        gtol : float, optional
            Tolerance for the KKT system, by default 1e-8.
        max_iter : int, optional
            Maximum number of iterations, by default 100.
        mu : float, optional
            Initial interior point bairrier parameter, by default 1.0.
        scale_mu : float, optional
            Shrinkage factor for mu updates, by default 0.1
        scale_step : float, optional
            Shrinkage factor for step size updates, by default 0.99
        verbose : bool, optional
            Indicator of if print out convergence history, by default False

        Returns
        -------
        np.ndarray
            Solution vector.
        """
        if self.linear_constraints is None:
            if verbose:
                print(f"{type(self).__name__}: no constraints, using simple "
                      "linear solve.")
            return -spsolve(self.h_mat, self.g_vec)

        # initialize the parameters
        p = [
            np.zeros(self.g_vec.size),
            np.ones(self.c_vec.size),
            np.ones(self.c_vec.size)
        ]

        f = self.get_kkt(p, mu)
        gnorm = np.max(np.abs(np.hstack(f)))
        xdiff = 1.0
        step = 1.0
        counter = 0

        if verbose:
            print(f"{type(self).__name__}:")
            print(f"{counter=:3d}, {gnorm=:.2e}, {xdiff=:.2e}, {step=:.2e}, "
                  f"{mu=:.2e}")

        while (gnorm > gtol) and (xdiff > xtol) and (counter < max_iter):
            counter += 1

            # cache convenient variables
            sv_vec = p[2] / p[1]
            sf2_vec = f[1] / p[1]
            csv_mat = self.c_mat.copy()
            csv_mat.data *= np.take(sv_vec, csv_mat.indices)

            # compute all directions
            mat = self.h_mat + csv_mat.T.dot(self.c_mat)
            vec = -f[2] + self.c_mat.T.dot(sf2_vec - sv_vec*f[0])
            dx = spsolve(mat, vec)
            ds = -f[0] - self.c_mat.dot(dx)
            dv = -sf2_vec - sv_vec*ds
            dp = [dx, ds, dv]

            # get step size
            step = self.get_step(p, dp, scale=scale_step)

            # update parameters
            for i in range(len(p)):
                p[i] += step * dp[i]

            # update mu
            mu = scale_mu*p[1].dot(p[2])/len(p[1])

            # update f and gnorm
            f = self.get_kkt(p, mu)
            gnorm = np.max(np.abs(np.hstack(f)))
            xdiff = step*np.max(np.abs(dp[0]))

            if verbose:
                print(f"{counter=:3d}, {gnorm=:.2e}, {xdiff=:.2e}, "
                      f"{step=:.2e}, {mu=:.2e}")

        return p[0]
