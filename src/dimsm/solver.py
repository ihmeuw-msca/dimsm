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
    def __init__(self,
                 h_mat: np.ndarray,
                 g_vec: np.ndarray,
                 linear_constraints: Optional[LinearConstraint] = None):
        self.h_mat = h_mat
        self.g_vec = g_vec
        self.linear_constraints = linear_constraints

        if self.linear_constraints is not None:
            mat = csc_matrix(self.linear_constraints.A)
            lb = self.linear_constraints.lb
            ub = self.linear_constraints.ub

            self.c_mat = csc_matrix(vstack([-mat[~np.isneginf(lb)],
                                            mat[~np.isposinf(ub)]]))
            self.c_vec = np.hstack([-lb[~np.isneginf(lb)],
                                    ub[~np.isposinf(ub)]])

    def get_f(self,
              p: List[np.ndarray],
              mu: float) -> List[np.ndarray]:
        return [
            self.c_mat.dot(p[0]) + p[1] - self.c_vec,
            p[1]*p[2] - mu,
            self.h_mat.dot(p[0]) + self.g_vec + self.c_mat.T.dot(p[2])
        ]

    def get_a(self,
              p: List[np.ndarray],
              dp: List[np.ndarray],
              scale: float = 0.99) -> float:
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
                 scale_a: float = 0.99,
                 verbose: bool = False):
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

        f = self.get_f(p, mu)
        error = np.max(np.abs(np.hstack(f)))
        xdiff = 1.0
        counter = 0

        if verbose:
            print(f"{type(self).__name__}:")
            print("iter  error     xdiff      mu")
            print(f"{counter:4d}  {error:.2e}  {xdiff:.2e}  {mu: .2e}")

        while (error > gtol) and (xdiff > xtol) and (counter < max_iter):
            counter += 1
            # cache convenient variables
            sv_vec = p[2] / p[1]

            sf2_vec = f[1] / p[1]

            csv_mat = self.c_mat.copy()
            csv_mat.data *= np.take(sv_vec, csv_mat.indices)

            cv_mat = self.c_mat.copy()
            cv_mat.data *= np.take(p[2], cv_mat.indices)

            # compute all directions
            mat = self.h_mat + csv_mat.T.dot(self.c_mat)
            vec = -f[2] + self.c_mat.T.dot(sf2_vec - sv_vec*f[0])
            dx = spsolve(mat, vec)
            ds = -f[0] - self.c_mat.dot(dx)
            dv = -sf2_vec - sv_vec*ds
            dp = [dx, ds, dv]

            # get step size
            a = self.get_a(p, dp, scale=scale_a)

            # update parameters
            for i in range(len(p)):
                p[i] += a * dp[i]

            # update mu
            mu = scale_mu*p[1].dot(p[2])/len(p[1])

            # update f and error
            f = self.get_f(p, mu)
            error = np.max(np.abs(np.hstack(f)))
            xdiff = a*np.max(np.abs(dp[0]))

            if verbose:
                print(f"{counter:4d}  {error:.2e}  {xdiff:.2e}  {mu: .2e}")

        return p[0]
