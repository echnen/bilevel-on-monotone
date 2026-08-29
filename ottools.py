#    Copyright (C) 2026 Radu Ioan Bot (radu.bot@univie.ac.at)
#                       Enis Chenchene (enis.chenchene@univie.ac.at)
#                       David Hulett (david.hulett@univie.ac.at)
#
#    This file is part of the example code repository for the paper:
#
#      R. I. Bot, E. Chenchene, D. Hulett.
#      Regularized extragradient method for structured bilevel optimization in continuous and discrete time.
#      2026. DOI: XX.XXXXX/arXiv.XXXX.YYYYY.
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""Provide construction and reference-solution utilities for optimal transport."""

import numpy as np
import scipy.sparse as sp


def create_ot_constraint_matrix(m, n):
    """
    Creates the sparse constraint matrix A such that

        A @ vec(P) = [a; b],

    where P is an (m,n) transport matrix and vec(P) uses
    NumPy's default row-major ordering.

    Parameters
    ----------
    m : int
        Number of source points.
    n : int
        Number of target points.

    Returns
    -------
    A : scipy.sparse.csr_matrix, shape (m +n, m * n)
    """

    # Row sums: P 1 = a
    A_rows = sp.kron(sp.eye(m, format="csr"), np.ones((1, n)), format="csr")

    # Column sums: P^T 1 = b
    A_cols = sp.kron(np.ones((1, m)), sp.eye(n, format="csr"), format="csr")

    return sp.vstack((A_rows, A_cols), format="csr")


def create_ot_cost_matrix(X, Y, squared=True):
    """
    Creates the OT cost matrix between two point clouds.

    """

    diff = X[:, None, :] - Y[None, :, :]
    C = np.sum(diff**2, axis=2)

    if not squared:
        C = np.sqrt(C)

    return C


def optimal_dual_potential(N, alpha=0.0):

    c_min = 4.0 * np.sin(np.pi / (2.0 * N)) ** 2
    u = alpha * np.ones(N)
    v = (-c_min - alpha) * np.ones(N)

    return np.concatenate((u, v))


def optimal_primal(N):

    Gamma = np.roll(np.eye(N), shift=-1, axis=1) / N

    return Gamma.ravel()
