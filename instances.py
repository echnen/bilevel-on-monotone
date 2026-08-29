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
"""Define the optimal-transport and toy bilevel problem instances."""

import numpy as np
from scipy import sparse as sp
from scipy.sparse.linalg import norm as sparse_norm

import structures as st


class LP_Transport:
    '''
    Inner : min of c^T x s.t. Ax = b, x \geq 0
    Outer : |x|^2

    '''

    def __init__(self, A, b, c, gamma_opt, psi_opt):

        np.random.seed(2)

        # building operator
        self.A = A
        self.b = b
        self.c = c

        self.Lip = sparse_norm(A.astype(float), ord=2)

        self.gamma_opt = gamma_opt
        self.psi_opt = psi_opt

        # subgradients
        self.dh_hat_available = True
        self.df_hat_available = False

    def Prox(self, s, eps_k, in_prox_gam, in_prox_psi):

        in_prox_gam = np.copy(in_prox_gam)
        in_prox_psi = np.copy(in_prox_psi)

        out_prox_gam = np.maximum(in_prox_gam - s * self.c, 0)
        out_prox_psi = in_prox_psi - s * self.b

        return out_prox_gam, out_prox_psi

    def dh(self, gamma, psi):

        out_gamma = np.zeros_like(gamma)
        out_gamma[0] = gamma[0]

        return out_gamma, np.zeros_like(psi)

    def V(self, gamma, psi):

        gamma_out = self.A.T @ psi
        psi_out = -self.A @ gamma

        return gamma_out, psi_out

    def H(self, gamma, psi):

        return 0.5 * gamma[0] ** 2 # 0.5 * np.sum(gamma ** 2)

    def f_hat(self, gamma, psi):

        return np.dot(self.c, gamma) + np.dot(self.b, psi)

    def dh_hat(self, gamma, psi):

        return np.zeros_like(gamma), np.zeros_like(psi)


class Toy_Example:
    """
    Toy example formulated as:

    min_x TV(x)
    s.t.  Vx = off_set

    """

    def __init__(self, dim, x_opt=None):

        # creating operator
        dim_effective = int(0.75 * (dim))
        diag_low = np.zeros(dim - 1)
        diag_low[:dim_effective - 1] = 1
        diag_high = np.zeros(dim - 1)
        diag_high[:dim_effective - 1] = -1
        self.V_mat = sp.diags(diagonals=[diag_low, diag_high], offsets=[-1, 1], format="csr") # (dim, dim)
        self.dim_effective = dim_effective

        # creating offset
        x_opt = np.zeros(dim)
        x_opt[dim_effective - 1] = 1
        x_opt[0] = 20
        self.off_set = self.V_mat @ x_opt

        # creating optimum
        x_opt[dim_effective:] = 0.5
        self.x_opt = x_opt

        # availability subgradients
        self.dh_hat_available = False
        self.df_hat_available = True

        # Lipschitz constant
        self.Lip = sp.linalg.norm(self.V_mat, ord=2) + 1


    def proj(self, x):

        x_out = np.copy(x)
        x_out[self.dim_effective:] = np.minimum(x[self.dim_effective:], 0.5)

        return x_out

    def V(self, x):

        return self.V_mat @ x - self.off_set + x - self.proj(x)

    def dh(self, x):

        return 0

    def Prox(self, s, eps_k, z):

        return st.prox_tv1d(z, eps_k * s)

    def f_hat(self, x):

        return 0

    def dh_hat(self, x):

        return np.zeros_like(x)

    def df_hat(self, x):

        return np.zeros_like(x)

    def H(self, x):

        return np.sum(np.abs(np.diff(x)))
