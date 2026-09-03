#    Copyright (C) 2026 Radu Ioan Bot (radu.bot@univie.ac.at)
#                       Enis Chenchene (enis.chenchene@univie.ac.at)
#                       David Hulett (david.hulett@univie.ac.at)
#
#    This file is part of the example code repository for the paper:
#
#      R. I. Bot, E. Chenchene, D. Hulett.
#      Regularized extragradient method for structured bilevel optimization in continuous and discrete time.
#      2026. DOI: 10.48550/arXiv.2608.29181.
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
"""Compute the one-dimensional total-variation proximal operator."""

import numpy as np


def prox_tv1d(x, tau):
    """Compute the proximal operator of the 1D total-variation norm.
    
    This implements the efficient algorithm from the literature for computing
    the proximal operator of tau * ||.||_TV, where ||.||_TV is the 1D total-variation norm.
    
    Parameters
    ----------
    x : array_like
        Input array.
    tau : float
        Regularization parameter (must be non-negative).
    
    Returns
    -------
    ndarray
        The proximal operator applied to x.
    
    Raises
    ------
    ValueError
        If tau is negative.
    """
    x = np.asarray(x, dtype=float)
    n = x.size

    if tau < 0:
        raise ValueError("tau must be non-negative.")
    if n == 0 or tau == 0:
        return x.copy()

    y = np.empty(n)
    k = k0 = kplus = kminus = 0
    vmin, vmax = x[0] - tau, x[0] + tau
    umin, umax = tau, -tau

    while True:
        if k == n - 1:
            if umin < 0:
                y[k0:kminus + 1] = vmin
                k0 = kminus + 1
                k = k0
                if k >= n:
                    return y
                kplus = kminus = k
                vmin, vmax = x[k], x[k] + 2 * tau
                umin, umax = tau, -tau
            elif umax > 0:
                y[k0:kplus + 1] = vmax
                k0 = kplus + 1
                k = k0
                if k >= n:
                    return y
                kplus = kminus = k
                vmin, vmax = x[k] - 2 * tau, x[k]
                umin, umax = tau, -tau
            else:
                vmin += umin / (k - k0 + 1)
                y[k0:k + 1] = vmin
                return y
        else:
            k += 1

            if x[k] + umin < vmin - tau:
                y[k0:kminus + 1] = vmin
                k0 = kminus + 1
                k = k0
                kplus = kminus = k
                vmin, vmax = x[k], x[k] + 2 * tau
                umin, umax = tau, -tau

            elif x[k] + umax > vmax + tau:
                y[k0:kplus + 1] = vmax
                k0 = kplus + 1
                k = k0
                kplus = kminus = k
                vmin, vmax = x[k] - 2 * tau, x[k]
                umin, umax = tau, -tau

            else:
                umin += x[k] - vmin
                umax += x[k] - vmax

                if umin >= tau:
                    vmin += (umin - tau) / (k - k0 + 1)
                    umin = tau
                    kminus = k

                if umax <= -tau:
                    vmax += (umax + tau) / (k - k0 + 1)
                    umax = -tau
                    kplus = k
