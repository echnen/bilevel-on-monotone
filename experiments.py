# -*- coding: utf-8 -*-
#
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
"""
Run this file to reproduce all numerical experiments in:

R. I. Bot, E. Chenchene, D. Hulett.
Regularized extragradient method for structured bilevel optimization in continuous and discrete time.
2026. DOI: XX.XXXXX/arXiv.XXXX.YYYYY.

For any comment, please contact: enis.chenchene@gmail.com
"""

import numpy as np
import optimization as opt
import plots as show
import instances as model
import ottools as ot


def experiment_optimal_assignment_bilevel(N, maxit):

    angles_mu = 2 * np.pi * np.arange(N) / N
    angles_nu = angles_mu + np.pi / N  # rotate by 30 degrees

    mu_weights = np.ones(N) / N
    mu_positions = np.column_stack((
        np.cos(angles_mu),
        np.sin(angles_mu),
    ))
    mu = (mu_weights, mu_positions)

    nu_weights = np.ones(N) / N
    nu_positions = np.column_stack((
        np.cos(angles_nu),
        np.sin(angles_nu),
    ))
    nu = (nu_weights, nu_positions)

    a, X = mu
    b, Y = nu

    m = len(a)
    n = len(b)

    A = ot.create_ot_constraint_matrix(m, n)
    C = ot.create_ot_cost_matrix(X, Y)

    c = C.ravel()        # row-major vectorization
    b = np.concatenate((a, b))

    gamma_opt, psi_opt = None, None
    Model = model.LP_Transport(A, b, c, gamma_opt, psi_opt)

    # step-size for proposed method
    sigma_e = 1
    c = 1
    s_BiEG = .9 / Model.Lip
    s_OEGH = 0.24 / Model.Lip

    # initialization
    psi_init = 5 * (np.random.rand(m + n) - 1)
    gam_init = 5 * np.random.rand(m * n) # / (m * n)

    # testing the following alphas
    delta = 0.35

    # first, find a true bilevel optimum
    print('Constructing optimum ...')
    Model.psi_opt = ot.optimal_dual_potential(N, alpha=0)
    Model.sig_opt = ot.optimal_primal(N)
    V_sig_opt, V_psi_opt = Model.V(Model.sig_opt, Model.psi_opt)
    p_sig, p_psi = Model.Prox(1, 0, Model.sig_opt - V_sig_opt, Model.psi_opt - V_psi_opt)
    print("optimality residual:",
          np.sqrt(np.sum((Model.sig_opt - p_sig)**2) + np.sum((Model.psi_opt - p_psi)**2)))
    print('Optimum solution constructed. Comparing BiEG with OEG-H ...')

    # non bilevel solution
    inner_gaps_nb, Inner_ress_nb, Outer_objs_nb, gam_fin_nb, psi_fin_nb \
        = opt.EG_primal_dual(psi_init, gam_init, s_BiEG, Model, maxit, opt_is_available=False)

    # proposed method
    eps_f = lambda k: c / (k + sigma_e + 1) ** delta
    Inner_gaps_BiEG, Inner_ress_BiEG, Outer_objs_BiEG, gam_fin_BiEG, psi_fin_BiEG \
        = opt.Bilevel_EG_primal_dual(psi_init, gam_init, s_BiEG, Model, maxit, eps_f, opt_is_available=True)

    # popov type approach
    Inner_gaps_OEGH, Inner_ress_OEGH, Outer_objs_OEGH, gam_fin_OEGH, psi_fin_OEGH = \
        opt.Bilevel_Popov_primal_dual(psi_init, gam_init, s_OEGH, Model, maxit, eps_f, opt_is_available=True)

    Inner_gaps_BiEG = np.array(Inner_gaps_BiEG)
    Inner_gaps_OEGH = np.array(Inner_gaps_OEGH)
    print(f'Negative values: {Inner_gaps_OEGH[Inner_gaps_OEGH < -1e-8]}')
    print(f'Negative values: {Inner_gaps_BiEG[Inner_gaps_BiEG < -1e-8]}')

    show.plot_ot_maps(mu, nu,
                      [gam_fin_nb, gam_fin_BiEG, gam_fin_OEGH])

    show.plot_ot_comparison(delta,
                            Inner_gaps_BiEG, Inner_ress_BiEG, Outer_objs_BiEG,
                            Inner_gaps_OEGH, Inner_ress_OEGH, Outer_objs_OEGH,
                            maxit)

    print(f'\n *********************** \n Optimal Bilevel Primal Solution: \n {Model.sig_opt}')
    print(f'\n *********************** \n Final Bilevel Primal Solution: \n {gam_fin_BiEG}')
    print(f'\n *********************** \n Final Bilevel Dual Solution: \n {psi_fin_BiEG}')



def experiment_toy_example(dim=100, maxit=1e5, s=None, c=1.0, sigma_e=1.0):

    np.random.seed(0)
    Model = model.Toy_Example(dim)
    Lip = Model.Lip
    eps_f = lambda k: c / (k + sigma_e + 1) ** delta

    # step size selection
    s_eg = 0.99 / Lip
    s_popov = 0.249 / Lip

    Deltas = [0.2, 0.35, 0.5]
    Tries = 20
    Metrics_EG = np.zeros((Tries, 3, 3, maxit))
    Metrics_Popov = np.zeros((Tries, 3, 3, maxit))

    for num_delta, delta in enumerate(Deltas):

        print(f'Performing the case delta = {delta} ...')
        for num_try in range(Tries):

            # sample initial point
            x_init = 10 * (np.random.rand(dim) -  0.5)

            # proposed eg method
            inner_gaps_EG, inner_ress_EG, outer_objs_EG, _ = opt.Bilevel_EG(x_init=x_init, s=s_eg, Model=Model, maxit=maxit, eps_f=eps_f)
            Metrics_EG[num_try, num_delta, 0, :] = inner_gaps_EG
            Metrics_EG[num_try, num_delta, 1, :] = inner_ress_EG
            Metrics_EG[num_try, num_delta, 2, :] = outer_objs_EG

            # optimistic method
            inner_gaps_popov, inner_ress_popov, outer_objs_popov, _ = opt.Bilevel_Popov(x_init=x_init, s=s_popov, Model=Model, maxit=maxit, eps_f=eps_f)
            Metrics_Popov[num_try, num_delta, 0, :] = inner_gaps_popov
            Metrics_Popov[num_try, num_delta, 1, :] = inner_ress_popov
            Metrics_Popov[num_try, num_delta, 2, :] = outer_objs_popov


    show.plot_toy_example(Metrics_EG, Metrics_Popov, maxit, Deltas)

    return


def experiment_trade_off(dim=100, maxit=1000, s=None, c=1.0, sigma_e=1.0):

    Model = model.Toy_Example(dim)
    Lip = Model.Lip
    print(Model.off_set)
    print(Model.x_opt)

    # step size selection
    s_eg = 0.99 / Lip

    Tries = 20
    Metrics_EG = np.zeros((Tries, 3, maxit))
    Deltas = np.linspace(0.1, 0.5, Tries)

    for num_try in range(Tries):

        delta = Deltas[num_try]
        eps_f = lambda k: c / (k + sigma_e + 1) ** delta

        # sample initial point
        x_init = 5 * (np.random.rand(dim) -  0.5)

        # proposed eg method
        inner_gaps_EG, inner_ress_EG, outer_objs_EG, _ = opt.Bilevel_EG(x_init=x_init, s=s_eg, Model=Model, maxit=maxit, eps_f=eps_f)
        Metrics_EG[num_try, 0, :] = inner_gaps_EG
        Metrics_EG[num_try, 1, :] = inner_ress_EG
        Metrics_EG[num_try, 2, :] = outer_objs_EG

    show.plot_delta_experiments(Metrics_EG, Deltas, maxit)

    return
