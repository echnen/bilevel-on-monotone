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
"""Implement the extragradient and Popov algorithms used in the experiments."""

import numpy as np


def Bilevel_EG(x_init, s, Model, maxit, eps_f):
    '''
    Algorithm in our paper.
    '''

    # storage
    Inner_gaps = []
    Inner_ress = []
    Outer_objs = []

    # initialize
    x_old = np.copy(x_init)
    x = np.copy(x_init)

    for k in range(maxit):

        eps_k = eps_f(k) # c / (k + sigma_e + 1) ** delta

        in_prox_y = x - s * (Model.V(x) + eps_k * Model.dh(x))
        y = Model.Prox(s, eps_k, in_prox_y)

        in_prox_x = x - s * (Model.V(y) + eps_k * Model.dh(y))
        x = Model.Prox(s, eps_k, in_prox_x)

        ## producing residuals
        # inner gap
        inner_gap = np.dot(x - Model.x_opt, Model.V(x)) + Model.f_hat(x) - Model.f_hat(Model.x_opt)

        # inner residual
        if Model.df_hat_available:
            xi_f_hat = Model.df_hat(x)

        elif Model.dh_hat_available:
            xi_k = (in_prox_x - x) / s
            xi_f_hat = xi_k - eps_k * Model.dh_hat(x)

        inner_res = np.sqrt(np.sum((Model.V(x) + xi_f_hat) ** 2))

        # outer objective
        outer_obj = Model.H(x) - Model.H(Model.x_opt)

        Inner_gaps.append(inner_gap)
        Inner_ress.append(inner_res)
        Outer_objs.append(outer_obj)

    return Inner_gaps, Inner_ress, Outer_objs, x


def Bilevel_EG_primal_dual(psi_init, sig_init, s, Model, maxit, eps_f, opt_is_available=False):
    '''
    Algorithm in our paper.
    '''

    # storage
    Inner_gaps = []
    Inner_ress = []
    Outer_objs = []

    # initialize
    x_sig = np.copy(sig_init) # primal variable
    x_psi = np.copy(psi_init) # dual variable

    for k in range(maxit):

        x_sig_old = np.copy(x_sig)
        x_psi_old = np.copy(x_psi)

        eps_k = eps_f(k) # c / (k + sigma_e + 1) ** delta

        V_x_sig_out, V_x_psi_out = Model.V(x_sig, x_psi)
        Grad_x_sig_out, Grad_x_psi_out = Model.dh(x_sig, x_psi)
        in_prox_y_sig = x_sig - s * (V_x_sig_out + eps_k * Grad_x_sig_out)
        in_prox_y_psi = x_psi - s * (V_x_psi_out + eps_k * Grad_x_psi_out)

        y_sig, y_psi = Model.Prox(s, eps_k, in_prox_y_sig, in_prox_y_psi)

        V_y_sig_out, V_y_psi_out = Model.V(y_sig, y_psi)
        Grad_y_sig_out, Grad_y_psi_out = Model.dh(y_sig, y_psi)
        in_prox_x_sig = x_sig - s * (V_y_sig_out + eps_k * Grad_y_sig_out)
        in_prox_x_psi = x_psi - s * (V_y_psi_out + eps_k * Grad_y_psi_out)

        x_sig, x_psi = Model.Prox(s, eps_k, in_prox_x_sig, in_prox_x_psi)

        ## producing residuals
        V_x_sig_out, V_x_psi_out = Model.V(x_sig, x_psi)

        # inner residual
        if Model.df_hat_available:
            xi_f_hat_sig, xi_f_hat_psi = Model.df_hat(x_sig, x_psi)

        elif Model.dh_hat_available:
            dh_x_sig, dh_x_psi = Model.dh_hat(x_sig, x_psi)
            xi_k_sig = (in_prox_y_sig - x_sig) / s
            xi_k_psi = (in_prox_x_psi - x_psi) / s

            xi_f_hat_sig = xi_k_sig - eps_k * dh_x_sig
            xi_f_hat_psi = xi_k_psi - eps_k * dh_x_psi

        inner_res = np.sum((V_x_sig_out + xi_f_hat_sig) ** 2) + np.sum((V_x_psi_out + xi_f_hat_psi) ** 2)
        Inner_ress.append(inner_res)

        if opt_is_available:

            # inner gap
            inner_gap = np.sum((x_sig - Model.sig_opt) * V_x_sig_out) \
                + np.sum((x_psi - Model.psi_opt) * V_x_psi_out) \
                + Model.f_hat(x_sig, x_psi) - Model.f_hat(Model.sig_opt, Model.psi_opt)
            Inner_gaps.append(inner_gap)

            # outer objective
            outer_obj = Model.H(x_sig, x_psi) - Model.H(Model.sig_opt, Model.psi_opt)
            Outer_objs.append(outer_obj)


    return Inner_gaps, Inner_ress, Outer_objs, x_sig, x_psi


def EG_primal_dual(psi_init, sig_init, s, Model, maxit, opt_is_available=False):
    '''
    Standard EG Algorithm.
    '''

    # storage
    Inner_gaps = []
    Inner_ress = []
    Outer_objs = []

    # initialize
    x_sig = np.copy(sig_init) # primal variable
    x_psi = np.copy(psi_init) # dual variable

    for k in range(maxit):

        V_x_sig_out, V_x_psi_out = Model.V(x_sig, x_psi)
        in_prox_y_sig = x_sig - s * V_x_sig_out
        in_prox_y_psi = x_psi - s * V_x_psi_out

        y_sig, y_psi = Model.Prox(s, 0, in_prox_y_sig, in_prox_y_psi)

        V_y_sig_out, V_y_psi_out = Model.V(y_sig, y_psi)
        in_prox_x_sig = x_sig - s * V_y_sig_out
        in_prox_x_psi = x_psi - s * V_y_psi_out

        x_sig, x_psi = Model.Prox(s, 0, in_prox_x_sig, in_prox_x_psi)

        ## producing residuals
        dh_x_sig, dh_x_psi = Model.dh_hat(x_sig, x_psi)
        V_x_sig_out, V_x_psi_out = Model.V(x_sig, x_psi)

        # inner residual
        xi_k_sig = (in_prox_x_sig - x_sig) / s
        xi_k_psi = (in_prox_x_psi - x_psi) / s

        inner_res = np.sum((V_x_sig_out + xi_k_sig) ** 2) + np.sum((V_x_psi_out + xi_k_psi) ** 2)
        Inner_ress.append(inner_res) # inner_res

        if opt_is_available:

            # inner gap
            inner_gap = np.dot(x_sig - Model.sig_opt, V_x_sig_out) \
                + np.dot(x_psi - Model.psi_opt, V_x_psi_out) \
                + Model.f_hat(x_sig, x_psi) - Model.f_hat(Model.sig_opt, Model.psi_opt)
            Inner_gaps.append(inner_gap)

            # outer objective
            outer_obj = Model.H(x_sig, x_psi) - Model.H(Model.sig_opt, Model.psi_opt)
            Outer_objs.append(outer_obj)


    return Inner_gaps, Inner_ress, Outer_objs, x_sig, x_psi


def Bilevel_Popov(x_init, s, Model, maxit, eps_f):
    '''
    Optimistic extragradient method for hierarchical HVI's;
    Algorithm 1 in Dvurechensky, Marschner, Shtern, and Staudigl; '26
    '''

    # storage
    Inner_gaps = []
    Inner_ress = []
    Outer_objs = []

    # initialize
    x_old = np.copy(x_init)
    x = np.copy(x_init)
    y = np.copy(x_init)

    for k in range(maxit):

        eps_k = eps_f(k) # c / (k + sigma_e + 1) ** delta

        in_prox_y = x - s * (Model.V(y) + eps_k * Model.dh(y))
        y = Model.Prox(s, eps_k, in_prox_y)

        in_prox_x = x - s * (Model.V(y) + eps_k * Model.dh(y))
        x = Model.Prox(s, eps_k, in_prox_x)

        ## producing residuals
        # inner gap
        inner_gap = np.dot(x - Model.x_opt, Model.V(x)) \
            + Model.f_hat(x) - Model.f_hat(Model.x_opt)

        # inner residual
        if Model.df_hat_available:
            xi_f_hat = Model.df_hat(x)
            inner_res = np.sqrt(np.sum((Model.V(x) + xi_f_hat) ** 2))

        elif Model.dh_hat_available:
            xi_k = (in_prox_x - x) / s
            xi_f_hat = xi_k - eps_k * Model.dh_hat(x)

        # outer objective
        outer_obj = Model.H(x) - Model.H(Model.x_opt)

        Inner_gaps.append(inner_gap)
        Inner_ress.append(inner_res)
        Outer_objs.append(outer_obj)

    return Inner_gaps, Inner_ress, Outer_objs, x


def Bilevel_Popov_primal_dual(psi_init, sig_init, s, Model, maxit, eps_f, opt_is_available=False):
    '''
    Algorithm in our paper.
    '''

    # storage
    Inner_gaps = []
    Inner_ress = []
    Outer_objs = []

    # initialize
    x_sig = np.copy(sig_init) # primal variable
    x_psi = np.copy(psi_init) # dual variable

    y_sig = np.copy(sig_init) # primal variable
    y_psi = np.copy(psi_init) # dual variable


    for k in range(maxit):

        x_sig_old = np.copy(x_sig)
        x_psi_old = np.copy(x_psi)

        eps_k = eps_f(k) # c / (k + sigma_e + 1) ** delta

        V_y_sig_out, V_y_psi_out = Model.V(y_sig, y_psi)
        Grad_y_sig_out, Grad_y_psi_out = Model.dh(y_sig, y_psi)
        in_prox_y_sig = x_sig - s * (V_y_sig_out + eps_k * Grad_y_sig_out)
        in_prox_y_psi = x_psi - s * (V_y_psi_out + eps_k * Grad_y_psi_out)

        y_sig, y_psi = Model.Prox(s, eps_k, in_prox_y_sig, in_prox_y_psi)

        V_y_sig_out, V_y_psi_out = Model.V(y_sig, y_psi)
        Grad_y_sig_out, Grad_y_psi_out = Model.dh(y_sig, y_psi)
        in_prox_x_sig = x_sig - s * (V_y_sig_out + eps_k * Grad_y_sig_out)
        in_prox_x_psi = x_psi - s * (V_y_psi_out + eps_k * Grad_y_psi_out)

        x_sig, x_psi = Model.Prox(s, eps_k, in_prox_x_sig, in_prox_x_psi)

        ## producing residuals
        V_x_sig_out, V_x_psi_out = Model.V(x_sig, x_psi)

        ## producing residuals
        # inner residual
        if Model.df_hat_available:
            xi_f_hat_sig, xi_f_hat_psi = Model.df_hat(x_sig, x_psi)

        elif Model.dh_hat_available:
            dh_hat_x_sig, dh_hat_x_psi = Model.dh_hat(x_sig, x_psi)
            xi_k_sig = (in_prox_y_sig - x_sig) / s
            xi_k_psi = (in_prox_x_psi - x_psi) / s

            xi_f_hat_sig = xi_k_sig - eps_k * dh_hat_x_sig
            xi_f_hat_psi = xi_k_psi - eps_k * dh_hat_x_psi

        inner_res = np.sum((V_x_sig_out + xi_f_hat_sig) ** 2) + np.sum((V_x_psi_out + xi_f_hat_psi) ** 2)
        Inner_ress.append(inner_res)

        if opt_is_available:

            # inner gap
            inner_gap = np.sum((x_sig - Model.sig_opt) * V_x_sig_out) \
                + np.sum((x_psi - Model.psi_opt) * V_x_psi_out) \
                + Model.f_hat(x_sig, x_psi) - Model.f_hat(Model.sig_opt, Model.psi_opt)
            Inner_gaps.append(inner_gap)

            # outer objective
            outer_obj = Model.H(x_sig, x_psi) - Model.H(Model.sig_opt, Model.psi_opt)
            Outer_objs.append(outer_obj)


    return Inner_gaps, Inner_ress, Outer_objs, x_sig, x_psi
