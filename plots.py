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

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import rc


rc('font', **{'family': 'serif', 'serif': ['Times'], 'size': 15})
rc('text', usetex=True)


def plot_ot_comparison(delta,
                       Inner_gaps_BiEG, Inner_ress_BiEG, Outer_objs_BiEG,
                       Inner_gaps_OEGH, Inner_ress_OEGH, Outer_objs_OEGH,
                       maxit=None):

    metrics_BiEG = [np.asarray(Inner_gaps_BiEG), np.asarray(Inner_ress_BiEG), np.asarray(Outer_objs_BiEG)]
    metrics_OEGH = [np.asarray(Inner_gaps_OEGH), np.asarray(Inner_ress_OEGH), np.asarray(Outer_objs_OEGH)]
    titles = ["Inner gap", "Inner residual", "Outer objective gap"]

    iterations = np.arange(1, maxit + 1)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))

    for j in range(3):
        bieg = metrics_BiEG[j]
        oegh = metrics_OEGH[j]

        ax[j].loglog(iterations, bieg, linewidth=2, color="blue", label="BiEG")
        ax[j].loglog(iterations, oegh, linewidth=2, color="red", label="OEG-H")

        ax[j].set_title(titles[j])
        ax[j].set_xlabel("Iteration")
        ax[j].grid(True, which="both", alpha=0.25)
        ax[j].set_xlim(1, maxit)

    ax[0].legend()
    plt.savefig('results/ot_comparison.pdf', bbox_inches='tight')
    plt.show()


def plot_ot_maps(mu, nu, gammas, threshold=1e-8):

    titles = ["An Optimal Plan", "Reconstructed Bilevel (BiEG)", "Reconstructed Bilevel (OEG-H)"]
    a, X = mu
    b, Y = nu
    a, X = np.asarray(a, float), np.asarray(X, float)
    b, Y = np.asarray(b, float), np.asarray(Y, float)

    num_plots = len(gammas)
    fig, ax = plt.subplots(1, num_plots, figsize=(5 * num_plots, 4.5), sharex=True, sharey=True)
    ax = np.atleast_1d(ax)

    s_mu = 10 + 60 * a / a.max()
    s_nu = 10 + 60 * b / b.max()

    for axis, gamma, title in zip(ax, gammas, titles):
        gamma = np.asarray(gamma, float)
        if gamma.ndim == 1:
            gamma = gamma.reshape(len(a), len(b))
        gamma = np.maximum(gamma, 0.0)
        gmax = gamma.max()

        if gmax > 0:
            for i in range(len(a)):
                for j in range(len(b)):
                    if gamma[i, j] > threshold:
                        lw = 0.5 + 10 * gamma[i, j]
                        axis.plot([X[i, 0], Y[j, 0]], [X[i, 1], Y[j, 1]],
                                  "k-", lw=lw, alpha=0.8, zorder=1)

        axis.scatter(X[:, 0], X[:, 1], s=s_mu, c="blue", marker="o",
                     linewidths=1.2, label=r"$\mu$", zorder=3)
        axis.scatter(Y[:, 0], Y[:, 1], s=s_nu, c="red", marker="o",
                     linewidths=1.2, label=r"$\nu$", zorder=3)

        axis.set_title(title)
        axis.set_xlabel(r"$x_1$")
        axis.set_aspect("equal")
        axis.grid(alpha=0.25)

    ax[0].set_ylabel(r"$x_2$")
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.15), ncol=2)
    fig.subplots_adjust(wspace=0.03, top=0.84, bottom=0.12)
    plt.savefig('results/ot_plans.pdf', bbox_inches='tight')
    plt.show()


def plot_toy_example(Metrics_EG, Metrics_Popov, maxit, Deltas):

    iterations = np.arange(1, maxit + 1)
    titles = ["Inner gap", "Inner residual", "Outer objective gap"]
    colors = ["blue", "red"]
    eps = 1e-16

    EG = np.maximum(np.abs(np.asarray(Metrics_EG)), eps)
    Popov = np.maximum(np.abs(np.asarray(Metrics_Popov)), eps)

    # Shape: (num_deltas, num_metrics, maxit)
    mean_EG = np.exp(np.mean(np.log(EG), axis=0))
    mean_Popov = np.exp(np.mean(np.log(Popov), axis=0))

    fig, ax = plt.subplots(len(Deltas), 3, figsize=(15, 12),
                           sharex=True, squeeze=False)

    for r, delta in enumerate(Deltas):
        for j in range(3):
            for i in range(EG.shape[0]):
                ax[r, j].loglog(iterations, EG[i, r, j], color=colors[0],
                                alpha=0.15, linewidth=0.8)
                ax[r, j].loglog(iterations, Popov[i, r, j], color=colors[1],
                                alpha=0.15, linewidth=0.8)

            line_eg, = ax[r, j].loglog(iterations, mean_EG[r, j], color=colors[0],
                           linewidth=2.5)
            line_popov, = ax[r, j].loglog(iterations, mean_Popov[r, j], color=colors[1],
                                          linewidth=2.5)

            if j == 0:
                rate = 1 / (iterations + 1) ** (2 * delta)
                rate_label = rf"$\mathcal{{O}}(k^{{-{2 * delta:g}}})$"
            elif j == 1:
                rate = 5 / (iterations + 1) ** delta # - 1 / (maxit + 1) ** delta + mean_EG[r, j][-1] + 1e-4
                rate_label = rf"$\mathcal{{O}}(k^{{-{delta:g}}})$"
            else:
                exponent = 0.5 - delta
                rate = 1 / (iterations + 1) ** exponent # - 1 / (maxit + 1) ** exponent + mean_EG[r, j][-1] + 1e-4
                rate_label = rf"$\mathcal{{O}}(k^{{-{exponent:g}}})$"

            ax[r, j].loglog(iterations, rate, "--", color="k", label=rate_label)
            ax[r, j].set_xlim(1, maxit)
            ax[r, j].grid(True, which="both", alpha=0.3)

            if r == 0:
                ax[r, j].set_title(titles[j])
            if j == 0:
                ax[r, j].set_ylabel(rf"$\delta={delta}$")
            if r == len(Deltas) - 1:
                ax[r, j].set_xlabel("Iteration")

            ax[r, j].legend()

    fig.legend([line_eg, line_popov], ["Bi-EG", "OEG-H"],
           loc="lower center", ncol=2, frameon=False,
           bbox_to_anchor=(0.5, -0.0001))

    plt.tight_layout(rect=[0, 0.05, 1, 1], w_pad=0.5, h_pad=0.5)
    plt.savefig('results/experiment_toy_example.pdf', bbox_inches='tight')
    plt.show()



def plot_delta_experiments(Metrics_EG, Deltas, maxit):

    Metrics_EG = np.asarray(Metrics_EG)
    Deltas = np.asarray(Deltas)
    iterations = np.arange(1, maxit + 1)
    titles = ["Inner gap", "Inner residual", "Outer objective gap"]
    eps = 1e-16

    cmap = LinearSegmentedColormap.from_list(
        "BlueWhite",
        [(0.0, 0.0, 0.35), (0.0, 0.3, 1.0), (0.75, 0.9, 1.0)]
    )
    norm = Normalize(Deltas.min(), Deltas.max())
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    fig, ax = plt.subplots(1, 4, figsize=(15, 5), gridspec_kw={"width_ratios": [1, 1, 1, 0.05]})
    cax = ax[-1]
    ax = ax[:3]

    for j in range(3):
        for i, delta in enumerate(Deltas):

            values = Metrics_EG[i, j]

            if j == 0:
                ax[j].set_ylim(1e-6, 1e0)

            if j == 2:
                ax[j].loglog(iterations, np.abs(values), color=cmap(norm(delta)), linewidth=1.3)
                negative = values < 0
                ax[j].fill_between(iterations, ax[j].get_ylim()[0], ax[j].get_ylim()[1],
                                   where=negative, color="yellow", alpha=0.01, step="mid",
                                   transform=ax[j].get_xaxis_transform())
            else:
                ax[j].loglog(iterations, values, color=cmap(norm(delta)), linewidth=1.3)


        ax[j].set_title(titles[j])
        ax[j].set_xlabel("Iteration")
        ax[j].set_xlim(1, maxit)
        ax[j].grid(True, which="both", alpha=0.3)

    fig.colorbar(sm, cax=cax, orientation="vertical", label=r"$\delta$")
    plt.tight_layout(w_pad=0.5)
    plt.savefig('results/testing_dependence_on_delta.pdf', bbox_inches='tight')
    plt.show()
