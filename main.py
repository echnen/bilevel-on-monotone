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
"""Run all numerical experiments and save their outputs in the results directory."""

import pathlib

import experiments as expm

if __name__ == "__main__":

    pathlib.Path("results").mkdir(parents=True, exist_ok=True)

    print('Starting experiment Bilevel optimal assignment ...')
    expm.experiment_optimal_assignment_bilevel(N=5, maxit=10000)

    print('Starting experiment toy example ...')
    expm.experiment_toy_example(dim=6, maxit=5000, s=None, c=1, sigma_e=10)

    print('Testing the influence of delta ...')
    expm.experiment_trade_off(dim=6, maxit=10000, s=None, c=1, sigma_e=1.0)
