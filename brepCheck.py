# BREP-Python checker
# Author: Ohki Katakura (University of Hertfordshire)
# Contact: contact@neuronalpail.com

# This programme is to examine BREP-Python with comparing
# the resulting files to that of original BREP.
# It is hard to reproduce the same result due to rounding
# errors, but can generate with the random number table
# from Chicken Scheme.
# NOTE: random numbers in normal distribution are different
# in the different platforms due to their algorithms.

###########################################################
### Licence, GNU General Public License v3.0 (GPL3)     ###
###########################################################

# Copyright 2021 Ohki Katakura

# This programme is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This programme is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this programme. If not, see <https://www.gnu.org/licenses/>.

###########################################################
### Code block                                          ###
###########################################################

import numpy as np

prec = 6
eps = 2e-2
MPIsize = 6

print('brepCheck.py, precision = {0}, max error = {1}'.format(prec, eps))

for f in ['AA', 'PF']:
    for i in range(MPIsize):
        x = np.loadtxt('{0}coordinates{1}.dat'.format(f,i))
        x = np.round(x,6)
        x = x[np.lexsort([x[:,3], x[:,2], x[:,1]])]
        y = np.loadtxt('py{0}coordinates{1}.dat'.format(f,i))
        y = np.round(y,6)
        y = y[np.lexsort([y[:,3], y[:,2], y[:,1]])]
        print(f,i,((y-eps<=x)&(x<=y+eps)).all())

for f in ['GoCaxon', 'GoCbdend', 'GoCadend']:
    x = np.loadtxt('{0}coordinates.sorted.dat'.format(f))
    x = np.round(x,6)
    x = x[np.lexsort([x[:,3], x[:,2], x[:,1]])]
    y = np.loadtxt('{0}coordinates.sorted.dat'.format(f))
    y = np.round(y,6)
    y = y[np.lexsort([y[:,3], y[:,2], y[:,1]])]
    print(f,((y-eps<=x)&(x<=y+eps)).all())