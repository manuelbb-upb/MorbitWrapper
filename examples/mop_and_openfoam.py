#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 15:01:53 2020

@author: manuelbb
"""

import numpy as np 
from solve_of_fourier import createFourierControl, cost_and_drag_objectives
import MorbitWrapper as mw

#mw.set_MORBIT_SYS_IMG("/project_files/MorbitWrapper/Morbit.so") # path to an up-to-date sysimage for fast startup

# Simulation setup
T = 2
h = 0.015

# Setup Fourier coefficients
n_terms = 5;                # number of terms in truncated Fourier series
n_params = 2 * n_terms - 1;     # resulting number of decision parameters

lb_amplitude = -3.0;
ub_amplitude = 3.0;
lb_phase = 0.0;
ub_phase = 2*np.pi;

# Optimization input vector will have form [ A_0, φ_0, A_1, φ_1, … ]
# Construct lower and upper variable boundaries accordingly.
lb = np.array( [ [ lb_amplitude, lb_phase ] for i in range(n_terms) ] ).flatten()
lb = np.delete(lb,1)
ub = np.array( [ [ ub_amplitude, ub_phase ] for i in range(n_terms) ] ).flatten()
ub = np.delete(ub,1)

# Define a bi-value objective map taking a n_params-array of decision variables:
def vector_objective( param_vector ):
    param_vector_augmented = np.hstack( [ param_vector[0], 0.0, param_vector[1:] ])
    fourier_params = param_vector_augmented.reshape( (-1, 2) )
    u,iu = createFourierControl( h, T, fourier_params )
    return np.array( cost_and_drag_objectives( u,iu, T, h ) )

x_0 = lb + (ub - lb ) * np.random.rand( n_params )

mop = mw.MOP(lb = lb, ub = ub)

mop.add_expensive_function( vector_objective, 2 )

beconf = mw.AlgoConfig( max_evals = 10 )
x, y = mop.optimize(x_0, conf)
