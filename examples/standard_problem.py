#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 16:22:00 2020

@author: manuelbb
"""
#%%
import os
import sys 
sys.path.append( os.path.join( os.path.dirname(__file__), ".." ) )
import morbitwrapper as mw
#%%

# mw.set_MORBIT_SYS_IMG(os.path.join( os.path.dirname(__file__), "pycall_morbit.sysimg") )
mw.set_JULIA_ENV("@v1.6")   # I have Morbit dev'ed in the default environment

mop = mw.MOP([-5, -5], [5,5] )

def f1(x):
    return (x[0]-1)**2 + (x[1]-1)**2
    
f2 = lambda x : (x[0]+1)**2 + (x[1]+1)**2 
df2 = lambda x : [ 2*(x[0]+1), 2*(x[1]+1) ]

# You can add as single objectives that are modelled:
# mop.add_lagrange_objective(f1, degree=2)
# mop.add_rbf_objective(f1)
# mop.add_cheap_objective(f2, grad = df2)

# Or use a vector valued objective.
F = lambda x : [f1(x), f2(x)]
mop.add_rbf_vec_objective(F, n_out = 2 )

conf = mw.AlgoConfig(
    max_iter = 50, 
    strict_backtracking = True,
    strict_acceptance_test = True,
    x_tol_rel = 1e-2,
)

#%%

# Either optimize in one go:
x0 = [3.0, 4.0]
# x, y = mw.optimize( mop, x0, conf )

# Or manually iterate:
# First, initialize the julia objects
mop_jl, iter_data_jl, sc_jl, algo_config_jl, MAX_ITER = mw.initialize_data( mop, x0, conf )

# Then call `iterate`. The julia objects are modified in place.
for i in range(MAX_ITER):
    abbort = mw.iterate( mop_jl, iter_data_jl, sc_jl, algo_config_jl )
    if abbort: 
        break;
mw.print_fin_info(mop)

# Retrieve final values from `iter_data_jl`. 
# `mop_jl` is needed to untransform and resort the internal vectors.
x, y = mw.get_ret_values( iter_data_jl, mop_jl )

mop.plot_objectives()
mop.plot_iterates()
