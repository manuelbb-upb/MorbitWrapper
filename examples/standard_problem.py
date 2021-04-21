#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 16:22:00 2020

@author: manuelbb
"""
#%%
import os
os.chdir( os.path.join( os.path.dirname(__file__), ".." ) )
import morbitwrapper as mw
#%%

mw.set_MORBIT_SYS_IMG("/home/manuelbb/Desktop/Morbit.sysimg") # path to an up-to-date sysimage for fast startup
mw.set_JULIA_RUNTIME("/opt/julia-1.6.0/bin/julia")
#mw.set_JULIA_ENV("/home/manuelbb/.julia/dev/Morbit")

mop = mw.MOP(lb = [-5, -5], ub =[5,5])

def f1(x):
    return (x[0]-1)**2 + (x[1]-1)**2
    
f2 = lambda x : (x[0]+1)**2 + (x[1]+1)**2 
df2 = lambda x : [ 2*(x[0]+1), 2*(x[1]+1) ]

#mop.add_lagrange_objective(f1, degree=2)
#mop.add_rbf_objective(f1)
#mop.add_cheap_objective(f2, grad = df2)

F = lambda x : [f1(x), f2(x)]

mop.add_rbf_vec_objective(F, n_out = 2)

conf = mw.AlgoConfig(
    max_iter = 20, 
    strict_backtracking = True,
    strict_acceptance_test = True,
)

#%%
x, y = mw.optimize( mop, [3.0, 4.0], conf )
