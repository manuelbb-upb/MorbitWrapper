#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 16:22:00 2020

@author: manuelbb
"""
#import os
#os.chdir( os.path.dirname( __file__ ) )
#%%
import os
os.chdir( os.path.join( os.path.dirname(__file__), ".." ) )
import MorbitWrapper as mw
#%%
#mw.set_MORBIT_SYS_IMG("/project_files/MorbitWrapper/Morbit.so") # path to an up-to-date sysimage for fast startup

mop = mw.MOP(lb = [-4, -4], ub =[4,4])

def f1(x):
    return (x[0]-1)**2 + (x[1]-1)**2
    
f2 = lambda x : (x[0]+1)**2 + (x[1]+1)**2 
df2 = lambda x : [ 2*(x[0]+1), 2*(x[1]+1) ]

mop.add_lagrange_objective(f1, degree=2)
#mop.add_rbf_objective(f2)
mop.add_cheap_objective(f2, grad = df2)

conf = mw.AlgoConfig(
    max_iter = 10, 
    all_objectives_descent = True 
)

#x, y = mw.optimize( mop, [3.0, 4.0], conf )
