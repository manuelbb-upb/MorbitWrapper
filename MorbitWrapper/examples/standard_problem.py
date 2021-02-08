#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 16:22:00 2020

@author: manuelbb
"""
import MorbitWrapper as mw

#mw.set_MORBIT_SYS_IMG("/project_files/MorbitWrapper/Morbit.so") # path to an up-to-date sysimage for fast startup

mop = mw.MOP(lb = [-4, -4], ub =[4,4])

def f1(x):
    return (x[0]-1)**2 + (x[1]-1)**2
    
f2 = lambda x : (x[0]+1)**2 + (x[1]+1)**2 
df2 = lambda x : [ 2*(x[0]+1), 2*(x[1]+1) ]

mop.add_expensive_function(f1)
mop.add_cheap_function(f2, df2)

conf = mw.AlgoConfig(
    sampling_algorithm = "monte_carlo", 
    max_iter = 20, 
    scale_values = False, 
    all_objectives_descent = True 
)
x, y = mop.optimize([3.14, 4], conf)
