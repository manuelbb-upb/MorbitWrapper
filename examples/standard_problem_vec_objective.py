#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 10:02:11 2020

@author: manuelbb
"""
import numpy as np
import MorbitWrapper as mw

#mw.set_MORBIT_SYS_IMG(None)

mop = mw.MOP(lb = [-4, -4], ub =[4,4])

def f(x):
    return np.array( [ [(x[0]-1)**2 + (x[1]-1)**2],
                        [(x[0]+1)**2 + (x[1]+1)**2 ] ], dtype=float).flatten()
    
mop.add_expensive_function(f, 2)

conf = mw.AlgoConfig( max_iter = 20 )
x, y = mop.optimize([3.14, 4], conf)
