#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 09:44:53 2021

@author: manuelbb
"""

from .utilities import tprint
from .MOPClasses import MOP, AlgoConfig
from .globals import julia_main
import numpy as np 

def optimize( problem, x_0, cfg = None ):
    
    # if not isinstance( problem, MOP ):
    #     tprint("Invalid `problem`... returning.")
    #     return
    # if not cfg or not isinstance( cfg, AlgoConfig ):
    #     tprint("Using default optimization settings.")
    #     cfg = AlgoConfig()
    # if len(x_0) == 0:
    #     tprint("Need a non-empty starting array x_0.")
    #     return         
    tprint("Using default optimization settings.")
    cfg.print_stop_info() 
    
    x_0 = np.array(x_0).flatten() 
    jl = julia_main()
    X, FX = jl.optimize_b( cfg.jlObj, problem.jlObj, x_0 )
    tprint("Finished.")
    return X, FX