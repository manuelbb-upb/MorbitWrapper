#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 09:44:53 2021

@author: manuelbb
"""

from .utilities import tprint
from .MOPClasses import MOP, GenericConfig
from .globals import julia_main
import numpy as np 

def optimize( problem, x_0, cfg = None, debug_level = "Warn" ):
    # set debug level
    jl = problem.eval(f"""
    import Logging;
    Logging.global_logger( Logging.ConsoleLogger(stderr, Logging.{debug_level}));
    """)
    
    # TODO: uncomment for production
    # if not isinstance( problem, MOP ):
    #     tprint("Invalid `problem`... returning.")
    #     return
    
    if not cfg:
        cfg = GenericConfig()
        tprint("Using default optimization settings.")
    if len(x_0) == 0:
        tprint("Need a non-empty starting array x_0.")
        return         
    
    cfg.print_stop_info() 
    
    x_0 = np.array(x_0).flatten() 
    jl = julia_main()

    X, FX, id = jl.optimize( problem.jlObj , x_0, algo_config = cfg.jlObj)
    tprint("Finished.")
    return X, FX