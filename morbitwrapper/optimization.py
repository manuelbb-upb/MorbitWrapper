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

def print_stop_info(config_jl, mop_jl = None):
    jl = julia_main()
    tprint( jl.eval("Morbit._stop_info_str")( config_jl, mop_jl ) )
      
def print_fin_info(mop):
    jl = julia_main()
    tprint( jl.eval("Morbit._fin_info_str")( mop.iterDataObj, mop.jlObj ) )

def _setup_logging_and_config( mop, x_0, cfg = None, debug_level = "Info"):
    # set debug level
    mop.eval(f"""
    import Logging;
    Logging.global_logger( Logging.ConsoleLogger(stderr, Logging.{debug_level}));
    """)
    
    # TODO: uncomment for production
    # if not isinstance( mop, MOP ):
    #     tprint("Invalid `mop`... returning.")
    #     return
    
    if not cfg:
        cfg = GenericConfig()
        tprint("Using default optimization settings.")
    if len(x_0) == 0:
        tprint("Need a non-empty starting array x_0.")
        raise Exception()
    
    print_stop_info(cfg.jlObj, mop.jlObj) 
    x_0 = np.array(x_0).flatten()
    
    return x_0, cfg 
    

def optimize( mop, x_0, cfg = None, debug_level = "Warn" ):
    x_0, cfg = _setup_logging_and_config( mop, x_0, cfg, debug_level )
    jl = julia_main()
    X, FX, iter_data_jl = jl.optimize( mop.jlObj , x_0, algo_config = cfg.jlObj)
    mop.iterDataObj = iter_data_jl
    print_fin_info(mop)
    return X, FX


def initialize_data( problem, x_0, cfg = None, debug_level = "Warn" ):
    "Setup Morbit and data for manual iteration."
    
    x_0, cfg = _setup_logging_and_config( problem, x_0, cfg, debug_level )
    jl = julia_main()
    
    jl.algo_config = cfg.jlObj
    jl.mop = problem.jlObj
    jl.x_0 = x_0 
    jl.eval("fx_0 = Float64[]")
    jl.populated_db = None
    mop_jl, iter_data_jl, sc_jl = jl.eval('Morbit.initialize_data( mop, x_0, fx_0; algo_config, populated_db )')
    
    MAX_ITER = jl.eval("Morbit.max_iter(algo_config)")
    
    problem.iterDataObj = iter_data_jl
    return mop_jl, iter_data_jl, sc_jl, jl.algo_config, MAX_ITER

def iterate( mop_jl, iter_data_jl, sc_jl, algo_config_jl ):
    jl = julia_main()
    iter_abbort = jl.eval('Morbit.iterate!')(iter_data_jl, mop_jl, sc_jl, algo_config_jl )
    return iter_abbort
    
def get_ret_values( iter_data_jl, mop_jl ):
    jl = julia_main()
    X,FX = jl.eval('Morbit.get_return_values')(iter_data_jl, mop_jl)
    jl.eval('Morbit.finalize_iter_data!')(iter_data_jl, mop_jl)
    return X, FX
    