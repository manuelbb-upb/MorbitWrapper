#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 10:31:11 2020

@author: manuelbb
"""

import os
import numpy as np

from .utilities import tprint, clean_args
from .runtime_setup import julia_main
from .OptOptionsClass import OptOptions

class MOP():
    def __init__(self, lb = [], ub = [], array_of_expensive_funcs = [], 
                 array_of_cheap_funcs = [], array_of_cheap_derivatives = []):

        # 1) INITIALIZE COMPLETE JULIA ENVIRONMENT
        self.jl = julia_main() 
        self.eval = self.jl.eval
        
        # 2) CONSTRUCT MixedMOP OBJECT IN JULIA
        if np.size( lb ) != np.size( ub ): 
            raise "Lower bounds 'lb' and upper bounds 'ub' must have same size."
        else:
            self.lb = np.array(lb).flatten()
            self.ub = np.array(ub).flatten()
        
        self.jl.lb = lb
        self.jl.ub = ub 
        self.obj = self.eval("mop = MixedMOP( lb = lb, ub = ub)")
    
    @property
    def n_objfs(self):
        return self.obj.n_exp + self.obj.n_cheap
    
    def set_function( self, func, grad = None ):
        index = self.n_objfs + 1
        exec( f"self.jl.func{index} = func;", {"self": self, "func": func, "index": index } )
        self.jl.eval(f'func{index}_handle = function(x) func{index}(x) end')
        
        if grad:
            exec( f"self.jl.grad{index} = grad;", {"self": self, "grad": grad, "index": index } )
            self.jl.eval(f'grad{index}_handle = function(x) grad{index}(x) end')
        
        return index
    
    def add_expensive_function(self, func, n_out = 1):        
        index = self.set_function(func)        
        self.eval( f'add_objective!(mop, func{index}_handle, :expensive, {n_out})')
        return index
        
    def add_cheap_function( self, func, grad ):       
        index = self.set_function(func, grad )        
        self.eval( f'add_objective!(mop, func{index}_handle, grad{index}_handle)')
        return index
    
    def optimize( self, x_0 = [], config_obj = None ):
        if not config_obj or not isinstance( config_obj, AlgoConfig ):
            tprint("Using default optimization settings.")
            config_obj = AlgoConfig()
        if len(x_0) == 0:
            raise "Need a non-empty starting array-like."
        
        x_0 = np.array(x_0).flatten()
        self.eval("optimize!")( config_obj.obj, self.obj, x_0 )
        
        if config_obj.obj.iter_data:
            X = config_obj.obj.iter_data.x
            FX = config_obj.obj.iter_data.f_x
            return X,FX
        else:
            return 
        
class AlgoConfig():
    """A wrapper class for the AlgoConfig structure provided by Morbit in Julia.
    
    Initiliaze by passing a dict of options or pass the arguments as keyword=value pairs.
    The created AlgoConfig instance becomes available as the .obj property"""
    
    def __init__(self, *args, **kwargs):
        self.jl = julia_main()
        self.eval = self.jl.eval
        
        if len(args) == 1 and isinstance(args[0], dict):
            arg_dict = args[0]
        else:
            arg_dict = kwargs
        
        arg_dict = clean_args(arg_dict)
        self.obj = self.eval('AlgoConfig')( **arg_dict )
        
        
        

        
