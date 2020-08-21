#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 10:31:11 2020

@author: manuelbb
"""

import numpy as np

from .utilities import tprint, clean_args
from .runtime_setup import julia_main

import matplotlib.pyplot as plt

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
        
        self.algo_config = None
    
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
    
    def add_batch_function(self, func, n_out = 1):
        index = self.set_function( func )
        self.eval( f'add_objective!(mop, func{index}_handle, :expensive, {n_out}, true)')
        return index
    
    def optimize( self, x_0 = [], config_obj = None ):
        if not config_obj or not isinstance( config_obj, AlgoConfig ):
            tprint("Using default optimization settings.")
            config_obj = AlgoConfig()
        if len(x_0) == 0:
            raise "Need a non-empty starting array x_0."
        self.algo_config = config_obj
        
        tprint("Starting optimization.")
        self.algo_config.print_stop_info()
        
        x_0 = np.array(x_0).flatten()
        self.eval("optimize!")( config_obj.obj, self.obj, x_0 )
        
        if config_obj.obj.iter_data:
            #X = self.eval("Morbit.unscale")( self.obj, self.algo_config.obj.iter_data.x )
            X = self.algo_config.obj.iter_data.x # already unscaled
            FX = self.algo_config.obj.iter_data.f_x
            self.algo_config.print_fin_info()
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
        
    def show(self):
        print( self.obj )
        
    def print_stop_info(self):
        print("\tStopping if either:")
        print(f"\t\t• Number of iterations reaches {self.max_iter}.")
        print(f"\t\t• Number of objective evaluations reaches {self.max_evals}.")
        print(f"\t\t• Trust region radius Δ becomes smaller than Δ_min = {self.Δ_min}.")
        print(f"\t\t• Trust region radius Δ becomes smaller than Δ_critical = {self.Δ_critical} AND")
        print(f"\t\t  stepsize is smaller than {self.stepsize_min}.")
        
    def print_fin_info(self):
        print(f"\tFinished after {self.n_iters} iterations and {self.n_evals} objective evaluations.")
        print(f"\tThere were {self.n_acceptable_steps} acceptable and {self.n_successful_steps} successful iterations.")
        
    def unscale_sites(self, list_of_scaled_site_arrays ):
        return self.eval( "Morbit.unscale" )( self.obj.problem, list_of_scaled_site_arrays )
    
    def scatter2_objectives(self, indices = [0,1]):
        if len(indices) == 0:
            return 
        else:
            f1 = self.values[ indices[0], : ]
            if len(indices) == 1:
                f2 = np.zeros( f1.shape )
            else:
                f2 = self.values[ indices[1], :]
            
        fig, ax = plt.subplots()
        
        ax.scatter( f1, f2)
        ax.plot( f1[self.iter_indices], f2[self.iter_indices], "kd-" )
        
    @property 
    def iter_indices(self):
        return self.obj.iter_data.iterate_indices - 1
    
    @property 
    def sites(self):
        pass    
    
    @property 
    def iter_sites(self):
        """List of numpy arrays corresponding to decision vectors that were centers of a trust region iteration."""
        return self.unscale_sites( self.obj.iter_data.sites_db[ self.iter_indices ] )
    
    @property 
    def values(self):
        """Matrix of evaluation results, each column is an objective vector."""
        return np.vstack( self.obj.iter_data.values_db ).transpose()

    @property
    def iter_values(self):
        return self.values[ self.iter_indices ]
    
    @property
    def n_iters(self):
        len_iter_array = len( self.obj.iter_data.iterate_indices )
        return 0 if len_iter_array == 0 else len_iter_array - 1
    
    @property
    def n_evals(self):
        return len( self.obj.iter_data.values_db )
    
    @property 
    def ρ_array(self):
        return self.obj.iter_data.ρ_array
    
    @property 
    def n_acceptable_steps(self):
        return np.sum( self.ρ_array >= self.ν_accept )
    
    @property
    def n_successful_steps(self):
        return np.sum( self.ρ_array >= self.ν_success )
    
    # UNICODE PROPERTIES THAT ARE IMPOSSIBLE TO ENTER IN SPYDER
    @property 
    def Δ_0(self):
        return getattr(self.obj, "Δ₀")
    
    def save(self, filename = None):
        self.eval("save")( self.opj, filename)
        
    def load(self, filename):
        self.obj = self.eval("load")(filename)

def get_property_function( propname ):
    return property( lambda self: getattr( self.obj, propname ) )

# Add properties that are read directly from the wrapped AlgoConfig Julia instance
for property_name in ["max_iter", "max_evals", "n_vars", "n_exp", 
                      "n_cheap", "max_model_points", "max_critical_loops","stepsize_min",
                      "Δ_critical", "Δ_max", "Δ_min", "ε_crit", "ν_success", "ν_accept", 
                      "γ_crit", "γ_grow", "γ_shrink", "γ_shrink_much"]:

    setattr( AlgoConfig, property_name, get_property_function(property_name) )
        
        
        

        
