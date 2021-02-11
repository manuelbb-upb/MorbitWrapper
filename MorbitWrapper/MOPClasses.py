#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 10:31:11 2020

@author: manuelbb
"""

import numpy as np

#from typing import Callable, Union, List, Any, NewType
#from typeguard import check_argument_types, check_type 

from .globals import julia_main

try:
    global plt
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    print("Matplotlib not found in current environment. Plotting will not work")
    global plt 
    plt = None 
    
class RbfConfig():
    jl_py_props = {
        "kernel" : "kernel",
        "shape_parameter" : "shape_parameter",
        "θ_enlarge_1" : "theta_enlarge_1",
        "θ_enlarge_2" : "theta_enlarge_2",
        "θ_pivot" : "theta_pivot",
        "θ_pivot_cholesky" : "theta_pivot_cholesky",
        "require_linear": "require_linear",
        "max_model_points" : "max_model_points",
        "use_max_points" : "use_max_points",
        "sampling_algorithm" : "sampling_algorithm",
        "constrained" : "constrained",
        "max_evals" : "max_evals",
    }
    
    py_jl_props = {v:k for (k,v) in jl_py_props.items() }
    
    def __init__(self, *args, **kwargs ):
        self.jl = julia_main()
        self.eval = self.jl.eval
                
        if len(args) == 1 and isinstance(args[0], dict):
            arg_dict = args[0]
        else:
            arg_dict = kwargs
            
        init_props = { RbfConfig.py_jl_props[k] : v for (k,v) in arg_dict.items() 
                      if k in RbfConfig.py_jl_props.keys() }
        self.jlObj = self.jl.RbfConfig(*[], **init_props )
        

# Set properties for RbfConfig class dynamically
for jl_prop, py_prop in RbfConfig.jl_py_props.items():
    
    def get_property( jl_prop, py_prop ):
        
        @property
        def new_property(self):
            return self.jl.getfield( self.jlObj, self.jl.Symbol( jl_prop ) )
        
        @new_property.setter 
        def new_property(self,val):
            self.jl.setfield_b( self.jlObj, self.jl.Symbol(jl_prop), val )
        
        return new_property
    setattr(RbfConfig, py_prop, get_property(jl_prop, py_prop))    


class MOP():
    def __init__(self, lb = [], ub = []):

        # 1) INITIALIZE COMPLETE JULIA ENVIRONMENT
        self.jl = julia_main() 
        self.eval = self.jl.eval
        
        # 2) CONSTRUCT MixedMOP OBJECT IN JULIA
        if np.size( lb ) != np.size( ub ): 
            raise "Lower bounds 'lb' and upper bounds 'ub' must have same size."
        else:
            self.lb = np.array(lb).flatten()
            self.ub = np.array(ub).flatten()
      
        self.jlObj = self.jl.MixedMOP(*[], **{"lb" : self.lb, "ub" : self.ub} )
        
        self.algo_config = None
    
    @property
    def n_objfs(self):
        return self.jlObj.n_objfs 
    
    # def set_function( self, func, grad = None ):
    #     index = self.n_objfs + 1
    #     exec( f"self.jl.func{index} = func;", {"self": self, "func": func, "index": index } )
    #     self.jl.eval(f'func{index}_handle = function(x) func{index}(x) end')
        
    #     if grad:
    #         exec( f"self.jl.grad{index} = grad;", {"self": self, "grad": grad, "index": index } )
    #         self.jl.eval(f'grad{index}_handle = function(x) grad{index}(x) end')
        
    #     return index
    
    # def add_expensive_function(self, func, n_out = 1):        
    #     index = self.set_function(func)        
    #     self.eval( f'add_objective!(mop, func{index}_handle, :expensive, {n_out})')
    #     return index    
    
    # def add_cheap_function( self, func, grad ):       
    #     index = self.set_function(func, grad )        
    #     self.eval( f'add_objective!(mop, func{index}_handle, grad{index}_handle)')
    #     return index
    
    # def add_batch_function(self, func, n_out = 1):
    #     index = self.set_function( func )
    #     self.eval( f'add_objective!(mop, func{index}_handle, :expensive, {n_out}, true)')
    #     return index
    
        

class AlgoConfig():
    """A wrapper class for the AlgoConfig structure provided by Morbit in Julia.
    
    Initiliaze by passing a dict of options or pass the arguments as keyword=value pairs.
    The created AlgoConfig instance becomes available as the .obj property"""
    
    
    jl_py_props = {
        "n_vars" : ("n_vars", "Int", "Number of decision variables (inferred)."),
        "n_objfs" : ("n_objfs", "Int", "Number of objective functions (inferred)."),
        "max_iter" : ("max_iter", "Int", "Maximum number of iterations."),
        "count_nonlinear_iterations" : ("count_conlinear_iterations", "Boolean", "Count iterations if model is not fully linear."),
        "max_evals" : ("max_evals", "Int", "Maximum number of objective evaluations (including AutoDiff and FiniteDiff evals)."),
        "descent_method" : ("descent_method", "Symbol", "Method for step calculation ('steepest', 'ps', 'cg', 'ds')"),
        "ideal_point" : ("ideal_point", "Vector{R} where R<:Real", "Utopia vector for image space subproblems."),
        "image_direction" : ("image_direction", "Vector{R} where R<:Real", "Alternative to utopia vector."),
        "θ_ideal_point" : ("theta_ideal_point", "Real", "Scaling factor for trust region radius in subproblem ideal point calculation."),
        "all_objectives_descent" : ("all_objectives_descent", "Boolean", "Boolean Flag: Need descent in **all** surrogates?"),
        "radius_update" : ("radius_update", "Symbol", "Radius update method ('standard' or 'steplength')"),
        "μ" : ("mu", "Real", "Largest criticality factor for criticality test."),
        "β" : ("beta", "Real", "Smaller criticality factor for criticality test."),
        "ε_crit" : ("eps_crit", "Real", "Threshold for criticality test."),
        "max_critical_loops" : ("max_critical_loops", "Int", "Maximum number of critical loops before break."),
        "x_stop_function" : ("x_stop_function", "F where F<:Function", "Function returning Bool to stop depending on iterate value."),
        "ν_success" : ("nu_success", "Real", "Large acceptance parameter."), # real  
        "ν_accept" : ("nu_accept", "Real", "Small acceptance parameter."),  # real
        "γ_crit" : ("gamma_crit", "Real", "Radius shrinking factor in criticality loop."),  # real
        "γ_grow" : ("gamma_grow", "Real", "Radius growing factor."), # real
        "γ_shrink" : ("gamma_shrink", "Real", "Radius shrinking factor for acceptable trial points."), # real
        "γ_shrink_much" : ("gamma_shrink_much", "Real", "Severe radius shrinking factor"),
        "Δ₀" : ("delta_init", "Real", "Initial trust region radius (relative to [0,1]^n if box constrained)."),  # real
        "Δ_max" : ("delta_max", "Real", "Maximum trust region radius"),
        "Δ_critical" : ("delta_critical", "Real", "Critical trust region radius - relative stopping test."),
        "Δ_min" : ("delta_min", "Real", "Minimum trust region radius."),
        "stepsize_min" : ("stepsize_min", "Real", "Relative stopping tolerance on iterates"),
        "use_eval_database" : ("use_eval_db", "Boolean", "Boolean Flag: Store intermediate results."),
        #"iter_data" : ("iter_data", Any, "Julia object storing iteration information."),
    }
    
    # TODO this is a bit redundant...
    py_jl_props = {v[0] : k for (k,v) in jl_py_props.items() }
    
    def __init__(self, *args, **kwargs):
        self.jl = julia_main()
        self.eval = self.jl.eval
                
        if len(args) == 1 and isinstance(args[0], dict):
            arg_dict = args[0]
        else:
            arg_dict = kwargs
            
        init_props = {AlgoConfig.py_jl_props[k] : v for (k,v) in arg_dict.items() 
                      if k in AlgoConfig.py_jl_props.keys() }
        self.jlObj = self.jl.AlgoConfig(*[], **init_props )
        
                
    def show(self):
        print( self.jlObj )
        
    def print_stop_info(self):
        print("\tStopping if either:")
        print(f"\t\t• Number of iterations reaches {self.max_iter}.")
        print(f"\t\t• Number of objective evaluations reaches {self.max_evals}.")
        print(f"\t\t• Trust region radius Δ becomes smaller than delta_min = {self.delta_min}.")
        print(f"\t\t• Trust region radius Δ becomes smaller than delta_critical = {self.delta_critical} AND")
        print(f"\t\t  stepsize is smaller than {self.stepsize_min}.")
        
    def print_fin_info(self):
        print(f"\tFinished after {self.n_iters} iterations and {self.n_evals} objective evaluations.")
    #     print(f"\tThere were {self.n_acceptable_steps} acceptable and {self.n_successful_steps} successful iterations.")
        
    # def unscale_sites(self, list_of_scaled_site_arrays ):
    #     return self.eval( "Morbit.unscale" )( self.jlObj.problem, list_of_scaled_site_arrays )
    
    # def scatter2_objectives(self, indices = [0,1]):
    #     if len(indices) == 0:
    #         return 
    #     else:
    #         f1 = self.values[ indices[0], : ]
    #         if len(indices) == 1:
    #             f2 = np.zeros( f1.shape )
    #         else:
    #             f2 = self.values[ indices[1], :]
            
    #     fig, ax = plt.subplots()
        
    #     ax.scatter( f1, f2)
    #     ax.plot( f1[self.iter_indices], f2[self.iter_indices], "kd-" )
        
    # @property 
    # def iter_indices(self):
    #     return self.jlObj.iter_data.iterate_indices - 1
    
    # @property 
    # def sites(self):
    #     pass    
    
    # @property 
    # def iter_sites(self):
    #     """List of numpy arrays corresponding to decision vectors that were centers of a trust region iteration."""
    #     return self.unscale_sites( self.jlObj.iter_data.sites_db[ self.iter_indices ] )
    
    # @property 
    # def values(self):
    #     """Matrix of evaluation results, each column is an objective vector."""
    #     return np.vstack( self.jlObj.iter_data.values_db ).transpose()

    # @property
    # def iter_values(self):
    #     return self.values[ self.iter_indices ]
    
    @property
    def n_iters(self):
        len_iter_array = len( self.jlObj.iter_data.iterate_indices )
        return 0 if len_iter_array == 0 else len_iter_array - 1
    
    @property
    def n_evals(self):
        return len( self.jlObj.iter_data.values_db )
    
    # @property 
    # def ρ_array(self):
    #     return self.jlObj.iter_data.ρ_array
    
    # @property 
    # def n_acceptable_steps(self):
    #     return np.sum( self.ρ_array >= self.ν_accept )
    
    # @property
    # def n_successful_steps(self):
    #     return np.sum( self.ρ_array >= self.ν_success )
    
    # # UNICODE PROPERTIES THAT ARE IMPOSSIBLE TO ENTER IN SPYDER
    # @property 
    # def Δ_0(self):
    #     return getattr(self.jlObj, "Δ₀")
    
    # def save(self, filename = None):
    #     self.eval("save_config")( self.jlObj, filename)
        
    # def load(self, filename):
    #     self.jlObj = self.eval("load_config")(filename)

# def get_property_function( propname ):
#     new_prop = property()
#     new_prop.getter( lambda self: getattr( self.jlObj, propname ) )

# # Add properties that are read directly from the wrapped AlgoConfig Julia instance
# for jl_prop, prop_tup in AlgoConfig.jl_py_props.items():
        
#     setattr( AlgoConfig, property_name, get_property_function(property_name) )
        
# Set properties for AlgoConfig class dynamically
for jl_prop, prop_tuple in AlgoConfig.jl_py_props.items():
    
    def get_property( jl_prop, prop_tuple ):
        prop_name, jl_type, prop_doc = prop_tuple
        
        @property
        def new_property(self):
            return self.jl.getfield( self.jlObj, self.jl.Symbol( jl_prop ) )
        
        @new_property.setter 
        def new_property(self,val):
            self.jl.setfield_b( self.jlObj, self.jl.Symbol(jl_prop), val )
        
        new_property.__doc__ = jl_type + ": " + prop_doc
        return new_property
    setattr(AlgoConfig, prop_tuple[0], get_property(jl_prop, prop_tuple))    


        
