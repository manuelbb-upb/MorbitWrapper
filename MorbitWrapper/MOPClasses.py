#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 10:31:11 2020

@author: manuelbb
"""

import numpy as np

#from typing import Callable, Union, List, Any, NewType
#from typeguard import check_argument_types, check_type 

from inspect import isfunction
from .globals import julia_main
from uuid import uuid4
try:
    global plt
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    print("Matplotlib not found in current environment. Plotting will not work")
    global plt 
    plt = None
    
def merge_kwargs(*args, **kwargs):
    if len(args) == 1 and isinstance(args[0], dict):
        arg_dict0 = args[0]
    else:
        arg_dict0 = {}
    return { **arg_dict0, **kwargs }

def get_kwargs(allowed_properties, *args, **kwargs ):
    arg_dict = merge_kwargs(*args, **kwargs)
    allowed_keys = allowed_properties.keys()
    return { allowed_properties[k] : v for (k,v) in arg_dict.items() if k in allowed_keys }

def batch_eval(*args, **kwargs):
    arg_dict = merge_kwargs(*args, **kwargs)
    return False if "batch_eval" not in arg_dict.keys() else arg_dict["batch_eval"]    

def wrap_func( func ):
    jl = julia_main()
    # TODO this is a very hacky workaround...  
    # either make Morbit accepy PyCall.PyObjects or 
    # find a way to deepcopy a PyObject in the first argument of `pycall`
    tmp_name = "tmp_" + str(uuid4()).replace("-","_")
    exec( f"jl.{tmp_name} = func")
    jl_func = jl.eval(f"""
    function(x :: Union{{Real, Vector{{R}} where R<:Real}})
        return PC.pycall(
            {tmp_name},
            PC.PyAny,
            x
        )
    end
    """)
    return jl_func

def wrap_funcs_in_dict( arg_dict ):
    for k,v in arg_dict.items():
        if isfunction(v):
            arg_dict[k] = wrap_func( v )
        elif (isinstance( v, list ) or isinstance(v, tuple ) or 
              isinstance(v, np.ndarray)) and isfunction(v[0]):
            arg_dict[k] = []
            for vi in v:
                arg_dict[k].append( wrap_func( vi ) )
            
    return arg_dict
    
# Define Configuration Class wrappers. They all have the same structure...
# TODO Can we do this in a more dynamical/meta-y way? Do we want to?
    
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
                
        init_props = get_kwargs( RbfConfig.py_jl_props, *args, **kwargs )
        self.jlObj = self.jl.RbfConfig(*[], **init_props )
        
class LagrangeConfig():
    jl_py_props = {
        "degree" : "degree",
        "θ_enlarge" : "theta_enlarge",
        "ε_accept" : "eps_accept",
        "Λ" : "lambda",
        "allow_not_linear" : "allow_not_linear",
        "optimized_sampling" : "optimized_sampling",
        "use_saved_sites" : "use_saved_sites",
        "io_lock" : "threading_lock",
        "max_evals" : "max_evals"
    }
    py_jl_props = {v:k for (k,v) in jl_py_props.items() }
    
    def __init__(self, *args, **kwargs ):
        self.jl = julia_main()
        self.eval = self.jl.eval
                
        init_props = get_kwargs( LagrangeConfig.py_jl_props, *args, **kwargs )
        self.jlObj = self.jl.LagrangeConfig(*[], **init_props )
        
    
class TaylorConfig():
    jl_py_props = {
        "degree" : "degree",
        "gradients" : "gradients",
        "hessians" : "hessians",
        "jacobian" : "jacobian",
        "max_evals" : "max_evals"
    }
    py_jl_props = { 
        **{v:k for (k,v) in jl_py_props.items() },
        "grad" : "gradients",
        "gradient" : "gradients",
        "hessian" : "hessians",
    }                   
    
    def __init__(self, *args, **kwargs ):
        self.jl = julia_main()
        self.eval = self.jl.eval
                
        init_props = get_kwargs( TaylorConfig.py_jl_props, *args, **kwargs )
        self.jlObj = self.jl.TaylorConfig(*[], **init_props )
        
 
class ExactConfig():
    jl_py_props = {
        "gradients" : "gradients",
        "jacobian" : "jacobian",
        "max_evals" : "max_evals"
    }
    py_jl_props = { 
        **{v:k for (k,v) in jl_py_props.items() },
        "grad" : "gradients",
        "gradient" : "gradients",
        "hessian" : "hessians",
    } 
    
    def __init__(self, *args, **kwargs ):
        self.jl = julia_main()
        self.eval = self.jl.eval
                
        init_props = get_kwargs( ExactConfig.py_jl_props, *args, **kwargs )
        self.jlObj = self.jl.ExactConfig(*[], **init_props )

# Set properties for Config classes dynamically
for ConfClass in [ExactConfig, RbfConfig, LagrangeConfig, TaylorConfig]:
    for py_prop, jl_prop in ConfClass.py_jl_props.items():
        
        def get_property( jl_prop, py_prop ):
            
            @property
            def new_property(self):
                return self.jl.getfield( self.jlObj, self.jl.Symbol( jl_prop ) )
            
            @new_property.setter 
            def new_property(self,val):
                self.jl.setfield_b( self.jlObj, self.jl.Symbol(jl_prop), val )
            
            return new_property
        setattr(ConfClass, py_prop, get_property(jl_prop, py_prop))    

# Wrapper for 'MixedMOP' Julia class. Defines a problem.
class MOP():
    def __init__(self, lb = [], ub = []):

        # 1) INITIALIZE COMPLETE JULIA ENVIRONMENT
        self.jl = julia_main() 
        self.eval = self.jl.eval
        
        # 2) CONSTRUCT MixedMOP OBJECT IN JULIA
        if np.size( lb ) != np.size( ub ): 
            print("Lower bounds 'lb' and upper bounds 'ub' must have same size.")
            return
        else:
            self.lb = np.array(lb).flatten()
            self.ub = np.array(ub).flatten()
      
        self.jlObj = self.jl.MixedMOP(*[], **{"lb" : self.lb, "ub" : self.ub} )
            
    @property
    def n_objfs(self):
        return self.jlObj.n_objfs 
    
    def _prepare_args(self, allowed_properties, *args, **kwargs): 
        arg_dict0 = get_kwargs(allowed_properties, *args, **kwargs)
        arg_dict = wrap_funcs_in_dict(arg_dict0)
        can_batch = batch_eval(*args, **kwargs) 
        return arg_dict, can_batch
        
    def _add_objective(self, func, cfg, can_batch = False ):
        self.jl.add_objective_b( self.jlObj, wrap_func( func ), cfg.jlObj, batch_eval = can_batch )
        return self.n_objfs
        
    def add_cheap_objective(self, func, *args, **kwargs):
        arg_dict, batch_eval = self._prepare_args(ExactConfig.py_jl_props, *args, **kwargs)
        cfg = ExactConfig(*[], **arg_dict) 
        return self._add_objective(func, cfg, batch_eval)
    
    def add_lagrange_objective(self, func, *args, **kwargs):
        arg_dict, batch_eval = self._prepare_args(LagrangeConfig.py_jl_props, *args, **kwargs)
        cfg = LagrangeConfig(*[], **arg_dict) 
        return self._add_objective(func, cfg, batch_eval)
    
    def add_taylor_objective(self, func, *args, **kwargs):
        arg_dict, batch_eval = self._prepare_args(TaylorConfig.py_jl_props, *args, **kwargs)
        cfg = TaylorConfig(*[], **arg_dict) 
        return self._add_objective(func, cfg, batch_eval)
    
    def add_rbf_objective(self, func, *args, **kwargs):
        arg_dict, batch_eval = self._prepare_args(RbfConfig.py_jl_props, *args, **kwargs)
        cfg = RbfConfig(*[], **arg_dict) 
        return self._add_objective(func, cfg, batch_eval)
    
    def add_cheap_vec_objective(self, func, *args, batch_eval = False, n_out = -1, **kwargs ):
        pass

    # TODO vector (and batch) objectives
            

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
        print(f"\tThere were {self.n_acceptable_steps} acceptable and {self.n_successful_steps} successful iterations.")
     
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
        
    @property 
    def iter_data(self):
        return self.jlObj.iter_data 
    
    @property 
    def iter_indices(self):
        return self.iter_data.iterate_indices - 1
    
    @property 
    def eval_sites(self):
        return self.iter_data.sites_db 
    
    @property 
    def eval_vectors(self): 
        return self.iter_data.values_db 
    
    @property
    def iter_sites(self):
        return self.eval_sites[ self.iter_indices ]
    
    @property
    def iter_vectors(self):
        return self.eval_vectors[ self.iter_indices ]
        
    @property
    def n_iters(self):
        len_iter_array = len( self.jlObj.iter_data.iterate_indices )
        return 0 if len_iter_array == 0 else len_iter_array - 1
    
    @property
    def n_evals(self):
        return len( self.jlObj.iter_data.values_db )
    
    @property 
    def ρ_array(self):
        return self.jlObj.iter_data.ρ_array
    
    @property 
    def n_acceptable_steps(self):
        return np.sum( self.ρ_array >= self.ν_accept )
    
    @property
    def n_successful_steps(self):
        return np.sum( self.ρ_array >= self.ν_success )
    
    # def save(self, filename = None):
    #     self.eval("save_config")( self.jlObj, filename)
        
    # def load(self, filename):
    #     self.jlObj = self.eval("load_config")(filename)

        
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


        
