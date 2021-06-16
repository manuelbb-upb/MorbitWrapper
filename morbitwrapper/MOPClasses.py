#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 10:31:11 2020

@author: manuelbb
"""

import numpy as np

from inspect import isfunction
from .globals import julia_main
from uuid import uuid4

from .utilities import tprint


try:
    global plt
    import matplotlib.pyplot as plt
    from matplotlib import cm
    import matplotlib.patches as mpatches
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
    # either make Morbit accept PyCall.PyObjects or 
    # find a way to deepcopy a PyObject in the first argument of `pycall`
    tmp_name = "tmp_" + str(uuid4()).replace("-","_")
    exec( f"jl.{tmp_name} = func")
    jl_func = jl.eval(f"""
    function(x)
        return PC.pycall(
            {tmp_name},
            PC.PyAny,
            x
        )
    end
    """)
    return jl_func

def forbid_autodiff( arg_dict ):
    jl = julia_main()
    for g in ["grad", "gradient", "gradients", "jacobian"]:
        if g in arg_dict.keys():
            if (arg_dict[g] == "autodiff" or arg_dict[g] == jl.Symbol("autodiff")):
                print("WARNING Automatic differentiation of Python functions not supported. Using Finite Differences instead.\nYou can also provide a gradient function with the keyword `grad`.")
                arg_dict.pop(g)
            else:
                break
    else:
        print("WARNING Automatic differentiation of Python functions not supported. Using Finite Differences instead.\nYou can also provide a gradient function with the keyword `grad`.")
    
        arg_dict["gradients"] = jl.Symbol("fdm")
    return
            

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
        "polynomial_degree" : "polynomial_degree",
        "θ_enlarge_1" : "theta_enlarge_1",
        "θ_enlarge_2" : "theta_enlarge_2",
        "θ_pivot" : "theta_pivot",
        "θ_pivot_cholesky" : "theta_pivot_cholesky",
        "require_linear": "require_linear",
        "max_model_points" : "max_model_points",
        "use_max_points" : "use_max_points",
        "sampling_algorithm" : "sampling_algorithm",
        "sampling_algorithm2" : "sampling_allgorithm2",
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
        "save_path" : "save_path",
        "io_lock" : "io_lock",
        "algo1_solver" : "algo1_solver",
        "algo2_solver" : "algo2_solver",
        "algo1_max_evals" : "algo1_max_evals",
        "algo2_max_evals" : "algo2_max_evals",
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
    def __init__(self, lb = [], ub = [], n_vars = -1):

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
      
        if np.size(lb) > 0:
            self.jlObj = self.jl.MixedMOP(self.lb, self.ub)
        elif np.size(lb) == 0 and n_vars > 0:
            self.jlObj = self.jl.MixedMOP( n_vars )
        else:
            print("Number of variables must be specified for unconstrained problems.")
            return
        
        self.iterDataObj = None
            
    @property
    def n_objfs(self):
        return self.eval("Morbit.num_objectives")(self.jlObj)
    
    def _prepare_args(self, allowed_properties, *args, **kwargs): 
        arg_dict0 = get_kwargs(allowed_properties, *args, **kwargs)
        arg_dict = wrap_funcs_in_dict(arg_dict0)
        can_batch = batch_eval(*args, **kwargs) 
        return arg_dict, can_batch
        
    def _add_objective_jl(self, func, cfg, can_batch = False ):
        self.jl.add_objective_b( self.jlObj, wrap_func( func ), cfg.jlObj, batch_eval = can_batch )
        return self.n_objfs
    
    def _add_objective(self, func, cfgClass, *args, **kwargs):
        arg_dict, batch_eval = self._prepare_args(cfgClass.py_jl_props, *args, **kwargs)
        if cfgClass == ExactConfig:
            forbid_autodiff(arg_dict) 
        cfg = cfgClass(*[], **arg_dict) 

        return self._add_objective_jl(func, cfg, batch_eval)
    
    def _add_vec_objective_jl(self, func, cfg, can_batch = False, n_out = -1):
        self.jl.add_vector_objective_b( 
            self.jlObj, 
            wrap_func( func ), 
            cfg.jlObj, 
            n_out = n_out, 
            batch_eval = can_batch 
        )
        return self.n_objfs
        
    def _add_vec_objective(self, func, cfgClass, *args, n_out = -1, **kwargs ):
        if n_out < 1:
            print("Please specify number of function outputs with kwargs `n_out`.")
            return
                
        arg_dict, batch_eval = self._prepare_args(cfgClass.py_jl_props, *args, **kwargs)
        if cfgClass == ExactConfig:
            forbid_autodiff(arg_dict) 
            
        cfg = cfgClass(*[], **arg_dict) 
        
        return self._add_vec_objective_jl(func, cfg, batch_eval, n_out)
    
    def add_cheap_objective(self, func, *args, **kwargs):
        return self._add_objective(func, ExactConfig, *args, **kwargs )
    
    def add_lagrange_objective(self, func, *args, **kwargs):
        return self._add_objective(func, LagrangeConfig, *args, **kwargs )
    
    def add_taylor_objective(self, func, *args, **kwargs):
        return self._add_objective(func, TaylorConfig, *args, **kwargs )
    
    def add_rbf_objective(self, func, *args, **kwargs):
        return self._add_objective(func, RbfConfig, *args, **kwargs )

    def add_cheap_vec_objective(self, func, *args, **kwargs ):
        return self._add_vec_objective( func, ExactConfig, *args, **kwargs )
    
    def add_lagrange_vec_objective(self, func, *args, **kwargs ):
        return self._add_vec_objective( func, LagrangeConfig, *args, **kwargs )
    
    def add_taylor_vec_objective(self, func, *args, **kwargs ):
        return self._add_vec_objective( func, TaylorConfig, *args, **kwargs )
    
    def add_rbf_vec_objective(self, func, *args, **kwargs ):
        return self._add_vec_objective( func, RbfConfig, *args, **kwargs )
              
    @property 
    def iter_data(self):
         return self.iterDataObj
    
    @property 
    def eval_sites(self):
        if self.iterDataObj:
            return self.eval("Morbit.get_sites ∘ Morbit.db")( self.iterDataObj )
    
    @property 
    def eval_vectors(self): 
        if self.iterDataObj:
            return self.eval("Morbit.get_values ∘ Morbit.db")( self.iterDataObj )
    
    @property
    def iter_sites(self):
        if self.iterDataObj:
            return self.eval("Morbit.get_iterate_sites ∘ Morbit.db")( self.iterDataObj )
    
    @property
    def iter_vectors(self):
        if self.iterDataObj:
            return self.eval("Morbit.get_iterate_values ∘ Morbit.db")( self.iterDataObj )
        
    @property
    def n_iters(self):
        if self.iterDataObj:
            return self.eval("Morbit.num_iterations")( self.iterDataObj )
    #     len_iter_array = len( self.jlObj.iter_data.iterate_indices )
    #     return 0 if len_iter_array == 0 else len_iter_array - 1
    
    @property
    def n_evals(self):
         return len( self.eval_sites )
    
    # @property 
    # def ρ_array(self):
    #     return self.jlObj.iter_data.ρ_array
    
    # @property 
    # def n_acceptable_steps(self):
    #     return np.sum( self.ρ_array >= self.ν_accept )
    
    # @property
    # def n_successful_steps(self):
    #     return np.sum( self.ρ_array >= self.ν_success )
    
    # def save(self, filename = None):
    #     self.eval("save_config")( self.jlObj, filename)
        
    # def load(self, filename):
    #     self.jlObj = self.eval("load_config")(filename)
    
    def plot_objectives(self, objf_indices = None, iter_indices = None, scale = True):
        if self.iter_data and plt:
            fig, ax = plt.subplots()
            
            if not iter_indices: 
                iter_indices = list(range(self.n_iters))
                
            if not objf_indices:
                objf_indices = list(range(self.n_objfs))
                
            n_iter = len(iter_indices)
                
            # scale each dimension of each value vector
            Y = [ v[objf_indices] for v in self.iter_vectors ]
            Ymin = np.min( Y )
            if scale:   
                ax.set_title("Objective Values (Scaled)")
                LB = np.min( Y, axis = 0 )
                UB = np.max( Y, axis = 0 )
                W = UB - LB
                Ys = [ (y - LB)/W for y in Y ]
            else:
                ax.set_title("Objective Values")
                Ys = Y
            
            colors = [ cm.coolwarm( j ) for j in np.linspace(0,1, n_iter ) ]
            j = 0
            for (j,i) in enumerate(iter_indices):
                y = Ys[i]
                ax.fill_between(range(len(y)), Ymin, y, color = colors[j])
                
            ax.set_xticks( range(len(objf_indices) ))
            ax.set_xticklabels( [str(i) for i in objf_indices] )
            ax.set_xlabel("Objective Indices")
            
            leg_indices = np.linspace(0, n_iter - 1, min(10, n_iter), dtype = int)
            
            leg_handles = [ 
                mpatches.Patch(
                    color=colors[j], 
                    label=f"It. {iter_indices[j]}") 
                for j in leg_indices 
            ]
            ax.legend( bbox_to_anchor = (1.05, 1), loc = "upper left", handles = leg_handles )
            
            fig.show()
            
    def plot_iterates(self, dims = [0,1]):
        # TODO: scaled and unscaled?
        if self.iter_data and plt:
            fig, ax = plt.subplots()
            ax.set_title("Iterates and Evaluation Sites.")
            all_sites = [ v[dims] for v in self.eval_sites ]
            it_sites = [ v[dims] for v in self.iter_sites ]
            
            ax.scatter( [v[0] for v in all_sites ], [v[1] for v in all_sites], color = "lightgray" )
            ax.plot( [v[0] for v in it_sites ], [v[1] for v in it_sites], "o-")
            ax.set_xlabel(f"var {dims[0]}")
            ax.set_ylabel(f"var {dims[1]}")
            fig.show()
            
    # TODO 
    def plot_objective_iterates(self, objf_indices = [0,1]):
        # Plot evaluations of objectives with indices `objf_indices` 
        # and iterations in objective space, similar to `plot_iterates`.
        pass
            
    
            
class GenericConfig():
    def __init__(self, jlObj = None):
        self.jl = julia_main()
        self.eval = self.jl.eval
        
        if not jlObj:
            jlObj = self.eval("Morbit.EmptyConfig()")
            
        self.jlObj = jlObj
       
    def show(self):
        print( self.jlObj )
        

class AlgoConfig( GenericConfig ):
    """A wrapper class for the AlgoConfig structure provided by Morbit in Julia.
    
    Initiliaze by passing a dict of options or pass the arguments as keyword=value pairs.
    The created AlgoConfig instance becomes available as the .obj property"""
    
    
    jl_py_props = {
        "max_evals" : ("max_evals", "(Int) Maximum number of objective evaluations (including AutoDiff and FiniteDiff evals)."),
        "max_iter" : ("max_iter", "(Int) Maximum number of iterations."),
        "γ_crit" : ("gamma_crit", "(Real) Radius shrinking factor in criticality loop."),
        "max_critical_loops" : ("max_critical_loops", "(Int) Maximum number of critical loops before break."),
        "ε_crit" : ("eps_crit", "(Real) Threshold for criticality test."),
        "count_nonlinear_iterations" : ("count_conlinear_iterations", "(Boolean) Count iterations if model is not fully linear."),
        "Δ₀" : ("delta_init", "Initial trust region radius (relative to [0,1]^n if box constrained)."),
        "Δ_max" : ("delta_max", "Maximum trust region radius"),
        "f_tol_rel" : ("f_tol_rel", ""),
        "x_tol_rel" : ("x_tol_rel", ""),
        "f_tol_abs" : ("f_tol_abs", ""),
        "x_tol_abs" : ("x_tol_abs", ""),
        "ω_tol_abs" : ("omega_tol_abs", ""),
        "Δ_tol_rel" : ("delta_tol_rel", ""),
        "ω_tol_abs" : ("omega_tol_abs", ""),
        "Δ_tol_abs" : ("delta_tol_abs", ""),
        "descent_method" : ("descent_method","Method for step calculation ('steepest_descent', 'ps', 'ds')"),
        "strict_backtracking" : ("strict_backtracking", "Bool"),
        "reference_direction" : ("reference_direction", "(Vector{<:Real}) Alternative to utopia vector."),
        "reference_point" : ("reference_point", "(Vector{<:Real}) Utopia vector for image space subproblems."),
        "max_ideal_point_problem_evals" : ("max_ideal_point_problem_evals", ""),
        "max_ps_problem_evals" : ("max_ps_problem_evals", ""),
        "max_ps_polish_evals" : ("max_ps_polish_evals", ""),
        "ps_algo" : ("ps_algo", ""),
        "ideal_point_algo" : ("ideal_point_algo", ""),
        "ps_polish_algo" : ("ps_polish_algo",""),
        "strict_acceptance_test" : ("strict_acceptance_test", ""),
        "ν_success" : ("nu_success", "Large acceptance parameter."), # real  
        "ν_accept" : ("nu_accept", "Small acceptance parameter."),  # real
        "db" :("db", ""),
        "μ" : ("mu", "Largest criticality factor for criticality test."),
        "β" : ("beta", "Smaller criticality factor for criticality test."),
        "radius_update_method" : ("radius_update_method", "(Symbol) Radius update method ('standard' or 'steplength')"),
        "γ_grow" : ("gamma_grow", "Radius growing factor."), # real
        "γ_shrink" : ("gamma_shrink", "Radius shrinking factor for acceptable trial points."), # real
        "γ_shrink_much" : ("gamma_shrink_much", "Severe radius shrinking factor"),
    }
    
    # TODO this is a bit redundant...
    py_jl_props = {v[0] : k for (k,v) in jl_py_props.items() }
    
    def __init__(self, *args, **kwargs):
       
        if len(args) == 1 and isinstance(args[0], dict):
            arg_dict = args[0]
        else:
            arg_dict = kwargs
            
        init_props = {AlgoConfig.py_jl_props[k] : v for (k,v) in arg_dict.items() 
                      if k in AlgoConfig.py_jl_props.keys() }
        jl = julia_main()
        super().__init__( jl.AlgoConfig(*[], **init_props ) )


        
# Set properties for AlgoConfig class dynamically
for jl_prop, prop_tuple in AlgoConfig.jl_py_props.items():
    
    def get_property( jl_prop, prop_tuple ):
        prop_name, prop_doc = prop_tuple
        
        @property
        def new_property(self):
            return self.jl.getfield( self.jlObj, self.jl.Symbol( jl_prop ) )
        
        @new_property.setter 
        def new_property(self,val):
            self.jl.setfield_b( self.jlObj, self.jl.Symbol(jl_prop), val )
        
        new_property.__doc__ = prop_doc
        return new_property
    setattr(AlgoConfig, prop_tuple[0], get_property(jl_prop, prop_tuple))    


        
