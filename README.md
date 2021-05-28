# MorbitWrapper
Python wrapper for Morbit julia module.
Basic functionalities of the [original module](https://github.com/manuelbb-upb/Morbit) are wrapped and you 
can refer to the README and doucmentation of the Julia package for more detailed description of the internal parameters.  
Note, the Python constructors are written to accept the parametrs in a more verbose way: 
`mu` instead of `μ` and `delta_init` instead of `Δ₀`.

## Basic usage 
**There is a folder "examples" with some scripts inside.**

### Wrapper settings 

First `morbitwrapper` should be imported.
```python
import MorbitWrapper as mw
```
The wrapper needs a working Julia installation, preferably version 1.6+.
By default, we assume `julia` to be available in the global `PATH`.
You can set an alternative version via
```python
mw.set_JULIA_RUNTIME( abspath_to_julia_bin )
```
You can also specify a Julia environment, where `Morbit.jl` is available:
```python 
mw.set_JULIA_ENV("@v1.6") # if you have Morbit in the default environment of Julia 1.6.x
```
If Morbit is not available in the environment, then the wrapper will try to 
`add` it. 
Depending on the file system permissions this could error.
The default setting is `None` which leads to a temporary environment being created.

!!! note 
    Due to Julia being JIT compiled, the initial startup time might be long.
    To overcome this, a sysimage with `Morbit.jl` and `PyCall` can be compiled.
    Simply specify `mw.set_MORBIT_SYS_IMG( target_path )`.
    The creation will take a lot of time (first, a base image has to be compiled 
    and then two sysimage have to be compiled on top of that).  
    If, at some point the sysimage has to be updated, simply delete the output file.
    To deactive the image, delete the code snippet or use 
    ```python
    mw.set_MORBIT_SYS_IMG( None )
    ```
    
The Julia depot path can be set via `mw.set_JULIA_DEPOT_PATH`. 
This can be used to setup a self-contained Project environment.

All setings can be set in a JSON file and then be loaded with `mw.load_settings`.
A possible settings file could look like this:
```
{
    "JULIA_RUNTIME" : "./julia-1.6.1/bin/julia",
    "JULIA_DEPOT_PATH" : "./.julia_depot",
    "JULIA_ENV" : "@v1.6",
    "MORBIT_SYS_IMG" : "./pycall_morbit.sysimg"
}
```
Paths are interpreted relative to the location of the JSON-file.

### Problem Setup and Optimization
Below is a minimal example to solve a box-constrained problem with two 
objectives, where one objective is considered expensive and the other is cheap:

```python

mop = mw.MOP(lb = [-4.0, -4.0], ub =[4.0,4.0])

def f1(x):
    return (x[0]-1)**2 + (x[1]-1)**2
    
f2 = lambda x : (x[0]+1)**2 + (x[1]+1)**2 
df2 = lambda x : [ 2*(x[0]+1), 2*(x[1]+1) ]

mop.add_rbf_objective(f1)
mop.add_cheap_objective(f2, grad = df2)

conf = mw.AlgoConfig( max_iter = 10 )
x, y = mop.optimize([3.14, 4], conf)
```

For unconstrained problems the number of variables has to be provided with the keyword
argument `n_vars`:
```python
mop = mw.MOP( nvars = 2 )
```

A scalar-valued objective functions `func` can be added 
with the following methods:
* `mop.add_rbf_objective(func, **kwargs)` for modelling with radial basis functions.
* `mop.add_lagrange_objective(func, **kwargs)` for modelling with Lagrange polynomials.
* `mop.add_taylor_objective(func, **kwargs)` for modelling with Taylor polynomials.
* `mop.add_cheap_objective(func, **kwargs)` to not perform modelling.

The supported keyword arguments for RBF objectives can be found 
with `mw.RbfConfig.jl_py_props.values()`. 
Similarly, there are classes `mw.LagrangeConfig`, `mw.TaylorConfig` and `mw.ExactConfig`.

To add vector objectives, simply use the `add_rbf_vec_objective` etc.
**These methods require the `n_out` (number of outputs) keyword argument!**

### Plotting and Iteration Data
After optimization, the problem `mop` contains an interesting Julia object, 
`mop.iterDataObj`. 
Some iteration information is stored and can be retrieved as numpy arrays with 
the following methods:
```python
mop.n_iters()       # number of iterations
mop.n_evals()       # number of evaluation sites
mop.eval_sites()    # all evaluation vectors in decision space
mop.eval_vectors()  # all vectors in objective space
mop.iter_sites()    # iteration sites in decision space 
mop.iter_vectors()  # iteration vectors in objective space.
```
Try the following plotting functions:
```python 
mop.plot_objectives( objf_indices = None, iter_indices = None, scale = True) 
mop.plot_iterates( dims = [0,1] )
```