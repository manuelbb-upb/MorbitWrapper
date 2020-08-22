# MorbitWrapper
Python wrapper for Morbit julia module.
Basic functionalities of the [original module](https://github.com/manuelbb-upb/Morbit) are wrapped and you can refer to the README there for description of the internal parameters.
Note that the only difference is, that `Δ_0` is the inital trust region radius in Python because it is easier to type.

## Basic usage 
Refer to the examples subfolder for more involved examples.
```python
import MorbitWrapper as mw

# mw.set_MORBIT_SYS_IMG(None) # deactivate use of a precompiled sysimage, results in slow startup

mop = mw.MOP(lb = [-4, -4], ub =[4,4])

def f1(x):
    return (x[0]-1)**2 + (x[1]-1)**2
    
f2 = lambda x : (x[0]+1)**2 + (x[1]+1)**2 
df2 = lambda x : [ 2*(x[0]+1), 2*(x[1]+1) ]

mop.add_expensive_function(f1)
mop.add_cheap_function(f2, df2)

conf = mw.AlgoConfig( max_iter = 10, all_objectives_descent = True )
x, y = mop.optimize([3.14, 4], conf)

# conf.save("results.jld") 
```
