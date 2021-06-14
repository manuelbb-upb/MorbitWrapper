# This demo shows how to combine both the Julia algorithm 
# as well as a MATLAB objective function from within Python.

# For this to work the Matlab Engine for Python has to be installed.
# See https://de.mathworks.com/help/matlab/matlab-engine-for-python.html

import sys 
import os 

# comment the following line if morbitwrapper has been installed globally:
sys.path.append( os.path.join( os.path.dirname( __file__ ), ".." ) )

import morbitwrapper as mw
import numpy as np
import matlab.engine
import matlab as ml

# initialize matlab engine first
eng = matlab.engine.start_matlab()

# define a vector objectives that wraps the matlab objective 
# `matlab_objectives` (defined in .m-file)
def objfs( x_vec ):
    # x_vec is provided by the algorithm and will be
    # a numpy array, so we convert to matlab array first
    x = ml.double( x_vec.tolist() )
    # `matlab_objectives` has to outputs, we have specifiy this  
    y = eng.matlab_objectives( x, nargout = 2 )
    return np.array(y, dtype=float)

#def objfs( x ):
#    return [ np.sum( (x - 1)**2 ), np.sum( ( x + 1 )**2 ) ]

# now initialize our optimization problem
# # variable boundaries:
n_vars = 2
lb = np.full( n_vars, -4 )
ub = np.full( n_vars, 4 )

# # define constrained problem (invokes julia)
mw.set_JULIA_RUNTIME("julia")
mop = mw.MOP( lb, ub )

# # add vector objective
mop.add_rbf_vec_objective( objfs, n_out = 2 )

# # optimize!
cfg = mw.AlgoConfig( max_iter = 20 )
x0 = np.array( [-3.0, 4.0] )

x, fx = mw.optimize( mop, x0, cfg )
