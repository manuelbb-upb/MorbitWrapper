#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 18:03:08 2020

@author: manuelbb
"""


from SolveOF.OpenFOAM.classesOpenFOAM import ClassOpenFOAM, ClassObservable

import MorbitWrapper as mw

import numpy as np
import os 
from pathlib import Path
from datetime.datetime import now
import tempfile
from multiprocessing import Pool, cpu_count

n_processes = 4 #cpu_count()    # Number of cores to use in ParallelMode

# MOP Solver Configuration
n_vars = 2 # 2 for simple Bergmann Control (sine) or 5 for more complex control
opt = mw.AlgoConfig( max_evals = 20, Δ_0 = .1 )
lb = np.zeros(n_vars, dtype = float)
ub = np.array( [30.0, 2*np.pi] ) # np.array([30, 30, 2*np.pi, 2*np.pi, 2*np.pi])
x_0 = np.array( [4.0, 1.0/120] )

# -------------------------------------------------------------------------------------------------------------------- #
# OpenFOAM: Define model
# -------------------------------------------------------------------------------------------------------------------- #
pathProblem = os.path.abspath( '/project_files/SolveOF/OpenFOAM/problems/cylinder' )

nProc = 2

nInputs = 1
dimInputs = 1
iInputs = [0]

# -------------------------------------------------------------------------------------------------------------------- #
# OpenFOAM: Define what to observe
# -------------------------------------------------------------------------------------------------------------------- #

writeGrad = False
writeY = False

forceCoeffsPatches = ['cylinder']
ARef = 1.0
lRef = 1.0

# -------------------------------------------------------------------------------------------------------------------- #
# Set parameters
# -------------------------------------------------------------------------------------------------------------------- #

T = 30
h = 1.5e-2

Re = 200.0
hFOAM = 0.01
dimSpace = 2
      
uMin = uMax = uGrid = np.array([[]])

dimZ = 2
y0 = []


def createBergmannControl( paramVector ):
    ''' 
    Return the values of the function
    A_1 * sin( 2*π*S_1*t ) 
    on a regular time grid from 0 to `T`with stepsize `h`.
    paramVector contains the parameters
    A_1, S_1
    in that order
    '''
    global h,T
    
    A_1, S_1 = paramVector
    nt = int(T / h + 1)
    t = np.linspace(0, T, nt)
    
    def control( tau ):
        return A_1 * np.sin( 2*np.pi*S_1*tau )
    
    u = control(t).reshape((-1,1))    
    iu = np.array([])
    
    return u,iu 
    
#def createBergmannControl( paramVector ):
#    ''' 
#    Return the values of the function
#    A_1 * sin( 2*π*S_1*t ) * sin( 2*π*S_2*t - A_2 * sin(2*π*S_3*t) )
#    on a regular time grid from 0 to `T`with stepsize `h`.
#    paramVector contains the parameters
#    A_1, A_2, S_1, S_2, S_3 
#    in that order
#    '''
#    global h,T
#    
#    A_1, A_2, S_1, S_2, S_3 = paramVector
#    nt = int(T / h + 1)
#    t = np.linspace(0, T, nt)
#    
#    def control( tau ):
#        return A_1 * np.sin( 2*np.pi*S_1*tau ) * np.sin( 2*np.pi*S_2*tau - A_2 * np.sin(2*np.pi*S_3*tau) )
#    
#    u = control(t)
#    iu = np.array([])
#    
#    return u,iu 

def simulation( param_vector ):
    with tempfile.TemporaryDirectory() as pathOut:
        u, iu = createBergmannControl( param_vector )
        return cost_and_drag_objectives(u,iu, pathOut)
        
def batch_simulation( list_of_param_vecs ):
    if isinstance( list_of_param_vecs, np.ndarray ):
        print("Single simulation.")
        return simulation( list_of_param_vecs )
    else:
        print(f"Working on {len(list_of_param_vecs)} problems with {n_processes} cores.")
        with Pool(n_processes) as p:
            ret_values = p.map( simulation, list_of_param_vecs ) 
            return ret_values                    

def cost_and_drag_objectives( u, iu,pathOut ):
   
#    # Model creation
    obs = ClassObservable(forceCoeffsPatches=forceCoeffsPatches, ARef=ARef, lRef=lRef, writeY=writeY, writeGrad=writeGrad)
    of = ClassOpenFOAM(pathProblem, obs, pathOut, nProc, nInputs, dimInputs, iInputs, h=hFOAM, Re=Re, dimSpace=dimSpace)
    model = of.createOrUpdateModel(uMin=uMin, uMax=uMax, hWrite=h, dimZ=dimZ, uGrid=uGrid)
    
    # Simulation
    [y, z, t, _] = model.integrate(y0, u, 0.0)
    
#    # Integration over time
    cost_u = np.trapz( (u**2).flatten(), t )
#    drag = max(u)
    drag = np.trapz( (z[:,0]**2).flatten(), t )
    
    return np.array([cost_u, drag], dtype=float)

if __name__ == "__main__":
    mop = mw.MOP( lb = lb, ub = ub )
    mop.add_batch_function(batch_simulation, 2)
    
    x, y = mop.optimize(x_0, opt)
    
    opt.save( os.path.join( Path.home(), now.strftime("results_%d_%h_%y_%H_%M_%S.jld" ) ) )