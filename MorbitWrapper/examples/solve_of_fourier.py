from SolveOF.OpenFOAM.classesOpenFOAM import ClassOpenFOAM, ClassObservable

import numpy as np
import os 

def createFourierControl(h, T, fourier_params):
    '''
    Return the values of a truncated Fourier series on a regular time grid from 0 to `T` with stepsize `h`.
    Each *row* of `fourier_params` is set of parameters [ A_k, φ_k ] determining the k-th term
    of a Fourier series in amplitude phase form, i.e. A_k * cos( 2*π*k*t/T - φ_k ).
    (φ_0 is ignored; if only 1 row is present, then the constant function A_k/2 is evaluated.)    
    '''
    
    n_terms, n_coeff = fourier_params.shape
    
    if n_coeff != 2 or n_terms == 0:
        raise "Third parameter `fourier_params` must have shape (k,2), k >= 1." 
    
    nt = int(T / h + 1)
    t = np.linspace(0, T, nt)
    
    def fourier_series( tau ):
        ret_val = fourier_params[0,0] / 2.0;
        for k in range(1, n_terms ): 
            ret_val += ( fourier_params[k,0] * np.cos( 2*np.pi * k * tau / T - fourier_params[k,1] ) )
        return ret_val
    
    u = np.zeros((nt,1), dtype = float )
    for i in range(nt):
        u[i] = fourier_series(t[i])

    iu = np.array([])

    return u, iu

def createBergmannControl(h, T, paramVector ):
    ''' 
    Return the values of the function
    A_1 * sin( 2*π*S_1*t ) * sin( 2*π*S_2*t - A_2 * sin(2*π*S_3*t) )
    on a regular time grid from 0 to `T`with stepsize `h`.
    paramVector contains the parameters
    A_1, A_2, S_1, S_2, S_3 
    in that order
    '''
    
    A_1, A_2, S_1, S_2, S_3 = paramVector
    nt = int(T / h + 1)
    t = np.linspace(0, T, nt)
    
    def control( tau ):
        return A_1 * np.sin( 2*np.pi*S_1*tau ) * np.sin( 2*np.pi*S_2*tau - A_2 * np.sin(2*np.pi*S_3*tau) )
    
    u = control(t)
    iu = np.array([])
    
    return u,iu 
                    

def cost_and_drag_objectives( u, iu, T = 2.0, h = 0.1 ):
    dir_path = "/project_files/SolveOF/"
    
    # -------------------------------------------------------------------------------------------------------------------- #
    # OpenFOAM: Define model
    # -------------------------------------------------------------------------------------------------------------------- #
    pathProblem = os.path.join( dir_path, 'OpenFOAM/problems/cylinder')
    pathOut = os.path.join(dir_path, 'OpenFOAM/data')
    
    nProc = 1
    
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
    Re = 100.0
    hFOAM = 0.01
    dimSpace = 2
          
    uMin = uMax = uGrid = np.array([[]])
    
    dimZ = 2
    y0 = []
    
    # -------------------------------------------------------------------------------------------------------------------- #
    # Model creation
    # -------------------------------------------------------------------------------------------------------------------- #
    
    obs = ClassObservable(forceCoeffsPatches=forceCoeffsPatches, ARef=ARef, lRef=lRef, writeY=writeY, writeGrad=writeGrad)
    of = ClassOpenFOAM(pathProblem, obs, pathOut, nProc, nInputs, dimInputs, iInputs, h=hFOAM, Re=Re, dimSpace=dimSpace)
    model = of.createOrUpdateModel(uMin=uMin, uMax=uMax, hWrite=h, dimZ=dimZ, uGrid=uGrid)
    
    
    [y, z, t, _] = model.integrate(y0, u, 0.0)
    
    #plot( z={'t': t, 'z': z, 'iplot': 0}, u={'t': t, 'u': u, 'iplot': 1})
    
    cost_u = np.trapz( (u**2).flatten(), t )
    drag = np.trapz( (z[:,0]**2).flatten(), t )
    
    print(f"Control cost was {cost_u:.4f} and drag integral is {drag:.4f}.")
    return cost_u, drag

if __name__ == "__main__":
    n_fourier_terms = 10
    fourier_params = np.array( [ [-1.0, 0.0] ] ) + np.array( [ [ 2.0, 2.0*np.pi ] ] ) * np.random.rand( n_fourier_terms, 2 )

    cost_and_drag_objectives(fourier_params)