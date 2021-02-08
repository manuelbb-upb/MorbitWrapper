from SolveOF.OpenFOAM.classesOpenFOAM import ClassOpenFOAM, ClassObservable
from SolveOF.QuaSiModO import createSequence
from SolveOF.visualization import plot

import numpy as np

import os 
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

T = 5.0
h = 0.05

uMin = [-2.0]
uMax = [2.0]
uGrid = np.array([uMin, [0.0], uMax])

dimZ = 2
y0 = []

# -------------------------------------------------------------------------------------------------------------------- #
# Model creation
# -------------------------------------------------------------------------------------------------------------------- #

obs = ClassObservable(forceCoeffsPatches=forceCoeffsPatches, ARef=ARef, lRef=lRef, writeY=writeY, writeGrad=writeGrad)
of = ClassOpenFOAM(pathProblem, obs, pathOut, nProc, nInputs, dimInputs, iInputs, h=hFOAM, Re=Re, dimSpace=dimSpace)
model = of.createOrUpdateModel(uMin=uMin, uMax=uMax, hWrite=h, dimZ=dimZ, uGrid=uGrid)

u, iu = createSequence(h, T, uGrid, nhMin=1, nhMax=10, typeSequence='spline')

[y, z, t, _] = model.integrate(y0, u, 0.0)
#[y2, z2, t2, _] = model.integrate(y0, u, t[-1])

#plot(z={'t': t, 'z': z, 'iplot': 0}, z2={'t': t2, 'z2': z2, 'iplot': 1}, u={'t': t, 'u': u, 'iplot': 2}, iu={'t': t, 'iu': iu, 'iplot': 3})
plot( z={'t': t, 'z': z, 'iplot': 0}, u={'t': t, 'u': u, 'iplot': 1})
