#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 16:12:14 2020

@author: manuelbb
"""

from .globals import JL_MAIN, get_MORBIT_PATH, get_MORBIT_SYS_IMG, PATCH_ENV, WRAPPER_PATH
from .utilities import tprint

from julia.api import Julia

from os.path import isfile


#import os
#from subprocess import check_output, CalledProcessError   

# def patch_environment():
#     # Patch environment for julia-py to be able to pre-compile Plots.jl
#     julia_cmd = "julia"
#     patch_script_path = os.path.join( WRAPPER_PATH, "patch_openssl_jll.jl")
#     # run a subprocess so as to have a seperate julia process
    
#     try:
#         ld_path = check_output([julia_cmd, patch_script_path]).decode().replace("\n", "")
#         compile_env = os.environ.copy()
#         if not "LD_LIBRARY_PATH" in compile_env.keys():
#             compile_env["LD_LIBRARY_PATH"] = ld_path
#         else:
#             compile_env["LD_LIBRARY_PATH"] = ld_path + ":" + compile_env["LD_LIBRARY_PATH"]
        
#         return compile_env
    
#     except CalledProcessError as e:
#         print("Could not patch the enivronment, compilation is likely to fail.")
#         print(e)
     

def initialize_julia( ):
    global JL_MAIN
    
    MORBIT_SYS_IMG = get_MORBIT_SYS_IMG()
    if isinstance(MORBIT_SYS_IMG, str) and isfile( MORBIT_SYS_IMG ):
        sysimage_path = MORBIT_SYS_IMG
        tprint(f"\tUsing sysimage at {sysimage_path}")
    else:
        sysimage_path = None
        tprint(f"\tGlobal variable MORBIT_SYS_IMG = {MORBIT_SYS_IMG} not usable.\n\tYou can create a sysimage using 'create_sysimg()'.")
        
    # # low level initialization    
    # if PATCH_ENV:
    #     tprint("Patching environment.")
    #     os.environ["LD_LIBRARY_PATH"] = patch_environment()["LD_LIBRARY_PATH"]
    #     importlib.reload(julia.api)
    #print(os.environ["LD_LIBRARY_PATH"])
    
    tprint("MOP: Initializing Julia runtime. First time might take a bit.")
    #from julia.api import Julia # NOTE import happens here so that the module is aware of environment variable changes
    
    try:
        Julia( compiled_modules = False, sysimage = sysimage_path) # we have to to this twice somehow because PyJulia cannot handle custom sysimages well
    except:
        "Could not load Julia. Maybe patch the environment."
        raise
    
    from julia import Main 
    
    # loading MORBIT module
    Main.eval('println(ENV["LD_LIBRARY_PATH"])')
    Main.using("Pkg")
    Main.eval(f'Pkg.activate("{get_MORBIT_PATH()}")') 
    tprint("Loading Morbit module. If no sysimage was provided this will take some time.")
    Main.using("Morbit")
    
    tprint("Julia runtime all set up!")
    
    JL_MAIN = Main

def julia_main():
    global JL_MAIN
    if not JL_MAIN:
        initialize_julia()
    return JL_MAIN
