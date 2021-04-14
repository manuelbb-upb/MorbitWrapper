#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 16:40:36 2020

@author: manuelbb
"""

import os.path

JULIA_MAIN = None    

# PROJECT SETTINGS (Defaults for Singularity Container)
#JULIA_RUNTIME = None    # None if 'julia' is available in PATH; else provide path to Julia installation

MORBIT_SYS_IMG = None

# some setter function to change module variables from the outside    

def set_MORBIT_SYS_IMG( new_val ):
    global MORBIT_SYS_IMG
    MORBIT_SYS_IMG = new_val

def get_MORBIT_SYS_IMG():
    return MORBIT_SYS_IMG

def set_JULIA_MAIN( jl_instance ):
    global JULIA_MAIN 
    JULIA_MAIN = jl_instance

def get_JULIA_MAIN():
    global JULIA_MAIN
    return JULIA_MAIN 

def julia_main():
    jl = get_JULIA_MAIN()
    if not jl:
        jl = initialize_julia()
    return jl

def initialize_julia( ): 
    from .utilities import tprint

    MORBIT_SYS_IMG = get_MORBIT_SYS_IMG()
    if isinstance(MORBIT_SYS_IMG, str) and os.path.isfile( MORBIT_SYS_IMG ):
        sysimage_path = MORBIT_SYS_IMG
        tprint(f"\tUsing sysimage at {sysimage_path}")
    else:
        sysimage_path = None
        tprint(f"\tGlobal variable MORBIT_SYS_IMG = {MORBIT_SYS_IMG} not usable.\n\tYou can create a sysimage using 'create_sysimg()'.")
    
    tprint("MOP: Initializing Julia runtime. First time might take a bit.")
    
    try:
        from julia.api import Julia
        Julia( compiled_modules = False, sysimage = sysimage_path) # we have to to this twice somehow because PyJulia cannot handle custom sysimages well
    except:
        "Could not load Julia. Maybe patch the environment."
        raise
    
    from julia import Main

    Main.include( os.path.join( os.path.dirname( __file__ ) , "pycall_sym.jl" ) )
    # loading MORBIT module
    #Main.using("Pkg")
    tprint("Loading Morbit module. If no sysimage was provided this will take some time.")
    Main.using("Morbit")
    
    tprint("Julia runtime all set up!")
    
    set_JULIA_MAIN(Main)
    return Main

                    
                
                
            