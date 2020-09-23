#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 16:12:14 2020

@author: manuelbb
"""

from .globals import set_JULIA_MAIN, get_MORBIT_PATH, get_MORBIT_SYS_IMG
from .utilities import tprint

from julia.api import Julia

from os.path import isfile
     

def initialize_julia( ):    
    MORBIT_SYS_IMG = get_MORBIT_SYS_IMG()
    if isinstance(MORBIT_SYS_IMG, str) and isfile( MORBIT_SYS_IMG ):
        sysimage_path = MORBIT_SYS_IMG
        tprint(f"\tUsing sysimage at {sysimage_path}")
    else:
        sysimage_path = None
        tprint(f"\tGlobal variable MORBIT_SYS_IMG = {MORBIT_SYS_IMG} not usable.\n\tYou can create a sysimage using 'create_sysimg()'.")
    
    tprint("MOP: Initializing Julia runtime. First time might take a bit.")
    #from julia.api import Julia # NOTE import happens here so that the module is aware of environment variable changes
    
    try:
        Julia( compiled_modules = False, sysimage = sysimage_path) # we have to to this twice somehow because PyJulia cannot handle custom sysimages well
    except:
        "Could not load Julia. Maybe patch the environment."
        raise
    
    from julia import Main 
    
    # loading MORBIT module
    Main.using("Pkg")
    Main.eval(f'Pkg.activate("{get_MORBIT_PATH()}")') 
    tprint("Loading Morbit module. If no sysimage was provided this will take some time.")
    Main.using("Morbit")
    
    tprint("Julia runtime all set up!")
    
    set_JULIA_MAIN(Main)
    return Main

