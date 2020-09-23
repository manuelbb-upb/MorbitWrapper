#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 16:40:36 2020

@author: manuelbb
"""

from os.path import join
from .runtime_setup import initialize_julia
import pathlib

JULIA_MAIN = None    

# PROJECT SETTINGS (Defaults for Singularity Container)
#JULIA_RUNTIME = None    # None if 'julia' is available in PATH; else provide path to Julia installation

PATCH_ENV = True    # set to true if sysimage or Module relies on Plots

PROJECT_FILES_PATH = "/project_files"
MORBIT_PATH = join(PROJECT_FILES_PATH, "Morbit")       # Path to Morbit project folder
MORBIT_SYS_IMG = join("/user_files", "pyMorbit.so") # pre-compiled Julia sysimage to speed up initialization
WRAPPER_PATH = pathlib.Path(__file__).parent.absolute()

def path_info():
    print("PROJECT_FILES_PATH: " + PROJECT_FILES_PATH )
    print("MORBIT_SYS_IMG: " + MORBIT_SYS_IMG )
    print("MORBIT_PATH: " + MORBIT_PATH )

# some setter function to change module variables from the outside    
def set_MORBIT_SYS_IMG( new_val ):
    global MORBIT_SYS_IMG
    MORBIT_SYS_IMG = new_val

def set_MORBIT_PATH( new_val ):
    global MORBIT_PATH
    MORBIT_PATH = new_val
    
def get_MORBIT_SYS_IMG():
    return MORBIT_SYS_IMG

def get_MORBIT_PATH():
    global MORBIT_PATH
    return MORBIT_PATH

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

# READ ALL CONFIGURATION ARGUMENTS DIRECTLY FROM JULIA FILE
import re
pattern = r"(?<=(?:@with_kw mutable struct AlgoConfig))[\W|\w]*(?=(?:end))"     # regex to match definition block of AlgoConfig in 'data_structures.jl'
CONFIG_ARGS = {}
with open(join(MORBIT_PATH, "src", "data_structures.jl"), "r", encoding="utf-8") as f:
    content = f.read()
    regex_match = re.search(pattern, content)
    if regex_match:
        data = regex_match.group(0)
        lines = [line.strip() for line in data.split("\n")]
        for line in lines:
            if len(line) > 0 and not ( line.startswith("#") or line.startswith("@") ):
                if "::" in line:
                    split_elem = "::"
                elif "=" in line:
                    split_elem = "="
                else:
                    continue
                k,v = [s.strip() for s in line.split(split_elem, 1)]
                CONFIG_ARGS[k] = v
                    
                    
                
                
            
