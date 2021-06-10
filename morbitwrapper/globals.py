#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 16:40:36 2020

@author: manuelbb
"""

import json
from pathlib import Path
import os.path
from julia import JuliaError, Julia
from julia.api import JuliaInfo
#from julia.tools import build_pycall
from .utilities import tprint
from julia import install
from julia.sysimage import build_sysimage
from find_libpython import find_libpython
import tempfile
from julia.sysimage import install_packagecompiler_cmd, check_call
import logging


MORBIT_REPO_URL = r"https://github.com/manuelbb-upb/Morbit.jl.git"
MORBIT_UUID = "88936782-c8cd-4a0f-b259-ffb12bfd2869"

JULIA_MAIN = None    

JULIA_RUNTIME = None    # None if 'julia' is available in PATH; else provide path to Julia installation

JULIA_ENV = None

MORBIT_SYS_IMG = None

#JULIA_LOAD_PATH = None
JULIA_DEPOT_PATH = None

# some setter function to change module variables from the outside    

def set_MORBIT_SYS_IMG( new_val ):
    global MORBIT_SYS_IMG
    MORBIT_SYS_IMG = new_val

def get_MORBIT_SYS_IMG():
    return MORBIT_SYS_IMG

def set_JULIA_RUNTIME( new_val ):
    global JULIA_RUNTIME
    JULIA_RUNTIME = new_val
    logging.info( f"Julia RUNTIME set to {JULIA_RUNTIME}" )


def get_JULIA_RUNTIME():
    return JULIA_RUNTIME

def get_JULIA_RUNTIME_NAME():
    jl_rt = get_JULIA_RUNTIME()
    if jl_rt:
        return jl_rt 
    else: 
        return "julia"

def set_JULIA_ENV( new_val ):
    global JULIA_ENV
    JULIA_ENV = new_val
    logging.info( f"Julia Environment set to {JULIA_ENV}" )

def get_JULIA_ENV():
    return JULIA_ENV

def set_JULIA_MAIN( jl_instance ):
    global JULIA_MAIN 
    JULIA_MAIN = jl_instance

def get_JULIA_MAIN():
    global JULIA_MAIN
    return JULIA_MAIN 

def set_JULIA_DEPOT_PATH( new_val ):
    global JULIA_DEPOT_PATH
    JULIA_DEPOT_PATH = new_val
    os.environ["JULIA_DEPOT_PATH"] = new_val
    logging.info( f"Julia Depot Path set to {JULIA_DEPOT_PATH}" )

def get_JULIA_DEPOT_PATH():
    global JULIA_DEPOT_PATH
    return JULIA_DEPOT_PATH


def load_settings( json_path ):
    
    def _clean_settings_path( p ):
        path = Path(p)
        if path.is_absolute() or str(path).startswith("@"):
            return p 
        else:
            return str( Path( json_path ).parent.joinpath(p).resolve() )
        
    try:
        with open( json_path, "r") as f:
            settings = json.load( f )
        
        if "MORBIT_SYS_IMG" in settings.keys():
            set_MORBIT_SYS_IMG( _clean_settings_path( settings["MORBIT_SYS_IMG"] ) )
        
        if "JULIA_RUNTIME" in settings.keys():
            set_JULIA_RUNTIME( _clean_settings_path( settings["JULIA_RUNTIME"] ) )
        
        if "JULIA_ENV" in settings.keys(): 
            set_JULIA_ENV( _clean_settings_path( settings["JULIA_ENV"] ) )
            
        if "JULIA_DEPOT_PATH" in settings.keys(): 
            set_JULIA_DEPOT_PATH( _clean_settings_path( settings["JULIA_DEPOT_PATH"] ) )
            
    except Exception as e:
        print("Could not open settings file, using defaults.")
        print(e)

def julia_main():
    jl = get_JULIA_MAIN()
    if not jl:
        jl = initialize_julia()
    return jl

def prepare_morbit( jlMain ):
    jlMain.eval("using Pkg;")
    if not get_JULIA_ENV():
        jlMain.eval("new_env = tempname();") 
        set_JULIA_ENV(jlMain.new_env)
    jlMain.eval(f'Pkg.activate( prepare_path( "{get_JULIA_ENV()}" ) )')
    jlMain.eval(f"""
    if !(haskey( Pkg.dependencies(), Base.UUID("{MORBIT_UUID}") ) )
          Pkg.add(; url = "{MORBIT_REPO_URL}" )
    end
    """)

def load_morbit( jlMain ):
    # loading MORBIT module
    try:
        jlMain.using("Pkg");
        if get_JULIA_ENV():
            tprint(f"Activating {get_JULIA_ENV()}.")
            #jlMain.eval(f'pkg"activate {get_JULIA_ENV()}"')
            jlMain.eval(f'Pkg.activate( prepare_path( "{get_JULIA_ENV()}" ) )')
        tprint("Loading Morbit module.")
        jlMain.using("Morbit")            
    except JuliaError as e:
        tprint("Could not load Morbit!!!")
        raise e

def _make_precompile_file( temp_work_dir ):
    pre_script_path = os.path.join(temp_work_dir, "pre_script")
    with open( pre_script_path, "w+" ) as fp:
        #fp.write("using Pkg;\n")
        #fp.write('Pkg.test("Morbit");')
         fp.write("""
         using Pkg;
         using Morbit;
         test_dir = joinpath(pkgdir(Morbit), "test");
         test_load_dir = joinpath(@__DIR__, "test_env", "Project.toml");
         push!(LOAD_PATH, test_load_dir)
         include( joinpath(test_dir, "runtests.jl") );
         """)
    
    return pre_script_path

def base_img_cmd(compiler_env, pre_script, outname, morbit_env):
    compile_script = os.path.join( os.path.dirname(__file__), "compile_morbit.jl" )
    cmd = [ get_JULIA_RUNTIME_NAME(), "--color=yes", compile_script, compiler_env,
           pre_script, outname, MORBIT_REPO_URL, morbit_env ]
    return cmd

def compile_base_img( temp_work_dir ):
    compiler_env = os.path.join( temp_work_dir, "compiler_env" )
    morbit_img_path = os.path.join( temp_work_dir, "_morbit_base.sysimg" )
    morbit_env = get_JULIA_ENV() if get_JULIA_ENV() else "" 
    pre_script = _make_precompile_file( temp_work_dir )
    check_call(install_packagecompiler_cmd(get_JULIA_RUNTIME_NAME(), compiler_env), 
               cwd=temp_work_dir)
    check_call(base_img_cmd(compiler_env, pre_script, morbit_img_path, morbit_env),
               cwd=temp_work_dir)
    return morbit_img_path

def make_sysimage():
    out_path = get_MORBIT_SYS_IMG()
    try:
        with tempfile.TemporaryDirectory() as twd:
            
            base_img = compile_base_img( twd )
            
            compiler_env = os.path.join( twd, "compiler_env" )
            build_sysimage( 
                out_path,
                julia = get_JULIA_RUNTIME_NAME(),
                base_sysimage = base_img,
                compiler_env = compiler_env
                )
    except Exception as e:
        print("Could not generate sysimage.")
        print(e)
    return out_path
    
    
def init_api(sysimage_path):
    # First, check if PyCall is installed &&
    # from julia.api import JuliaInfo
    # from find_libpython import find_libpython()
    # jlinfo = JuliaInfo.load()
    # jlinfo.libpython == find_libpython() 
    #
    # if not successful:
    # from julia.tools import build_pycall
    # build_pycall()
    
    rebuild_pycall = False
    try: 
        jlinfo = JuliaInfo.load(julia = get_JULIA_RUNTIME())
        if not jlinfo.libpython_path == find_libpython():
            rebuild_pycall = True
    except JuliaError:
        rebuild_pycall = True
        
    if rebuild_pycall:
        install()
       
    try:
        Julia( runtime = get_JULIA_RUNTIME(), compiled_modules = False, sysimage = sysimage_path)
    except JuliaError as e:
        print("Could not load Julia.")
        raise e

def initialize_julia(): 
    MORBIT_SYS_IMG = get_MORBIT_SYS_IMG()
    if MORBIT_SYS_IMG:
        if os.path.isfile( MORBIT_SYS_IMG ):
            tprint(f"\tUsing sysimage at {MORBIT_SYS_IMG}")
            sysimage_path = MORBIT_SYS_IMG
        else:
            tprint(f"\tGoing to compile a sysimage at {MORBIT_SYS_IMG}")
            make_sysimage()
            sysimage_path = None
    else:    
        sysimage_path = None
        tprint("\tNo sysimage specified.")
    
    tprint("MOP: Initializing Julia runtime. First time might take a bit.")
       
    init_api(sysimage_path)
    
    from julia import Main

    # include "hack" to make symbols available on the python side
    Main.include( os.path.join( os.path.dirname( __file__ ) , "pycall_sym.jl" ) )
    Main.include( os.path.join( os.path.dirname( __file__ ) , "prepare_path.jl" ) )
    
    prepare_morbit( Main )
    load_morbit( Main )
    
    tprint("Julia runtime all set up!")
    
    set_JULIA_MAIN(Main)
    return Main

                    
                
                
            
