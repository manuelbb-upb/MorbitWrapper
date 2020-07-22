#from .globals import JULIA_RUNTIME, PROJECT_FILES_PATH, MORBIT_PATH, MORBIT_SYS_IMG
from .globals import WRAPPER_PATH

from time import strftime
import os
from random import random
from julia.tools import julia_py_executable

from .globals import  WRAPPER_PATH
#from .runtime_setup import patch_environment
from subprocess import check_call, CalledProcessError

# def get_api_obj():
#     api = LibJulia.load()
#     api.init_julia( ["--compiled-modules=no", "--startup=no"] )
#     return api

def create_sysimg(outpath = None):
    
    if not outpath:
        print(f"Output path set to {os.getcwd()}.")
        outpath = os.getcwd()
        out_filename = strftime("%d_%m_%Y_%H_%M_%S") + "_sysimg.so"
    else:
        pth, fn = os.path.split( outpath )
        if len(fn) == 0:
            out_filename = strftime("%d_%m_%Y_%H_%M_%S") + "_sysimg.so"
        else:
            out_filename = fn
        outpath = pth        
    
    start_dir = os.getcwd()
    os.chdir(outpath)
    
    # check before if current folder is writable (else the whole compilation could be for naught)
    try: 
        fname = f"tmpfile_{random()*1e5:5.0f}"
        test_file = open(fname, "w")
        test_file.close()
        os.remove(fname)
    except:
        raise Exception(f"Current dir '{outpath}' is not writable.")
        
    # PATCH environment
    #patch_environment() DOES NOT WORK, ENV VAR MUST BE SET OUTSIDE AND BEFOREHAND IN SHELL
    
    # START JULIA SIMILAR TO julia-py as provided by the pip-module 
    #module_path = os.path.dirname( inspect.getfile(LibJulia) )
    #julia_patch = os.path.join( module_path, "patch.jl")
    
    #os.environ["_PYJULIA_JULIA"] = "julia"
    #os.environ["_PYJULIA_JULIA_PY"] = "/usr/bin/julia-py"       # in the container this is ensured by 'ln -s /usr/local/bin/julia-py /usr/bin' 
    #os.environ["_PYJULIA_PATCH_JL"] = julia_patch
    
    #print("Starting a seperate Julia runtime.")
    #api = get_api_obj()
    
    #def eval_and_check( eval_string ):
     #   if not isinstance( eval_string, bytes ):
      #      eval_string = eval_string.encode("utf-8")
        
      #  code = api.jl_eval_string( eval_string )
       # if not code:
        #    raise Exception(f"Julia code did not execute:\n{eval_string}\n")
    
    julia_cmd = julia_py_executable()
    #patch_script_path = os.path.join(WRAPPER_PATH, "patch_openssl_jll.jl")    
    compile_script_path = os.path.join(WRAPPER_PATH, "compile.jl")   

    # 1) Get path to append to LD_LIBRARY_PATH
    #compile_env = patch_environment()
    # 2) Do the actual compiling
    try: 
        check_call([julia_cmd, compile_script_path, out_filename])
    except CalledProcessError as e:
        print("Could not compile the sysimage.")
        print(e)
            
    #for s in statements:
     #   eval_and_check(s)
        
    #api.jl_atexit_hook(0)

    os.chdir(start_dir)
    return os.path.join( outpath, out_filename )