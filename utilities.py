#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 16:16:13 2020

@author: manuelbb
"""
from uuid import uuid4
import time
from .globals import CONFIG_ARGS, julia_main

TIME_FORMAT = "%H:%M:%S"

def tprint( *args, **kwargs ):
    print( time.strftime(TIME_FORMAT), end=" -- " )
    print( *args, *kwargs )
    
def clean_args( arg_dict ):
    jl = julia_main()
    py_dict = {}
    
    jl.eval( "conf = AlgoConfig()" )
    
    for k,v in arg_dict.items():
        if k in CONFIG_ARGS.keys():
            if k in ["rbf_kernel", "descent_method", "sampling_algorithm"]:
                #jl.eval(f'args[:{k}] = Symbol("{v}")')
                jl.eval(f'conf.{k} = Symbol("{v}")')
            else:
                #py_dict[k] = v
                jl.eval(f'x -> setfield!(conf,:{k}, x)')(v)
                print(jl.eval(f"conf.{k}"))
            # TODO maybe also check datatypes here, would need a conversion map etc.
        elif k == "Δ_0":
            py_dict["Δ₀"] = v
        else:
            print(f"Parameter {k} not supported.")
    
    conf = jl.eval("conf")
    #conf["max_iter"]
    return conf

    
    