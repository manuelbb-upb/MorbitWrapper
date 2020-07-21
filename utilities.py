#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 16:16:13 2020

@author: manuelbb
"""

import time
from .globals import CONFIG_ARGS

TIME_FORMAT = "%H:%M:%S"

def tprint( *args, **kwargs ):
    print( time.strftime(TIME_FORMAT), end=" -- " )
    print( *args, *kwargs )
    
def clean_args( arg_dict ):
    return_dict = {}
    for k,v in arg_dict.items():
        if k in CONFIG_ARGS.keys():
            return_dict[k] = v      # TODO maybe also check datatypes here, would need a conversion map etc.
        elif k == "Δ_0":
            return_dict["Δ₀"] = v
        else:
            print(f"Parameter {k} not supported.")
    return return_dict

    
    