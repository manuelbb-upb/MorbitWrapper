#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 10:20:53 2021

@author: manuelbb
"""

import morbitwrapper as mw
import os.path 
from morbitwrapper.globals import get_MORBIT_SYS_IMG, get_JULIA_RUNTIME_NAME
from morbitwrapper.globals import build_sysimage, compile_base_img 
import tempfile
#%%
mw.load_settings("morbit_settings.json")
out_path = get_MORBIT_SYS_IMG()
#%%
with tempfile.TemporaryDirectory() as twd:
    
    base_img = compile_base_img( twd )
    
    compiler_env = os.path.join( twd, "compiler_env" )
    build_sysimage( 
        out_path,
        julia = get_JULIA_RUNTIME_NAME(),
        base_sysimage = base_img,
        compiler_env = compiler_env
        )