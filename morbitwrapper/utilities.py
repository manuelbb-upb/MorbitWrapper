#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 16:16:13 2020

@author: manuelbb
"""

import time

#from .globals import julia_main

TIME_FORMAT = "%H:%M:%S"

def tprint( *args, **kwargs ):
    print( time.strftime(TIME_FORMAT), end=" -- " )
    print( *args, *kwargs )
