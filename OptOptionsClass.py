#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 16:09:00 2020

@author: manuelbb
"""

from .runtime_setup import julia_main

class OptOptions():
    
    def __init__(self, **kwargs) :
        julia_main()
        return