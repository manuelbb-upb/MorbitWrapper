#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 16:25:44 2020

@author: manuelbb
"""

from .MOPClasses import MOP, AlgoConfig
from .sysimg import create_sysimg 

from .globals import path_info, set_MORBIT_PATH, set_MORBIT_SYS_IMG, get_MORBIT_PATH, get_MORBIT_SYS_IMG

#NOTE
# DONT make globals accessible from outside the module directly
# … leads only to confusion; use setters and getters instead