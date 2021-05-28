#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 16:25:44 2020

@author: manuelbb
"""

from .MOPClasses import RbfConfig, MOP, AlgoConfig, LagrangeConfig, TaylorConfig, ExactConfig
from .optimization import optimize, initialize_data, iterate, get_ret_values
from .optimization import print_fin_info, print_stop_info

from .globals import load_settings, make_sysimage

from .globals import set_MORBIT_SYS_IMG, get_MORBIT_SYS_IMG, set_JULIA_ENV
from .globals import get_JULIA_ENV, set_JULIA_RUNTIME, get_JULIA_RUNTIME
from .globals import get_JULIA_DEPOT_PATH, set_JULIA_DEPOT_PATH
from .globals import julia_main

#NOTE
# DONT make globals accessible from outside the module directly
# … leads only to confusion; use setters and getters instead