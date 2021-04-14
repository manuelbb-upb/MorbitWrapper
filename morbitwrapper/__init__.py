#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 16:25:44 2020

@author: manuelbb
"""

from .MOPClasses import RbfConfig, MOP, AlgoConfig
from .optimization import optimize

from .globals import set_MORBIT_SYS_IMG, get_MORBIT_SYS_IMG

#NOTE
# DONT make globals accessible from outside the module directly
# … leads only to confusion; use setters and getters instead