#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 09:56:27 2017

@author: abk
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas


#%% read data

data = []

for l in open("CollegeMsg.txt"):
    row = [int(x) for x in l.split()]
    if len(row) > 0:
        data.append(row)

data = np.array(data)
sender = data.T[0]
receiver = data.T[1]
timestamp = data.T[2]
#%%
