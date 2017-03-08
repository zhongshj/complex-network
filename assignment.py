#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 17:13:19 2017

@author: abk
"""
#%%packages
import numpy as np
import pandas as pd


#%%read dataset
dataset = pd.read_csv("primary school data.csv")

node1 = np.array(dataset['node1'])
node2 = np.array(dataset['node2'])
time = np.array(dataset['time'])

#%%
size = np.max([np.max(node1),np.max(node2)])
mat = np.zeros([size,size])
for i in list(range(np.size(node1))):
    mat[node1[i]][node2[i]] = 1
    mat[node2[i]][node1[i]] = 1