#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 09:56:27 2017

@author: abk
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from datetime import datetime
import pytz

def degree(mat,i):
    return sum(mat[i])

def gen_mat(node1,node2):
    mat = np.zeros([size,size])
    for i in list(range(np.size(node1))):
        mat[node1[i]-1,node2[i]-1] = 1
        mat[node2[i]-1,node1[i]-1] = 1
    return mat
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

#%% convert timestamp to month, day, weekday, hour, minute

time_formulate = np.zeros([np.size(timestamp),5])
for i in range(np.size(timestamp)):
    time_formulate[i,0] = datetime.fromtimestamp(timestamp[i]-3600*7,pytz.timezone('UTC')).month
    time_formulate[i,1] = datetime.fromtimestamp(timestamp[i]-3600*7,pytz.timezone('UTC')).day
    time_formulate[i,2] = datetime.fromtimestamp(timestamp[i]-3600*7,pytz.timezone('UTC')).weekday()+1
    time_formulate[i,3] = datetime.fromtimestamp(timestamp[i]-3600*7,pytz.timezone('UTC')).hour
    time_formulate[i,4] = datetime.fromtimestamp(timestamp[i]-3600*7,pytz.timezone('UTC')).minute    
#%% create matrix
size = np.max([np.max(sender),np.max(receiver)])
print("number of nodes: ",size) #number of nodes

#create link matrix
m = np.zeros([size,size])
m_t = np.zeros([size,size])
for i in range(np.size(sender)):
    m[sender[i]-1,receiver[i]-1] = m[sender[i]-1,receiver[i]-1] + 1

    