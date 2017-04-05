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
num = np.size(timestamp)

#%% convert timestamp to month, day, weekday, hour, minute

time_formulate = np.zeros([np.size(timestamp),6])
for i in range(np.size(timestamp)):
    time_formulate[i,0] = datetime.fromtimestamp(timestamp[i]-3600*7,pytz.timezone('UTC')).month
    time_formulate[i,1] = datetime.fromtimestamp(timestamp[i]-3600*7,pytz.timezone('UTC')).day
    time_formulate[i,2] = datetime.fromtimestamp(timestamp[i]-3600*7,pytz.timezone('UTC')).weekday()+1
    time_formulate[i,3] = datetime.fromtimestamp(timestamp[i]-3600*7,pytz.timezone('UTC')).hour
    time_formulate[i,4] = datetime.fromtimestamp(timestamp[i]-3600*7,pytz.timezone('UTC')).minute    
    if time_formulate[i,0] >= 4:
        time_formulate[i,5] = time_formulate[i,1]
    if time_formulate[i,0] >= 5:
        time_formulate[i,5] = time_formulate[i,5] + 30
    if time_formulate[i,0] >= 6:
        time_formulate[i,5] = time_formulate[i,5] + 31
    if time_formulate[i,0] >= 7:
        time_formulate[i,5] = time_formulate[i,5] + 30
    if time_formulate[i,0] >= 8:
        time_formulate[i,5] = time_formulate[i,5] + 31
    if time_formulate[i,0] >= 9:
        time_formulate[i,5] = time_formulate[i,5] + 31
    if time_formulate[i,0] >= 10:
        time_formulate[i,5] = time_formulate[i,5] + 30
time_formulate[:,5] = time_formulate[:,5] - min(time_formulate[:,5])
days = time_formulate[:,5]
    #%% create matrix
size = np.max([np.max(sender),np.max(receiver)])
print("number of nodes: ",size) #number of nodes

#create link matrix(obly upper half)
m = np.zeros([size,size])
for i in range(np.size(sender)):
    m[sender[i]-1,receiver[i]-1] = m[sender[i]-1,receiver[i]-1] + 1
    
#%%

#sender
sender_array = sum(m.T)

#receiver
receiver_array = sum(m)

##%% communication
#comm = np.zeros([size,size])
#for i in range(size):
#    for j in range(size):
#        comm[i,j] = min(m[i,j],m.T[i,j])
#        
#%% connectivity

def connectivity(m, l):
    l_2 = l.copy()
    for i in l:
        for j in range(size):
            if m[i,j] != 0 or m[j,i] != 0:
                l_2.add(j)
                print(i,j)
    print(len(l_2))
    return l_2
#%%
l = connectivity(m,l)

#%%
reliable_1 = np.zeros(num)
lifespan = 1


for i in range(num):
    print("check:",i)
    send = sender[i]
    rec = receiver[i]
    day = days[i]
    j = i
    while days[j] <= day + lifespan:
        if rec == sender[j] and send == receiver[j]:
            reliable_1[i] = 1
            print("found")
            break
        else:
            j = j + 1
            if j >= num:
                break
            
#%% reciprocate
day = 195
p0 = np.zeros(day)
p1 = np.zeros(day)
p7 = np.zeros(day)

for i in range(num):
    p0[days[i]] = p0[days[i]] + 1
    if reliable_1[i] == 1:
        p1[days[i]] = p1[days[i]] + 1
    if reliable_7[i] == 1:
        p7[days[i]] = p7[days[i]] + 1
          
#%% plot reciprocate
plt.figure(figsize=(8,4)) 
line_0,=plt.plot(p0,'b-',label="Total")  
line_7,=plt.plot(p7,'r-',label="Replied in 7 days")  
line_1,=plt.plot(p1,'g-',label="Replied in 1 day")  

plt.legend(handles=[line_0, line_7, line_1],loc="upper right")
plt.xlabel("day")
plt.ylabel("mail number")
plt.title("Mailing frequency")
plt.savefig("reciprocate.eps")

#%%
start_day = np.ones(day) * 200
end_day = np.zeros(day)

for i in range(num):
    start_day[days[i]] = min(start_day[days[i]],sender[i])
    end_day[days[i]] = max(end_day[days[i]],sender[i])
   
#%%
plt.scatter(days,sender,s=0.2,marker='.')
plt.xlabel("day")
plt.ylabel("user")
plt.title("Activity")
plt.savefig("activity.eps")
    
    