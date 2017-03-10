#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 17:13:19 2017

@author: abk
"""
#%%packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt







#%%read dataset
dataset = pd.read_csv("primary school data.csv")

node1 = np.array(dataset['node1'])
node2 = np.array(dataset['node2'])
time = np.array(dataset['time'])

#%% 1.
size = np.max([np.max(node1),np.max(node2)])
print("number of nodes: ",size) #number of nodes

mat = np.zeros([size,size])
for i in list(range(np.size(node1))):
    mat[node1[i]-1,node2[i]-1] = 1
    mat[node2[i]-1,node1[i]-1] = 1
#0~241 in the matrix, 

link_num = sum(sum(mat))/2
print("number of links: ",link_num) #number of links

p = sum(sum(mat))/(size*(size-1))
print("link density:", p)   #link density

degree_array = np.array(sum(mat))
e_d = np.mean(degree_array)
v_d = np.var(degree_array)
print("average degree:", e_d)   #mean, variance
print("degree variance:", v_d)

#%% 2.
plt.hist(degree_array,bins=20)  #degree distribution
#I think ER random graph because the distribution is like poisson

#%% 3.

#%% 4. 
def L(mat,i):
    l_neighbour = []
    count = 0
    for j in list(range(np.size(mat[0]))):
        if(mat[i,j]==1):
            l_neighbour.append(j)
    a_neighbour = np.array(l_neighbour)
    for k in list(range(np.size(a_neighbour))):
        for l in list(range(k,np.size(a_neighbour))):
            if(mat[k,l]==1):
                count = count + 1
    return count
    
C = 0
for i in list(range(size)):
    C = C + 2 * L(mat,i)/(degree_array[i]*(degree_array[i]-1))
C = C / size

print("clustering coefficient: ", C)

#%% 5.
def search_line(mat,i,j):
    l_nei = []
    
