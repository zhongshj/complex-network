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

def degree(mat,i):
    return sum(mat[i])



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
M = mat
SIZE = np.size(M[0])

link_num = sum(sum(M))/2
print("number of links: ",link_num) #number of links

p = sum(sum(M))/(SIZE*(SIZE-1))
print("link density:", p)   #link density

degree_array = sum(M)

e_d = np.mean(degree_array)
v_d = np.var(degree_array)
print("average degree:", e_d)   #mean, variance
print("degree variance:", v_d)

#%% 2.
plt.hist(degree_array,bins=20)  #degree distribution
#I think ER random graph because the distribution is like poisson

#%% 3.

#%% 4. 

#this function returns a list of all neighbours of given node
def get_neighbour(i):
    l_neighbour = []
    for j in list(range(SIZE)):
        if(M[i,j]==1):
            l_neighbour.append(j)
    return l_neighbour

#this function calculates L()
def L(i):
    count = 0
    a_neighbour = get_neighbour(i)
    for k in list(range(np.size(a_neighbour))):
        for l in list(range(k+1,np.size(a_neighbour))):
            if(M[a_neighbour[k],a_neighbour[l]]==1):
                count = count + 1
    return count
    
C = 0
for i in list(range(SIZE)):
    C = C + 2 * L(i)/(degree_array[i]*(degree_array[i]-1))
C = C / SIZE

print("clustering coefficient: ", C)

#%% 5.

def spread(s_activated):
    l_activated = s_activated
    for i in list(s_activated):
        l_activated |= set(get_neighbour(i))    
    return l_activated
    
def hopcount(start,end):
    s_activated = set([start])
    count = 0
    while end not in s_activated:
        s_activated = spread(s_activated)
        count = count + 1
    return count

#hop_matrix = np.zeros([SIZE,SIZE])
#for i in list(range(SIZE)):
#    for j in list(range(i,SIZE)):
#        hop_matrix[i,j] = hopcount(i,j)
#hop_matrix = hop_matrix + hop_matrix.T
#HOP = hop_matrix

#%%
ave_hop = sum(sum(HOP))/(SIZE*(SIZE-1))
max_hop = np.max(np.max(HOP))
print("average hop:",ave_hop)
print("max hop:",max_hop)
#%%
