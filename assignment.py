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
import math
import random

def degree(mat,i):
    return sum(mat[i])



#%%read dataset
dataset = pd.read_csv("primary school data.csv")

node1 = np.array(dataset['node1'])
node2 = np.array(dataset['node2'])
time = np.array(dataset['time'])

size = np.max([np.max(node1),np.max(node2)])
print("number of nodes: ",size) #number of nodes

#create link matrix
mat = np.zeros([size,size])
for i in list(range(np.size(node1))):
    mat[node1[i]-1,node2[i]-1] = 1
    mat[node2[i]-1,node1[i]-1] = 1
#0~241 in the matrix, 
M = mat
SIZE = np.size(M[0])

def degree_correlation(mat,degree_array):
    sum_d_squr  = 0
    link_num = 8317
    for i in range(size):
        sum_d_squr = sum_d_squr + degree_array[i]*degree_array[i]
    sum_d_squr 
    u_Di = sum_d_squr / (2*link_num) 
    u_Di_squr = u_Di * u_Di
    
    
    sum_d_trpl = 0
    for i in range(size):
        sum_d_trpl = sum_d_trpl + degree_array[i]*degree_array[i]*degree_array[i]
    E_Di_squr = sum_d_trpl/(2*link_num)
    
        
    Di_Dj = np.dot(np.dot(degree_array,mat),degree_array)
    E_Di_Dj = Di_Dj/(2*link_num)
    
    
    pD = (E_Di_Dj - u_Di_squr )/(E_Di_squr-u_Di_squr)
    return pD





#%% 1.

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
sum_d_squr  = 0
for i in range(size):
    sum_d_squr = sum_d_squr + degree_array[i]*degree_array[i]
sum_d_squr 
u_Di = sum_d_squr / (2*link_num) 
u_Di_squr = u_Di * u_Di


sum_d_trpl = 0
for i in range(size):
    sum_d_trpl = sum_d_trpl + degree_array[i]*degree_array[i]*degree_array[i]
E_Di_squr = sum_d_trpl/(2*link_num)

    
Di_Dj = np.dot(np.dot(degree_array,M),degree_array)
E_Di_Dj = Di_Dj/(2*link_num)


pD = (E_Di_Dj - u_Di_squr )/(E_Di_squr-u_Di_squr)
pD

#%% assortativity is 0.11827144611912918 >0 , physical meaning is 
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
    
clu_array = np.zeros(SIZE)
for i in list(range(SIZE)):
    clu_array[i] = 2 * L(i)/(degree_array[i]*(degree_array[i]-1))

C = sum(clu_array) / SIZE

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


hop_matrix = np.zeros([SIZE,SIZE])
for i in list(range(SIZE)):
    for j in list(range(i,SIZE)):
        hop_matrix[i,j] = hopcount(i,j)
hop_matrix = hop_matrix + hop_matrix.T
HOP = hop_matrix

ave_hop = sum(sum(HOP))/(SIZE*(SIZE-1))
max_hop = np.max(np.max(HOP))
print("average hop:",ave_hop)
print("max hop:",max_hop)
#%% 6. Yes because average hop is 1.7 < log(SIZE)
#%% 7. 
eig_m = np.linalg.eigvals(M)
print("1st eigenvalue", eig[0])

#%% 8.
d = np.diag(degree_array)
laplace_matrix = d - M
eig_l = np.linalg.eigvals(laplace_matrix)
eig_l.sort()
print("second small eig for laplace:",eig_l[1])

#%% 9-15
DF = pd.DataFrame({'n1':node1,'n2':node2,'t':time})
#a = DF[DF.t==1].n1
#a = np.array(a)

def one_step_infect(old_set,timestamp):
    new_set = old_set
    n1 = np.array(DF[DF.t==timestamp].n1)
    n2 = np.array(DF[DF.t==timestamp].n2)
    for i in list(range(np.size(n1))):
        if n1[i] in old_set:
            new_set.add(n2[i])
        if n2[i] in old_set:
            new_set.add(n1[i])
    return new_set
        
def infection(seed,time):
    infect_set = set([seed])
    for i in list(range(time)):
        infect_set = one_step_infect(infect_set,i+1)
    return infect_set
    
#%% 9.

#this part runs about 1 hour and get the num infected by timestamp for each seed

##plot_array = np.zeros([SIZE,5846])
#for j in list(range(229,SIZE)):
#    #print(j)
#    infect_set = set([j+1])
#    for i in list(range(5846)):
#        infect_set = one_step_infect(infect_set,i+1)
#        set_size = len(infect_set)
#        plot_array[j,i] = set_size
#        print("seed:",j+1,"round:",i+1,"size:",set_size)
#np.savetxt('new.csv', plot_array, delimiter = ',')

ave_infected = sum(P)/SIZE
var_infected = np.zeros(5846)
for i in list(range(5846)):
    var_infected[i] = np.std(P[:,i])
    
plt.plot(ave_infected)
plt.plot(var_infected)

#%% 10.
threshold = SIZE * 0.8
reach_80 = np.zeros(SIZE)
for i in list(range(SIZE)):    
    for j in list(range(5846)):
        if P[i,j] >= threshold:
            reach_80[i] = j+1
            break
rank_R = np.argsort(reach_80)

#%% 11,12

#get the rank of several features
rank_D = np.argsort(degree_array)   #degree
rank_C = np.argsort(clu_array)  #cluster coefficient
rank_H = np.argsort(sum(hop_matrix)/(SIZE-1))   #mean hopcount

#1 count activate number for each node
#mat = np.zeros([size,size])
#for i in list(range(np.size(node1))):
#    mat[node1[i]-1,node2[i]-1] = mat[node1[i]-1,node2[i]-1] + 1
#    mat[node2[i]-1,node1[i]-1] = mat[node2[i]-1,node1[i]-1] + 1
#T = mat

freq_array = sum(T)
rank_F = np.argsort(freq_array) #activate frequency through time

step_array = np.array([0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5])
step_array = step_array * SIZE

rate_D = []
rate_C = []
rate_F = []
rate_H = []
for i in list(range(np.size(step_array))):
    sub_c = []
    sub_d = []
    sub_f = []
    sub_h = []
    sub_r = []
    for j in list(range(math.ceil(step_array[i]))):
        sub_c.append(rank_C[j])
        sub_d.append(rank_D[j])
        sub_r.append(rank_R[j])
        sub_f.append(rank_F[j])
        sub_h.append(rank_H[j])
    rate_C.append(len(set(sub_c)&set(sub_r))/len(sub_r))
    rate_D.append(len(set(sub_d)&set(sub_r))/len(sub_r))
    rate_F.append(len(set(sub_f)&set(sub_r))/len(sub_r))
    rate_H.append(len(set(sub_h)&set(sub_r))/len(sub_r))
    
    
plt.plot(step_array/SIZE,rate_C)  
plt.plot(step_array/SIZE,rate_D)  
plt.plot(step_array/SIZE,rate_F)  
plt.plot(step_array/SIZE,rate_H)
    
#%% 12.
#Randomized new matrix
def new_g():
    #get minimal pairs of edges
    l1 = []
    l2 = []
    for i in list(range(SIZE)):
        for j in list(range(i,SIZE)):
            if M[i,j] == 1:
                l1.append(i)
                l2.append(j)
    #return l1,l2
    #shuffle and make sure no self-loop
    b = True
    while b:
        #random.shuffle(l1)
        random.shuffle(l2)
        b = False
        print("...")
        for i in list(range(len(l1))):
            if l1[i]==l2[i]:
                b = True
                break
    
    mat = np.zeros([SIZE,SIZE])
    for i in list(range(len(l1))):
        mat[l1[i],l2[i]] = 1
        mat[l2[i],l1[i]] = 1
        
    degree_array = sum(mat)
    dc = degree_correlation(mat,degree_array)
    
    return mat,dc,degree_array
    #randomize
    
            

#remove duplicate links

