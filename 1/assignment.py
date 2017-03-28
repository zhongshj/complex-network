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

def gen_mat(node1,node2):
    mat = np.zeros([size,size])
    for i in list(range(np.size(node1))):
        mat[node1[i]-1,node2[i]-1] = 1
        mat[node2[i]-1,node1[i]-1] = 1
    return mat

#%%read dataset
dataset = pd.read_csv("primary school data.csv")

node1 = np.array(dataset['node1'])
node2 = np.array(dataset['node2'])
time = np.array(dataset['time'])

size = np.max([np.max(node1),np.max(node2)])
print("number of nodes: ",size) #number of nodes

#create link matrix
M = np.zeros([size,size])
for i in list(range(np.size(node1))):
    M[node1[i]-1,node2[i]-1] = 1
    M[node2[i]-1,node1[i]-1] = 1
#0~241 in the matrix, 

SIZE = np.size(M[0])

def degree_correlation(mat):
    degree_array = sum(mat)
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
plt.title("degree distribution")
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
def get_neighbour(i,mat):
    l_neighbour = []
    for j in list(range(SIZE)):
        if(mat[i,j]==1):
            l_neighbour.append(j)
    return l_neighbour

#this function calculates L()
def L(i,mat):
    count = 0
    a_neighbour = get_neighbour(i,mat)
    for k in list(range(np.size(a_neighbour))):
        for l in list(range(k+1,np.size(a_neighbour))):
            if(M[a_neighbour[k],a_neighbour[l]]==1):
                count = count + 1
    return count
def get_clu_array(mat):   
    clu_array = np.zeros(SIZE)
    for i in list(range(SIZE)):
        clu_array[i] = 2 * L(i,mat)/(degree_array[i]*(degree_array[i]-1))
    
    C = sum(clu_array) / SIZE
    print("clustering coefficient: ", C)
    return clu_array

#%% 5.

def spread(s_activated,mat):
    l_activated = s_activated
    for i in list(s_activated):
        l_activated |= set(get_neighbour(i,mat))    
    return l_activated
    
def hopcount(start,end,mat):
    s_activated = set([start])
    count = 0
    while end not in s_activated:
        s_activated = spread(s_activated,mat)
        count = count + 1
    return count

def get_hop(mat):
    hop_matrix = np.zeros([SIZE,SIZE])
    for i in list(range(SIZE)):
        for j in list(range(i,SIZE)):
            hop_matrix[i,j] = hopcount(i,j,mat)
    hop_matrix = hop_matrix + hop_matrix.T
    return hop_matrix
#%%
ave_hop = sum(sum(HOP))/(SIZE*(SIZE-1))
max_hop = np.max(np.max(HOP))
print("average hop:",ave_hop)
print("max hop:",max_hop)
#%% 6. Yes because average hop is 1.7 < log(SIZE)
#%% 7. 
eig_m = np.linalg.eigvals(M)
print("1st eigenvalue", eig_m[0])

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
    
def one_step_infect_improved(old_set,timestamp,dic):
    new_set = old_set
    n1 = np.array(dic[timestamp].n1)
    n2 = np.array(dic[timestamp].n2)
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
    
line_ave,=plt.plot(ave_infected,"b-",label="Average")
line_std,=plt.plot(var_infected,"b--",label="Std")

plt.legend(handles=[line_ave, line_std],loc="right")

#%% 10.
def get_rank_R(p):
    threshold = SIZE * 0.8
    reach_80 = np.zeros(SIZE)
    for i in list(range(SIZE)):    
        for j in list(range(5846)):
            if p[i,j] >= threshold:
                reach_80[i] = j+1
                break
    rank_R = np.argsort(reach_80)
    return rank_R
#%% 11,12

#get the rank of several features
rank_D = np.argsort(degree_array)   #degree
rank_C = np.argsort(clu_array)  #cluster coefficient
rank_H = np.argsort(sum(hop_matrix)/(SIZE-1))   #mean hopcount
#%%
def get_freq_array(node1,node2):
    #1 count activate number for each node
    t_mat = np.zeros([size,size])
    for i in list(range(np.size(node1))):
        t_mat[node1[i]-1,node2[i]-1] = t_mat[node1[i]-1,node2[i]-1] + 1
        t_mat[node2[i]-1,node1[i]-1] = t_mat[node2[i]-1,node1[i]-1] + 1
    freq_array = sum(t_mat)
    return freq_array
#%%
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
    
    
line_c,=plt.plot(step_array/SIZE,rate_C,label="Clustering Coefficient")  
line_d,=plt.plot(step_array/SIZE,rate_D,label="Degree")  
line_f,=plt.plot(step_array/SIZE,rate_F,label="Frequency")  
line_h,=plt.plot(step_array/SIZE,rate_H,label="Average Hopcount")
plt.legend(handles=[line_c, line_d, line_f, line_h],loc="bottom right")
plt.xlabel("f")
plt.ylabel("r")
#%% 14.
#Randomized new matrix
mat = M
save_mat = mat

#%%
#The following functions return randomized matrix and link list
def randomize_increase(mat,node1,node2):
    dc = degree_correlation(mat)
    while dc < 0.4:
        b = False
        while b == False:
            r1 = random.randint(0,SIZE-1)
            r2 = random.randint(0,SIZE-1)
            r3 = random.randint(0,SIZE-1)
            r4 = random.randint(0,SIZE-1)
            #print(r1,r2,r3,r4)
            #change 1-4, 2-3 to 1-2, 3-4
            if degree_array[r1]>degree_array[r2] and degree_array[r2]>degree_array[r3] and degree_array[r3]>degree_array[r4]:
                if mat[r1,r4] == 1 and mat[r2,r3] == 1 and mat[r1,r3] == 0 and mat[r2,r4] == 0 and mat[r1,r2] == 0 and mat[r3,r4] == 0:
                    mat[r1,r2] = 1
                    mat[r3,r4] = 1
                    mat[r1,r4] = 0
                    mat[r2,r3] = 0
                    mat[r2,r1] = 1
                    mat[r4,r3] = 1
                    mat[r4,r1] = 0
                    mat[r3,r2] = 0
                    #print(r1,r2,r3,r4)
                    for i in list(range(np.size(node1))):
                        if node1[i] == r1+1 and node2[i] == r4+1:
                            node2[i] = r2+1
                        if node1[i] == r2+1 and node2[i] == r3+1:
                            node1[i] = r4+1
                        if node1[i] == r4+1 and node2[i] == r1+1:
                            node1[i] = r2+1
                        if node1[i] == r3+1 and node2[i] == r2+1:
                            node2[i] = r4+1
                    b = True
        
        #print("link num:",sum(sum(mat))/2)
        #print("original num:",sum(sum(save_mat)/2))
        dc = degree_correlation(mat)
        print("degree_c",dc)
    return mat,node1,node2
    
def randomize_decrease(mat,node1,node2):
    dc = degree_correlation(mat)
    while dc > -0.3:
        b = False
        while b == False:
            r1 = random.randint(0,SIZE-1)
            r2 = random.randint(0,SIZE-1)
            r3 = random.randint(0,SIZE-1)
            r4 = random.randint(0,SIZE-1)
            #print(r1,r2,r3,r4)
            #change 1-2, 3-4 to 1-4, 2-3
            if degree_array[r1]>degree_array[r2] and degree_array[r2]>degree_array[r3] and degree_array[r3]>degree_array[r4]:
                if mat[r1,r2] == 1 and mat[r3,r4] == 1 and mat[r2,r4] == 0 and mat[r1,r3] == 0 and mat[r1,r4] == 0 and mat[r2,r3] == 0:
                    mat[r1,r4] = 1
                    mat[r3,r2] = 1
                    mat[r1,r2] = 0
                    mat[r3,r4] = 0
                    mat[r4,r1] = 1
                    mat[r2,r3] = 1
                    mat[r2,r1] = 0
                    mat[r4,r3] = 0
                    #print(r1,r2,r3,r4)
                    for i in list(range(np.size(node1))):
                        if node1[i] == r1+1 and node2[i] == r2+1:
                            node2[i] = r4+1
                        if node1[i] == r3+1 and node2[i] == r4+1:
                            node2[i] = r2+1
                        if node1[i] == r2+1 and node2[i] == r1+1:
                            node1[i] = r4+1
                        if node1[i] == r4+1 and node2[i] == r3+1:
                            node1[i] = r2+1
                    b = True
            
        #print("link num:",sum(sum(mat))/2)
        #print("original num:",sum(sum(save_mat)/2))
        dc = degree_correlation(mat)
        print("degree_c",dc)
    return mat, node1, node2
#%% randomize time list
t2 = time.copy()
t3 = time.copy()
t4 = time.copy()
random.shuffle(t2)
random.shuffle(t3)
random.shuffle(t4)

#%% plot
df1 = pd.DataFrame({'n1':node1,'n2':node2,'t':time})
df2 = pd.DataFrame({'n1':g2_1,'n2':g2_2,'t':t2})
df3 = pd.DataFrame({'n1':g3_1,'n2':g3_2,'t':t3})
df4 = pd.DataFrame({'n1':g4_1,'n2':g4_2,'t':t4})

dic1 = {}
dic2 = {}
dic3 = {}
dic4 = {}

for i in range(1,5847):
    temp1 = df1[df1['t']==i]
    temp2 = df2[df2['t']==i]
    temp3 = df3[df3['t']==i]
    temp4 = df4[df4['t']==i]
    dic1[i] = temp1[['n1','n2']]
    dic2[i] = temp2[['n1','n2']]
    dic3[i] = temp3[['n1','n2']]
    dic4[i] = temp4[['n1','n2']]

#%%
plot_array4 = np.zeros([SIZE,5846])
for j in range(SIZE):
    infect_set = set([j+1])
    print(j)
    for i in range(5846):
        infect_set = one_step_infect_improved(infect_set,i+1,dic4)
        set_size = len(infect_set)
        plot_array4[j,i] = set_size
        #print("seed:",j+1,"round:",i+1,"size:",set_size)
 #%%       
ave_plot1 = sum(plot_array1)/SIZE
ave_plot2 = sum(plot_array2)/SIZE
ave_plot3 = sum(plot_array3)/SIZE
ave_plot4 = sum(plot_array4)/SIZE
#%%
plt.plot(ave_plot1)


#%%
line_1,=plt.plot(ave_plot1,"b-",label="Gdata")
line_2,=plt.plot(ave_plot2,"b--",label="G2")
line_3,=plt.plot(ave_plot3,"b-.",label="G3")
line_4,=plt.plot(ave_plot4,"b:",label="G4")
plt.legend(handles=[line_1, line_2, line_3, line_4],loc="right")
plt.xlabel("T")
plt.ylabel("number of infected")

#%%
rank_D = np.argsort(degree_array)   #degree
step_array = np.array([0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5])
step_array = step_array * SIZE

array_C1 = get_clu_array(mat)
array_C2 = get_clu_array(g2)
array_C3 = get_clu_array(g3)
array_C4 = get_clu_array(g4)

print("C complete")

array_F1 = get_freq_array(node1,node2)
array_F2 = get_freq_array(g2_1,g2_2)
array_F3 = get_freq_array(g3_1,g3_2)
array_F4 = get_freq_array(g4_1,g4_2)

print("F complete")

array_H1 = sum(get_hop(mat))
print("H1 complete")
array_H2 = sum(get_hop(g2))
print("H2 complete")
array_H3 = sum(get_hop(g3))
print("H3 complete")
array_H4 = sum(get_hop(g4))
print("H4 complete")

#%%
rank_R1 = get_rank_R(plot_array1)
rank_R2 = get_rank_R(plot_array2)
rank_R3 = get_rank_R(plot_array3)
rank_R4 = get_rank_R(plot_array4)
#%%
rank_C1 = np.argsort(array_C1)
rank_C2 = np.argsort(array_C2)
rank_C3 = np.argsort(array_C3)
rank_C4 = np.argsort(array_C4)

rank_F1 = np.argsort(array_F1)
rank_F2 = np.argsort(array_F2)
rank_F3 = np.argsort(array_F3)
rank_F4 = np.argsort(array_F4)

rank_H1 = np.argsort(array_H1)
rank_H2 = np.argsort(array_H2)
rank_H3 = np.argsort(array_H3)
rank_H4 = np.argsort(array_H4)
#%%

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
        sub_c.append(rank_C4[j])
        sub_d.append(rank_D[j])
        sub_r.append(rank_R4[j])
        sub_f.append(rank_F4[j])
        sub_h.append(rank_H4[j])
    rate_C.append(len(set(sub_c)&set(sub_r))/len(sub_r))
    rate_D.append(len(set(sub_d)&set(sub_r))/len(sub_r))
    rate_F.append(len(set(sub_f)&set(sub_r))/len(sub_r))
    rate_H.append(len(set(sub_h)&set(sub_r))/len(sub_r))
    
    
line_c,=plt.plot(step_array/SIZE,rate_C,'b-',label="Clustering Coefficient")  
line_d,=plt.plot(step_array/SIZE,rate_D,'b--',label="Degree")  
line_f,=plt.plot(step_array/SIZE,rate_F,'b-.',label="Frequency")  
line_h,=plt.plot(step_array/SIZE,rate_H,'b:',label="Average Hopcount")
plt.legend(handles=[line_c, line_d, line_f, line_h],loc="bottom right")
plt.xlabel("f")
plt.ylabel("r")
plt.title("G4")