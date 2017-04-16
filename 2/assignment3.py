#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 15:12:52 2017

@author: abk
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from datetime import datetime
import pytz

#%%

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
day = 195

#%%

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

#%%
#==============================================================================
# immutation
#==============================================================================

#%% temporal model
dic = {}
df = pd.DataFrame({'send':sender,'receive':receiver,'day':days})

for i in range(num):
    dic[i] = np.array([np.array(df[df['day']==i].send),np.array(df[df['day']==i].receive)]).T

#%%

def one_step_infect(old_set,day,dic,imu_set):
    new_set = old_set
    a = dic[day]
    for i in range(np.size(a,0)):
        if a[i,0] in old_set and a[i,0] not in imu_set:
            new_set.add(a[i,1])
    return new_set

def infection(seed,dic,start_day,imu_set):
    infect_set = set([seed])
    li = []
    
    # no infection before start_day
    for i in range(start_day):
        li.append(0)
        
    # start infection
    for i in range(start_day,195):
        infect_set = one_step_infect(infect_set,i,dic,imu_set)
        li.append(len(infect_set))
    return li
#%%

empty_t = []
early_t = []
ra_t = []
sa_t = []
rd_t = []
sd_t = []
rf_t = []
sf_t = []

time_index = np.arange(10,200,10)


for start_day in time_index:

    # select start_day
    
    imu_num = 100
    # get available user set
    user_set = set()
    for i in range(start_day):
        for j in dic[i]:
            user_set.add(j[0])
            user_set.add(j[1])
    
    user_num = len(user_set)
    user_set = list(user_set)
    print("start day:", start_day, "user num:", user_num)
    
    # empty set and early register set
    imu_empty = set()
    imu_early = set(np.arange(imu_num))
    
    # get 50 frequent user set
    send_num = np.zeros(user_num)
    receive_num = np.zeros(user_num)
    
    for i in range(start_day):
        for j in dic[i]:
            send_num[j[0]-1] = send_num[j[0]-1] + 1
            receive_num[j[1]-1] = receive_num[j[1]-1] + 1
    
    send_num_sort = np.argsort(send_num)
    receive_num_sort = np.argsort(receive_num)
    send_num_top = np.zeros(imu_num)
    receive_num_top = np.zeros(imu_num)
    for i in range(imu_num):
        receive_num_top[i] = receive_num_sort[-i-1]
        send_num_top[i] = send_num_sort[-i-1]  
                          
    imu_freq_r = set(receive_num_top)
    imu_freq_s = set(send_num_top)
    
    print("frequent user found")
    
    # get 50 large degree user
    m_1 = np.zeros([user_num,user_num])
    for i in range(start_day):
        for j in dic[i]:
            m_1[j[0]-1,j[1]-1] = 1
               
    send_degree = sum(m_1.T)
    receive_degree = sum(m_1)
    
    send_degree_sort = np.argsort(send_degree)
    receive_degree_sort = np.argsort(receive_degree)
    send_degree_top = np.zeros(imu_num)
    receive_degree_top = np.zeros(imu_num)
    for i in range(imu_num):
        receive_degree_top[i] = receive_degree_sort[-i-1]
        send_degree_top[i] = send_degree_sort[-i-1]  
                          
    imu_degree_r = set(receive_degree_top)
    imu_degree_s = set(send_degree_top)
    
    print("large degree user found")
    
    #
    send_active = np.zeros([user_num,start_day])
    receive_active = np.zeros([user_num,start_day])
    for i in range(start_day):
        for j in dic[i]:
            send_active[j[0]-1,i] = 1
            receive_active[j[1]-1,i] = 1
                          
    send_active = sum(send_active.T)
    receive_active = sum(receive_active.T)
    
    send_active_sort = np.argsort(send_active)
    receive_active_sort = np.argsort(receive_active)
    send_active_top = np.zeros(imu_num)
    receive_active_top = np.zeros(imu_num)
    for i in range(imu_num):
        receive_active_top[i] = receive_active_sort[-i-1]
        send_active_top[i] = send_active_sort[-i-1]
        
    imu_active_r = set(receive_active_top)
    imu_active_s = set(send_active_top)
    
    print("active user found")
    
    #
    
    #==============================================================================
    # start simulation
    #==============================================================================
    
    
    # start simulation
    spread_empty = np.zeros([user_num, day])
    spread_early = np.zeros([user_num, day])
    spread_rdegree = np.zeros([user_num, day])
    spread_sdegree = np.zeros([user_num, day])
    spread_ractive = np.zeros([user_num, day])
    spread_sactive = np.zeros([user_num, day])
    spread_rfreq = np.zeros([user_num, day])
    spread_sfreq = np.zeros([user_num, day])
    for i in range(user_num):
        spread_empty[i] = np.array(infection(user_set[i],dic,start_day,imu_empty))
        spread_early[i] = np.array(infection(user_set[i],dic,start_day,imu_early))
        spread_rdegree[i] = np.array(infection(user_set[i],dic,start_day,imu_degree_r))
        spread_sdegree[i] = np.array(infection(user_set[i],dic,start_day,imu_degree_s))
        spread_ractive[i] = np.array(infection(user_set[i],dic,start_day,imu_active_r))
        spread_sactive[i] = np.array(infection(user_set[i],dic,start_day,imu_active_s))
        spread_rfreq[i] = np.array(infection(user_set[i],dic,start_day,imu_freq_r))
        spread_sfreq[i] = np.array(infection(user_set[i],dic,start_day,imu_freq_s))
    
        print("seed:",user_set[i])
    
    
    
    early_t.append((sum(sum(spread_empty)-sum(spread_early))/user_num)/(194-start_day))
    ra_t.append((sum(sum(spread_empty)-sum(spread_rdegree))/user_num)/(194-start_day))
    sa_t.append((sum(sum(spread_empty)-sum(spread_sdegree))/user_num)/(194-start_day))
    rd_t.append((sum(sum(spread_empty)-sum(spread_ractive))/user_num)/(194-start_day))
    sd_t.append((sum(sum(spread_empty)-sum(spread_sactive))/user_num)/(194-start_day))
    rf_t.append((sum(sum(spread_empty)-sum(spread_rfreq))/user_num)/(194-start_day))
    sf_t.append((sum(sum(spread_empty)-sum(spread_sfreq))/user_num)/(194-start_day))

#%%
start_plot = 1
end_plot = 19

l_early,=plt.plot(time_index[start_plot:end_plot],early_t[start_plot:end_plot],label="early user")
l_ra,=plt.plot(time_index[start_plot:end_plot],ra_t[start_plot:end_plot],label="receive activity")
l_sa,=plt.plot(time_index[start_plot:end_plot],sa_t[start_plot:end_plot],label="send activity")
l_rd,=plt.plot(time_index[start_plot:end_plot],rd_t[start_plot:end_plot],label="receive degree")
l_sd,=plt.plot(time_index[start_plot:end_plot],sd_t[start_plot:end_plot],label="send degree")
l_rf,=plt.plot(time_index[start_plot:end_plot],rf_t[start_plot:end_plot],label="receive frequency")
l_sf,=plt.plot(time_index[start_plot:end_plot],sf_t[start_plot:end_plot],label="send frequency")

plt.legend(handles=[l_early, l_ra, l_sa, l_rd, l_sd, l_rf, l_sf],loc="bottom left")
plt.xlabel("infection start day")
plt.ylabel("average protected user")
#plt.yscale('log')
plt.title("Infection reduction by immutation methods")
plt.savefig("imu.eps")

#%%

line_empty,=plt.plot(sum(spread_empty)/user_num,label="no immutation")
line_early,=plt.plot(sum(spread_early)/user_num,label="early user")
line_ra,=plt.plot(sum(spread_ractive)/user_num,label="receive activity")
line_sa,=plt.plot(sum(spread_sactive)/user_num,label="send activity")
line_rd,=plt.plot(sum(spread_rdegree)/user_num,label="receive degree")
line_sd,=plt.plot(sum(spread_sdegree)/user_num,label="send degree")
line_rf,=plt.plot(sum(spread_rfreq)/user_num,label="receive frequency")
line_sf,=plt.plot(sum(spread_sfreq)/user_num,label="send frequency")
plt.legend(handles=[line_empty, line_early, line_ra, line_sa, line_rd, line_sd, line_rf, line_sf],loc="bottom right")
plt.xlabel("day")
plt.ylabel("average infect")
plt.title("Infection by different immutation from day 125")
plt.savefig("imu125.eps")

#%%