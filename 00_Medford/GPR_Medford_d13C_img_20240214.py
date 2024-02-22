import os
import os.path
from os import path
import sys
import datetime
import re
import shutil
import time
from time import gmtime, strftime
import csv
import numpy
from numpy.fft import fft, ifft, ifftshift
import matplotlib
import matplotlib.pyplot as plt
import loess2D
import scipy
from scipy import stats
from scipy import interpolate

import glob

print (str(__file__))

print (time.strftime("%H:%M:%S"))

t1 = float(time.time())

backslash = '\\'

wkspc = str(os.getcwd()).replace(backslash,"/") + "/"
wkspc_data = wkspc + '01_Data/'

gpsdata = glob.glob(wkspc_data + "*.csv")

for n in range(0,len(gpsdata)):
	SL_table = gpsdata[n]
	SL_data = numpy.genfromtxt(SL_table,delimiter=',')
		
	w1 = numpy.where(numpy.isnan(SL_data[:,1]))[0]
	SL_data = numpy.delete(SL_data,w1,axis=0)
	
	SL_data_2 = numpy.zeros((len(SL_data),5))
	SL_data_2[:,0] = SL_data[:,0]
	SL_data_2[:,1] = SL_data[:,1]
	SL_data_2[:,2] = SL_data[:,2]
	SL_data_2[:,3] = float(SL_table.split("\\")[1][4:6])
	SL_data_2[:,4] = float(SL_table.split("\\")[1][8])
	
	if float(SL_table.split("\\")[1][4:6]) == 0.:

		SL_data_2[:,0] = SL_data_2[:,0] * 1.0 # - 1172.2
		w1 = numpy.where((SL_data_2[:,0]>1138.6) & (SL_data_2[:,0]<1184.8))[0]	

	if float(SL_table.split("\\")[1][4:6]) == 1.:

		SL_data_2[:,0] = SL_data_2[:,0] *1.0 #- 898.17
		w1 = numpy.where((SL_data_2[:,0]>846.7) & (SL_data_2[:,0]<910.75))[0]	
		
	if float(SL_table.split("\\")[1][4:6]) == 2.:

		SL_data_2[:,0] = SL_data_2[:,0] * 1.0# - 562.9
		w1 = numpy.where((SL_data_2[:,0]>521.86) & (SL_data_2[:,0]<570.0))[0]	
		
	if float(SL_table.split("\\")[1][4:6]) == 3.:

		SL_data_2[:,0] = SL_data_2[:,0] * 1.0#+ 365.7
		w1 = numpy.where((SL_data_2[:,0]>317.441) & (SL_data_2[:,0]<378.23))[0]	
		
	if float(SL_table.split("\\")[1][4:6]) == 4.:
		
		SL_data_2[:,0] = SL_data_2[:,0] *1.0 # - 56.2
		w1 = numpy.where((SL_data_2[:,0]>48.07) & (SL_data_2[:,0]<68.9))[0]	
		
	if float(SL_table.split("\\")[1][5]) == 5.:
		
		SL_data_2[:,0] = SL_data_2[:,0] #- 56.2
		w1 = numpy.where((SL_data_2[:,0]>=51.02) & (SL_data_2[:,0]<=60.98))[0]			
		
	if float(SL_table.split("\\")[1][4:6]) >= 6.:
		w1 = numpy.where((SL_data_2[:,0]>=-100000000) & (SL_data_2[:,0]<=100000000.))[0]			
		
	if n == 0:
		SL_data_combined = SL_data_2[w1,:]	
	
	if n > 0:
		SL_data_combined = numpy.append(SL_data_combined,SL_data_2[w1,:],axis=0)


sites = numpy.unique(SL_data_combined[:,3])
site_matrix_depth = numpy.zeros((5,4))

site_locations = []

for n in range(0,5):
	if n == 0.:
		w1 = numpy.where((SL_data_combined[:,3] == n))[0]
		site_locations.append(w1,)
		
		site_matrix_depth[n,:] = 1138.6,1158.2,1172.2,1184.8
		
		recovery_BR = site_matrix_depth[n,0]
		core_BR = site_matrix_depth[n,1]
		base_BR = site_matrix_depth[n,3]
		CIE_BR = site_matrix_depth[n,2]
		high_BR = CIE_BR - 51.5
		low_BR = CIE_BR + 12.64				
		
	if n == 1.:
		w1 = numpy.where((SL_data_combined[:,3] == n))[0]
		site_locations.append(w1,)
		
		site_matrix_depth[n,:] = 846.7,865.1,898.2,910.75
		
		recovery_MI = site_matrix_depth[n,0]
		core_MI = site_matrix_depth[n,1]
		base_MI = site_matrix_depth[n,3]
		CIE_MI = site_matrix_depth[n,2]
		high_MI = CIE_MI - 51.5
		low_MI = CIE_MI + 12.64				
		
	if n == 2.:
		w1 = numpy.where((SL_data_combined[:,3] == n))[0]
		site_locations.append(w1,)
		
		site_matrix_depth[n,:] = 521.86,542.64,562.1,574.74
		
		recovery_AN = site_matrix_depth[n,0]
		core_AN = site_matrix_depth[n,1]
		base_AN = site_matrix_depth[n,3]
		CIE_AN = site_matrix_depth[n,2]
		high_AN = CIE_AN - 51.5
		low_AN = CIE_AN + 12.64		
		
	if n == 3.:
		w1 = numpy.where((SL_data_combined[:,3] == n))[0]
		site_locations.append(w1,)
		
		site_matrix_depth[n,:] = 317.441,331.42,365.7,378.23
		
		recovery_WL = site_matrix_depth[n,0]
		core_WL = site_matrix_depth[n,1]
		base_WL = site_matrix_depth[n,3]
		CIE_WL = site_matrix_depth[n,2]
		high_WL = CIE_WL - 51.5
		low_WL = CIE_WL + 12.64		
		
	if n == 4.:
		w1 = numpy.where((SL_data_combined[:,3] == n))[0]
		site_locations.append(w1,)

		sed_rate1 = 0.7
		depth_mod = 73.2916667 * sed_rate1
		depth_mod2 = (153.28125-73.2916667) * sed_rate1
		site_matrix_depth[n,:] = (56.180879409999996-depth_mod) - depth_mod2,56.180879409999996-depth_mod,56.180879409999996,69.302
		
		recovery_MAP3A = site_matrix_depth[n,0]
		core_MAP3A = site_matrix_depth[n,1]
		base_MAP3A = site_matrix_depth[n,3]
		CIE_MAP3A = 56.180879409999996
		high_MAP3A = CIE_MAP3A - 51.5
		low_MAP3A = CIE_MAP3A + 12.64

site_matrix_age = numpy.zeros((5,4))

CIE_time = 0.0

recovery_time = 153.28125
recovery_time_err = 0.0
body_time = 73.2916667
body_time_err = 0.0
CIE_time_2 = 0.0
CIE_time_2_err = 0.00
onset_time = -192.357481
onset_time_err = 0.0

w_d13C_BR_c = numpy.where((SL_data_combined[:,3]==0.0))[0]
w_d13C_MI_c = numpy.where((SL_data_combined[:,3]==1.0))[0]
w_d13C_AN_c = numpy.where((SL_data_combined[:,3]==2.0) )[0]
w_d13C_WL_c = numpy.where((SL_data_combined[:,3]==3.0))[0]
w_d13C_MAP3A_c = numpy.where((SL_data_combined[:,3]==4.0))[0]

mean_val =  0.0

w_d13C_bulk_MAP3A_c = numpy.where((SL_data_combined[:,3]==4.0) & (SL_data_combined[:,4]==0.0))[0]
w_d13C_org_MAP3A_c = numpy.where((SL_data_combined[:,3]==4.0) & (SL_data_combined[:,4]==1.0))[0]
w_TEX86_MAP3A_c = numpy.where((SL_data_combined[:,3]==4.0) & (SL_data_combined[:,4]==2.0))[0]
w_d13C_PF_MAP3A_c = numpy.where((SL_data_combined[:,3]==4.0) & (SL_data_combined[:,4]==3.0))[0]
w_d13C_BF_MAP3A_c = numpy.where((SL_data_combined[:,3]==4.0) & (SL_data_combined[:,4]==4.0))[0]

w_d13C_bulk_MAP3B_c = numpy.where((SL_data_combined[:,3]==5.0) & (SL_data_combined[:,4]==0.0))[0]
w_d13C_org_MAP3B_c = numpy.where((SL_data_combined[:,3]==5.0) & (SL_data_combined[:,4]==1.0))[0]
w_TEX86_MAP3B_c = numpy.where((SL_data_combined[:,3]==5.0) & (SL_data_combined[:,4]==2.0))[0]
w_d13C_PF_MAP3B_c = numpy.where((SL_data_combined[:,3]==5.0) & (SL_data_combined[:,4]==3.0))[0]
w_d13C_BF_MAP3B_c = numpy.where((SL_data_combined[:,3]==5.0) & (SL_data_combined[:,4]==4.0))[0]

w_d13C_bulk_c = numpy.where((SL_data_combined[:,4]==0.0))[0]
w_d13C_org_c = numpy.where((SL_data_combined[:,4]==1.0))[0]
w_d13C_PF_c = numpy.where((SL_data_combined[:,4]==3.0))[0]
w_d13C_BF_c = numpy.where((SL_data_combined[:,4]==4.0))[0]
w_TEX86_c = numpy.where((SL_data_combined[:,4]==2.0))[0]


#####

MCMC_iterations = 2000

per_year = 0.1

ice_end = 60.

t_vals = numpy.arange(51.,ice_end+ per_year,per_year)

s_vals1 = (t_vals * 0.0) + 4.0

s_vals1_tmp =  (t_vals * 0.0) + 4.
s_vals1 = numpy.append(s_vals1,s_vals1_tmp)

s_vals1_tmp =  (t_vals * 0.0) + 4.
s_vals1 = numpy.append(s_vals1,s_vals1_tmp)

s_vals1_tmp =  (t_vals * 0.0) + 4.
s_vals1 = numpy.append(s_vals1,s_vals1_tmp)

s_vals1_tmp =  (t_vals * 0.0) + 4.
s_vals1 = numpy.append(s_vals1,s_vals1_tmp)

##

s_vals1_tmp =  (t_vals * 0.0) + 5.
s_vals1 = numpy.append(s_vals1,s_vals1_tmp)

s_vals1_tmp =  (t_vals * 0.0) + 5.
s_vals1 = numpy.append(s_vals1,s_vals1_tmp)

s_vals1_tmp =  (t_vals * 0.0) + 5.
s_vals1 = numpy.append(s_vals1,s_vals1_tmp)

s_vals1_tmp =  (t_vals * 0.0) + 5.
s_vals1 = numpy.append(s_vals1,s_vals1_tmp)

s_vals1_tmp =  (t_vals * 0.0) + 5.
s_vals1 = numpy.append(s_vals1,s_vals1_tmp)

s_vals1_tmp =  (t_vals * 0.0) + 10.
s_vals1 = numpy.append(s_vals1,s_vals1_tmp)

####

s_vals2 = (t_vals * 0.0)

s_vals2_tmp =  (t_vals * 0.0) + 1.
s_vals2 = numpy.append(s_vals2,s_vals2_tmp)

s_vals2_tmp =  (t_vals * 0.0) + 2.
s_vals2 = numpy.append(s_vals2,s_vals2_tmp)

s_vals2_tmp =  (t_vals * 0.0) + 3.
s_vals2 = numpy.append(s_vals2,s_vals2_tmp)

s_vals2_tmp =  (t_vals * 0.0) + 4.
s_vals2 = numpy.append(s_vals2,s_vals2_tmp)

##

s_vals2_tmp =  (t_vals * 0.0) + 0.
s_vals2 = numpy.append(s_vals2,s_vals2_tmp)

s_vals2_tmp =  (t_vals * 0.0) + 1.
s_vals2 = numpy.append(s_vals2,s_vals2_tmp)

s_vals2_tmp =  (t_vals * 0.0) + 2.
s_vals2 = numpy.append(s_vals2,s_vals2_tmp)

s_vals2_tmp =  (t_vals * 0.0) + 3.
s_vals2 = numpy.append(s_vals2,s_vals2_tmp)

s_vals2_tmp =  (t_vals * 0.0) + 4.
s_vals2 = numpy.append(s_vals2,s_vals2_tmp)

s_vals2_tmp =  (t_vals * 0.0) + 10.
s_vals2 = numpy.append(s_vals2,s_vals2_tmp)

##

t_vals_tmp =  t_vals * 1.0
t_vals = numpy.append(t_vals,t_vals_tmp)
t_vals = numpy.append(t_vals,t_vals_tmp)
t_vals = numpy.append(t_vals,t_vals_tmp)
t_vals = numpy.append(t_vals,t_vals_tmp)

t_vals = numpy.append(t_vals,t_vals_tmp)
t_vals = numpy.append(t_vals,t_vals_tmp)
t_vals = numpy.append(t_vals,t_vals_tmp)
t_vals = numpy.append(t_vals,t_vals_tmp)
t_vals = numpy.append(t_vals,t_vals_tmp)

t_vals = numpy.append(t_vals,t_vals_tmp)

w_d13C_bulk_MAP3A = numpy.where((s_vals1==4.0) & (s_vals2==0.0))[0]

w_d13C_org_MAP3A = numpy.where((s_vals1==4.0) & (s_vals2==1.0))[0]
w_TEX86_MAP3A = numpy.where((s_vals1==4.0) & (s_vals2==2.0))[0]
w_d13C_PF_MAP3A = numpy.where((s_vals1==4.0) & (s_vals2==3.0))[0]
w_d13C_BF_MAP3A = numpy.where((s_vals1==4.0) & (s_vals2==4.0))[0]

w_d13C_bulk_MAP3B = numpy.where((s_vals1==5.0) & (s_vals2==0.0))[0]
w_d13C_org_MAP3B = numpy.where((s_vals1==5.0) & (s_vals2==1.0))[0]
w_TEX86_MAP3B = numpy.where((s_vals1==5.0) & (s_vals2==2.0))[0]
w_d13C_PF_MAP3B = numpy.where((s_vals1==5.0) & (s_vals2==3.0))[0]
w_d13C_BF_MAP3B = numpy.where((s_vals1==5.0) & (s_vals2==4.0))[0]

#####

count = 0
output_matrix = wkspc + 'MC_probability_output_1.csv'	
output_matrix = numpy.genfromtxt(output_matrix,delimiter=',')

w1 = numpy.where(numpy.mean(output_matrix,axis=1) ==0.0)[0]

output_matrix = numpy.delete(output_matrix,w1,axis=0)

#MAP3A

mean_d13C_bulk_MAP3A = numpy.nanmean(output_matrix[:,w_d13C_bulk_MAP3A],axis=0)
std_d13C_bulk_MAP3A = numpy.nanstd(output_matrix[:,w_d13C_bulk_MAP3A],axis=0)

mean_d13C_org_MAP3A = numpy.nanmean(output_matrix[:,w_d13C_org_MAP3A],axis=0)
std_d13C_org_MAP3A = numpy.nanstd(output_matrix[:,w_d13C_org_MAP3A],axis=0)

mean_TEX86_MAP3A = numpy.nanmean(output_matrix[:,w_TEX86_MAP3A],axis=0)
std_TEX86_MAP3A = numpy.nanstd(output_matrix[:,w_TEX86_MAP3A],axis=0)

mean_d13C_PF_MAP3A = numpy.nanmean(output_matrix[:,w_d13C_PF_MAP3A],axis=0)
std_d13C_PF_MAP3A = numpy.nanstd(output_matrix[:,w_d13C_PF_MAP3A],axis=0)

mean_d13C_BF_MAP3A = numpy.nanmean(output_matrix[:,w_d13C_BF_MAP3A],axis=0)
std_d13C_BF_MAP3A = numpy.nanstd(output_matrix[:,w_d13C_BF_MAP3A],axis=0)

#MAP3B

mean_d13C_bulk_MAP3B = numpy.nanmean(output_matrix[:,w_d13C_bulk_MAP3B],axis=0)
std_d13C_bulk_MAP3B = numpy.nanstd(output_matrix[:,w_d13C_bulk_MAP3B],axis=0)

mean_d13C_org_MAP3B = numpy.nanmean(output_matrix[:,w_d13C_org_MAP3B],axis=0)
std_d13C_org_MAP3B = numpy.nanstd(output_matrix[:,w_d13C_org_MAP3B],axis=0)

mean_TEX86_MAP3B = numpy.nanmean(output_matrix[:,w_TEX86_MAP3B],axis=0)
std_TEX86_MAP3B = numpy.nanstd(output_matrix[:,w_TEX86_MAP3B],axis=0)

mean_d13C_PF_MAP3B = numpy.nanmean(output_matrix[:,w_d13C_PF_MAP3B],axis=0)
std_d13C_PF_MAP3B = numpy.nanstd(output_matrix[:,w_d13C_PF_MAP3B],axis=0)

mean_d13C_BF_MAP3B = numpy.nanmean(output_matrix[:,w_d13C_BF_MAP3B],axis=0)
std_d13C_BF_MAP3B = numpy.nanstd(output_matrix[:,w_d13C_BF_MAP3B],axis=0)

###

output_matrix_2 = wkspc + 'MC_probability_output_2.csv'	
output_matrix_2 = numpy.genfromtxt(output_matrix_2,delimiter=',')

w1 = numpy.where(numpy.mean(output_matrix_2,axis=1) ==0.0)[0]

output_matrix_2 = numpy.delete(output_matrix_2,w1,axis=0)

###

output_matrix_3 = wkspc + 'MC_probability_output_3.csv'	
output_matrix_3 = numpy.genfromtxt(output_matrix_3,delimiter=',')

w1 = numpy.where(numpy.mean(output_matrix_3,axis=1) ==0.0)[0]

output_matrix_3 = numpy.delete(output_matrix_3,w1,axis=0)

output_matrix_6 = wkspc + 'MC_probability_output_6.csv'	
output_matrix_6 = numpy.genfromtxt(output_matrix_6,delimiter=',')

w1 = numpy.where(numpy.mean(output_matrix_6,axis=1) ==0.0)[0]

output_matrix_6 = numpy.delete(output_matrix_6,w1,axis=0)

w_d13C_glob = numpy.where((s_vals1==10.0)&(s_vals2==10.0))[0]
w_d13C_bulk = numpy.where((s_vals1==4.0)&(s_vals2==0.0))[0]
w_d13C_org = numpy.where((s_vals1==4.0)& (s_vals2==1.0))[0]
w_d13C_PF = numpy.where((s_vals1==4.0)& (s_vals2==3.0))[0]
w_d13C_BF = numpy.where((s_vals1==4.0)& (s_vals2==4.0))[0]
w_TEX86 = numpy.where((s_vals1==4.0)& (s_vals2==2.0))[0]

mean_global = numpy.nanmean(output_matrix_3[:,w_d13C_glob],axis=0)
std_global = numpy.nanstd(output_matrix_3[:,w_d13C_glob],axis=0)

mean_NJ_d13C_bulk = numpy.nanmean(output_matrix_2[:,w_d13C_bulk],axis=0)
std_NJ_d13C_bulk = numpy.nanstd(output_matrix_2[:,w_d13C_bulk],axis=0)

mean_NJ_d13C_org = numpy.nanmean(output_matrix_2[:,w_d13C_org],axis=0)
std_NJ_d13C_org = numpy.nanstd(output_matrix_2[:,w_d13C_org],axis=0)

mean_NJ_d13C_PF = numpy.nanmean(output_matrix_2[:,w_d13C_PF],axis=0)
std_NJ_d13C_PF = numpy.nanstd(output_matrix_2[:,w_d13C_PF],axis=0)

mean_NJ_d13C_BF = numpy.nanmean(output_matrix_2[:,w_d13C_BF],axis=0)
std_NJ_d13C_BF = numpy.nanstd(output_matrix_2[:,w_d13C_BF],axis=0)

mean_NJ_TEX86 = numpy.nanmean(output_matrix_2[:,w_TEX86],axis=0)
std_NJ_TEX86 = numpy.nanstd(output_matrix_2[:,w_TEX86],axis=0)

###

output_matrix_4 = wkspc + 'MC_probability_output_4.csv'	
output_matrix_4 = numpy.genfromtxt(output_matrix_4,delimiter=',')

w1 = numpy.where(numpy.mean(output_matrix_4,axis=1) ==0.0)[0]

output_matrix_4 = numpy.delete(output_matrix_4,w1,axis=0)

mean_timing = (numpy.nanmean(output_matrix_4,axis=0))
std_timing = numpy.nanstd(output_matrix_4,axis=0)*(1000000./1000.)

fig= plt.figure(1,figsize=(16.,16.5))

ax1 = plt.subplot(341)
###

MC_iterations = 0
for n in range(0,MC_iterations):
	
	val2 = numpy.random.normal(loc = mean_NJ_d13C_bulk, scale = std_NJ_d13C_bulk)
	
	ax1.plot((val2),t_vals[w_d13C_bulk],color='tomato',marker = 'x',markersize=2.0,linewidth=0.0,alpha=0.002)

out_mat_fold = 1

ax1.plot(SL_data_combined[w_d13C_bulk_c,1],mean_timing[w_d13C_bulk_c],color='tomato',marker='o',linewidth=0.0,markersize=2.0,alpha=0.75,label="NJ $\delta$$^{13}$C$_{bulk}$ data")

ax1.plot(mean_NJ_d13C_bulk,t_vals[w_d13C_bulk],color='tomato',linewidth=1.0)
ax1.plot([0,0],[0,0],color='tomato',linewidth=1.0,label="$\delta$$^{13}$C$_{bulk}$; $f(t)$ + $\Delta$$_{1}(t)$")
ax1.plot(mean_NJ_d13C_bulk + (std_NJ_d13C_bulk),t_vals[w_d13C_bulk],color='tomato',linestyle='--',linewidth=0.75)
ax1.plot(mean_NJ_d13C_bulk - (std_NJ_d13C_bulk),t_vals[w_d13C_bulk],color='tomato',linestyle='--',linewidth=0.75)
ax1.plot(mean_NJ_d13C_bulk + (2. * std_NJ_d13C_bulk),t_vals[w_d13C_bulk],color='tomato',linestyle=':',linewidth=0.5)
ax1.plot(mean_NJ_d13C_bulk - (2. * std_NJ_d13C_bulk),t_vals[w_d13C_bulk],color='tomato',linestyle=':',linewidth=0.5)
ax1.set_ylim(60,51)
ax1.set_xlim(-6.,0.5)

ax1.legend(fontsize=8)

ax1.set_title("MAP3A/B bulk")
ax1.set_xlabel("$\delta$$^{13}$C")

ax1.set_ylabel("Depth (ft.)")

ax1 = plt.subplot(342)

###

MC_iterations = 0
for n in range(0,MC_iterations):
	
	val2 = numpy.random.normal(loc = mean_NJ_d13C_org, scale = std_NJ_d13C_org)
	
	ax1.plot((val2),t_vals[w_d13C_org],color='forestgreen',marker = 'x',markersize=2.0,linewidth=0.0,alpha=0.002)

out_mat_fold = 1

ax1.plot(SL_data_combined[w_d13C_org_c,1],mean_timing[w_d13C_org_c],color='forestgreen',marker='o',linewidth=0.0,markersize=2.0,alpha=0.75,label="NJ $\delta$$^{13}$C$_{org}$ data")

ax1.plot(mean_NJ_d13C_org,t_vals[w_d13C_org],color='forestgreen',linewidth=1.0)
ax1.plot([0,0],[0,0],color='forestgreen',linewidth=1.0,label="$\delta$$^{13}$C$_{org}$; $f(t)$ + $\Delta$$_{2}(t)$")
ax1.plot(mean_NJ_d13C_org + (std_NJ_d13C_org),t_vals[w_d13C_org],color='forestgreen',linestyle='--',linewidth=0.75)
ax1.plot(mean_NJ_d13C_org - (std_NJ_d13C_org),t_vals[w_d13C_org],color='forestgreen',linestyle='--',linewidth=0.75)
ax1.plot(mean_NJ_d13C_org + (2. * std_NJ_d13C_org),t_vals[w_d13C_org],color='forestgreen',linestyle=':',linewidth=0.5)
ax1.plot(mean_NJ_d13C_org - (2. * std_NJ_d13C_org),t_vals[w_d13C_org],color='forestgreen',linestyle=':',linewidth=0.5)
ax1.set_ylim(60,51)
ax1.set_xlim(-31.,-24.5)

ax1.set_title("MAP3A/B organic")
ax1.set_xlabel("$\delta$$^{13}$C")

ax1.legend(fontsize=8)

ax1 = plt.subplot(343)
###

MC_iterations = 0
for n in range(0,MC_iterations):
	
	val2 = numpy.random.normal(loc = mean_NJ_d13C_PF, scale = std_NJ_d13C_PF)
	
	ax1.plot((val2),t_vals[w_d13C_PF],color='dodgerblue',marker = 'x',markersize=2.0,linewidth=0.0,alpha=0.002)

out_mat_fold = 1

ax1.plot(SL_data_combined[w_d13C_PF_c,1],mean_timing[w_d13C_PF_c],color='dodgerblue',marker='o',linewidth=0.0,markersize=2.0,alpha=0.75,label="NJ $\delta$$^{13}$C$_{PF}$ data")

ax1.plot(mean_NJ_d13C_PF,t_vals[w_d13C_PF],color='dodgerblue',linewidth=1.0)
ax1.plot([0,0],[0,0],color='dodgerblue',linewidth=1.0,label="$\delta$$^{13}$C$_{PF}$; $f(t)$ + $\Delta$$_{3}(t)$")
ax1.plot(mean_NJ_d13C_PF + (std_NJ_d13C_PF),t_vals[w_d13C_PF],color='dodgerblue',linestyle='--',linewidth=0.75)
ax1.plot(mean_NJ_d13C_PF - (std_NJ_d13C_PF),t_vals[w_d13C_PF],color='dodgerblue',linestyle='--',linewidth=0.75)
ax1.plot(mean_NJ_d13C_PF + (2. * std_NJ_d13C_PF),t_vals[w_d13C_PF],color='dodgerblue',linestyle=':',linewidth=0.5)
ax1.plot(mean_NJ_d13C_PF - (2. * std_NJ_d13C_PF),t_vals[w_d13C_PF],color='dodgerblue',linestyle=':',linewidth=0.5)

MC_iterations = 0
for n in range(0,MC_iterations):
	
	val2 = numpy.random.normal(loc = mean_NJ_d13C_BF, scale = std_NJ_d13C_BF)
	
	ax1.plot((val2),t_vals[w_d13C_BF],color='indigo',marker = 'x',markersize=2.0,linewidth=0.0,alpha=0.002)

out_mat_fold = 1

ax1.plot(SL_data_combined[w_d13C_BF_c,1],mean_timing[w_d13C_BF_c],color='indigo',marker='o',linewidth=0.0,markersize=2.0,alpha=0.75,label="NJ $\delta$$^{13}$C$_{BF}$ data")

ax1.plot(mean_NJ_d13C_BF,t_vals[w_d13C_BF],color='indigo',linewidth=1.0)
ax1.plot([0,0],[0,0],color='indigo',linewidth=1.0,label="$\delta$$^{13}$C$_{BF}$; $f(t)$ + $\Delta$$_{4}(t)$")
ax1.plot(mean_NJ_d13C_BF + (std_NJ_d13C_BF),t_vals[w_d13C_BF],color='indigo',linestyle='--',linewidth=0.75)
ax1.plot(mean_NJ_d13C_BF - (std_NJ_d13C_BF),t_vals[w_d13C_BF],color='indigo',linestyle='--',linewidth=0.75)
ax1.plot(mean_NJ_d13C_BF + (2. * std_NJ_d13C_BF),t_vals[w_d13C_BF],color='indigo',linestyle=':',linewidth=0.5)
ax1.plot(mean_NJ_d13C_BF - (2. * std_NJ_d13C_BF),t_vals[w_d13C_BF],color='indigo',linestyle=':',linewidth=0.5)
ax1.set_ylim(60,51)
ax1.set_xlim(-4.5,2.)

ax1.set_title("MAP3A/B forams")
ax1.set_xlabel("$\delta$$^{13}$C")

ax1.legend(fontsize=8)

ax1 = plt.subplot(344)

MC_iterations = 0
for n in range(0,MC_iterations):
	
	val2 = numpy.random.normal(loc = mean_NJ_TEX86, scale = std_NJ_TEX86)
	
	ax1.plot((val2),t_vals[w_TEX86],color='mediumvioletred',marker = 'x',markersize=2.0,linewidth=0.0,alpha=0.002)

out_mat_fold = 1

ax1.plot(SL_data_combined[w_TEX86_c,1],mean_timing[w_TEX86_c],color='mediumvioletred',marker='o',linewidth=0.0,markersize=2.0,alpha=0.75,label="TEX86 data")

ax1.plot(mean_NJ_TEX86,t_vals[w_TEX86],color='mediumvioletred',linewidth=1.0)
ax1.plot([0,0],[0,0],color='mediumvioletred',linewidth=1.0,label="TEX86; $n(t)$")
ax1.plot(mean_NJ_TEX86 + (std_NJ_TEX86),t_vals[w_TEX86],color='mediumvioletred',linestyle='--',linewidth=0.75)
ax1.plot(mean_NJ_TEX86 - (std_NJ_TEX86),t_vals[w_TEX86],color='mediumvioletred',linestyle='--',linewidth=0.75)
ax1.plot(mean_NJ_TEX86 + (2. * std_NJ_TEX86),t_vals[w_TEX86],color='mediumvioletred',linestyle=':',linewidth=0.5)
ax1.plot(mean_NJ_TEX86 - (2. * std_NJ_TEX86),t_vals[w_TEX86],color='mediumvioletred',linestyle=':',linewidth=0.5)
ax1.set_ylim(60,51)
ax1.set_xlim(26,38)

ax1.set_title("MAP3A/B temperature")
ax1.set_xlabel("Temperature ($\degree$C)")

ax1.legend(fontsize=8)

ax1 = plt.subplot(3,4,5)
###

MC_iterations = 0
for n in range(0,MC_iterations):
	
	val2 = numpy.random.normal(loc = mean_d13C_bulk_MAP3A, scale = std_d13C_bulk_MAP3A)
	
	ax1.plot((val2),t_vals[w_d13C_bulk_MAP3A],color='tomato',marker = 'x',markersize=2.0,linewidth=0.0,alpha=0.002)

out_mat_fold = 1

ax1.plot(SL_data_combined[w_d13C_bulk_MAP3A_c,1],mean_timing[w_d13C_bulk_MAP3A_c],color='tomato',marker='o',linewidth=0.0,markersize=4.0,alpha=0.5,label="MAP3A $\delta$$^{13}$C$_{bulk}$ data")

ax1.plot(mean_d13C_bulk_MAP3A,t_vals[w_d13C_bulk_MAP3A],color='tomato',linewidth=1.0)
ax1.plot([0,0],[0,0],color='tomato',linewidth=1.0,label="MAP3A $\delta$$^{13}$C$_{bulk}$; $g_{1,1}(t)$")
ax1.plot(mean_d13C_bulk_MAP3A + (std_d13C_bulk_MAP3A),t_vals[w_d13C_bulk_MAP3A],color='tomato',linestyle='--',linewidth=0.75)
ax1.plot(mean_d13C_bulk_MAP3A - (std_d13C_bulk_MAP3A),t_vals[w_d13C_bulk_MAP3A],color='tomato',linestyle='--',linewidth=0.75)
ax1.plot(mean_d13C_bulk_MAP3A + (2. * std_d13C_bulk_MAP3A),t_vals[w_d13C_bulk_MAP3A],color='tomato',linestyle=':',linewidth=0.5)
ax1.plot(mean_d13C_bulk_MAP3A - (2. * std_d13C_bulk_MAP3A),t_vals[w_d13C_bulk_MAP3A],color='tomato',linestyle=':',linewidth=0.5)
ax1.set_ylim(60,51)
ax1.set_xlim(-6.,0.5)

ax1.set_title("MAP3A bulk")
ax1.set_xlabel("$\delta$$^{13}$C")
ax1.set_ylabel("Depth (ft.)")
ax1.legend(fontsize=8)

ax1 = plt.subplot(3,4,9)
###

MC_iterations = 0
for n in range(0,MC_iterations):
	
	val2 = numpy.random.normal(loc = mean_d13C_bulk_MAP3B, scale = std_d13C_bulk_MAP3B)
	
	ax1.plot((val2),t_vals[w_d13C_bulk_MAP3B],color='tomato',marker = 'x',markersize=2.0,linewidth=0.0,alpha=0.002)

out_mat_fold = 1

ax1.plot(SL_data_combined[w_d13C_bulk_MAP3B_c,1],mean_timing[w_d13C_bulk_MAP3B_c],color='tomato',marker='o',linewidth=0.0,markersize=4.0,alpha=0.5,label="MAP3B $\delta$$^{13}$C$_{bulk}$ data")

ax1.plot(mean_d13C_bulk_MAP3B,t_vals[w_d13C_bulk_MAP3B],color='tomato',linewidth=1.0)
ax1.plot([0,0],[0,0],color='tomato',linewidth=1.0,label="MAP3B $\delta$$^{13}$C$_{bulk}$; $g_{1,1}(t)$")
ax1.plot(mean_d13C_bulk_MAP3B + (std_d13C_bulk_MAP3B),t_vals[w_d13C_bulk_MAP3B],color='tomato',linestyle='--',linewidth=0.75)
ax1.plot(mean_d13C_bulk_MAP3B - (std_d13C_bulk_MAP3B),t_vals[w_d13C_bulk_MAP3B],color='tomato',linestyle='--',linewidth=0.75)
ax1.plot(mean_d13C_bulk_MAP3B + (2. * std_d13C_bulk_MAP3B),t_vals[w_d13C_bulk_MAP3B],color='tomato',linestyle=':',linewidth=0.5)
ax1.plot(mean_d13C_bulk_MAP3B - (2. * std_d13C_bulk_MAP3B),t_vals[w_d13C_bulk_MAP3B],color='tomato',linestyle=':',linewidth=0.5)
ax1.set_ylim(60,51)
ax1.set_xlim(-6.,0.5)

ax1.set_title("MAP3B bulk")
ax1.set_xlabel("$\delta$$^{13}$C")
ax1.set_ylabel("Depth (ft.)")
ax1.legend(fontsize=8)

########################
########################
########################

ax1 = plt.subplot(3,4,6)
###

MC_iterations = 0
for n in range(0,MC_iterations):
	
	val2 = numpy.random.normal(loc = mean_d13C_org_MAP3A, scale = std_d13C_org_MAP3A)
	
	ax1.plot((val2),t_vals[w_d13C_org_MAP3A],color='forestgreen',marker = 'x',markersize=2.0,linewidth=0.0,alpha=0.002)

out_mat_fold = 1

ax1.plot(SL_data_combined[w_d13C_org_MAP3A_c,1],mean_timing[w_d13C_org_MAP3A_c],color='forestgreen',marker='o',linewidth=0.0,markersize=4.0,alpha=0.5,label="MAP3A $\delta$$^{13}$C$_{org}$ data")

ax1.plot(mean_d13C_org_MAP3A,t_vals[w_d13C_org_MAP3A],color='forestgreen',linewidth=1.0)
ax1.plot([0,0],[0,0],color='forestgreen',linewidth=1.0,label="MAP3A $\delta$$^{13}$C$_{org}$; $g_{2,1}(t)$")
ax1.plot(mean_d13C_org_MAP3A + (std_d13C_org_MAP3A),t_vals[w_d13C_org_MAP3A],color='forestgreen',linestyle='--',linewidth=0.75)
ax1.plot(mean_d13C_org_MAP3A - (std_d13C_org_MAP3A),t_vals[w_d13C_org_MAP3A],color='forestgreen',linestyle='--',linewidth=0.75)
ax1.plot(mean_d13C_org_MAP3A + (2. * std_d13C_org_MAP3A),t_vals[w_d13C_org_MAP3A],color='forestgreen',linestyle=':',linewidth=0.5)
ax1.plot(mean_d13C_org_MAP3A - (2. * std_d13C_org_MAP3A),t_vals[w_d13C_org_MAP3A],color='forestgreen',linestyle=':',linewidth=0.5)
ax1.set_ylim(60,51)
ax1.set_xlim(-31.,-24.5)

ax1.set_title("MAP3A organic")
ax1.set_xlabel("$\delta$$^{13}$C")
ax1.set_ylabel("Depth (ft.)")
ax1.legend(fontsize=8)

ax1 = plt.subplot(3,4,10)
###

MC_iterations = 0
for n in range(0,MC_iterations):
	
	val2 = numpy.random.normal(loc = mean_d13C_org_MAP3B, scale = std_d13C_org_MAP3B)
	
	ax1.plot((val2),t_vals[w_d13C_org_MAP3B],color='forestgreen',marker = 'x',markersize=2.0,linewidth=0.0,alpha=0.002)

out_mat_fold = 1

ax1.plot(SL_data_combined[w_d13C_org_MAP3B_c,1],mean_timing[w_d13C_org_MAP3B_c],color='forestgreen',marker='o',linewidth=0.0,markersize=4.0,alpha=0.5,label="MAP3B $\delta$$^{13}$C$_{org}$ data")

ax1.plot(mean_d13C_org_MAP3B,t_vals[w_d13C_org_MAP3B],color='forestgreen',linewidth=1.0)
ax1.plot([0,0],[0,0],color='forestgreen',linewidth=1.0,label="MAP3B $\delta$$^{13}$C$_{org}$; $g_{2,1}(t)$")
ax1.plot(mean_d13C_org_MAP3B + (std_d13C_org_MAP3B),t_vals[w_d13C_org_MAP3B],color='forestgreen',linestyle='--',linewidth=0.75)
ax1.plot(mean_d13C_org_MAP3B - (std_d13C_org_MAP3B),t_vals[w_d13C_org_MAP3B],color='forestgreen',linestyle='--',linewidth=0.75)
ax1.plot(mean_d13C_org_MAP3B + (2. * std_d13C_org_MAP3B),t_vals[w_d13C_org_MAP3B],color='forestgreen',linestyle=':',linewidth=0.5)
ax1.plot(mean_d13C_org_MAP3B - (2. * std_d13C_org_MAP3B),t_vals[w_d13C_org_MAP3B],color='forestgreen',linestyle=':',linewidth=0.5)
ax1.set_ylim(60,51)
ax1.set_xlim(-31.,-24.5)

ax1.set_title("MAP3B organic")
ax1.set_xlabel("$\delta$$^{13}$C")
ax1.set_ylabel("Depth (ft.)")
ax1.legend(fontsize=8)

########################
########################
########################

ax1 = plt.subplot(3,4,7)
###

MC_iterations = 0
for n in range(0,MC_iterations):
	
	val2 = numpy.random.normal(loc = mean_d13C_PF_MAP3A, scale = std_d13C_PF_MAP3A)
	
	ax1.plot((val2),t_vals[w_d13C_PF_MAP3A],color='dodgerblue',marker = 'x',markersize=2.0,linewidth=0.0,alpha=0.002)

out_mat_fold = 1

ax1.plot(SL_data_combined[w_d13C_PF_MAP3A_c,1],mean_timing[w_d13C_PF_MAP3A_c],color='dodgerblue',marker='o',linewidth=0.0,markersize=4.0,alpha=0.5,label="MAP3A $\delta$$^{13}$C$_{PF}$ data")

ax1.plot(mean_d13C_PF_MAP3A,t_vals[w_d13C_PF_MAP3A],color='dodgerblue',linewidth=1.0)
ax1.plot([0,0],[0,0],color='dodgerblue',linewidth=1.0,label="MAP3A $\delta$$^{13}$C$_{PF}$; $g_{3,1}(t)$")
ax1.plot(mean_d13C_PF_MAP3A + (std_d13C_PF_MAP3A),t_vals[w_d13C_PF_MAP3A],color='dodgerblue',linestyle='--',linewidth=0.75)
ax1.plot(mean_d13C_PF_MAP3A - (std_d13C_PF_MAP3A),t_vals[w_d13C_PF_MAP3A],color='dodgerblue',linestyle='--',linewidth=0.75)
ax1.plot(mean_d13C_PF_MAP3A + (2. * std_d13C_PF_MAP3A),t_vals[w_d13C_PF_MAP3A],color='dodgerblue',linestyle=':',linewidth=0.5)
ax1.plot(mean_d13C_PF_MAP3A - (2. * std_d13C_PF_MAP3A),t_vals[w_d13C_PF_MAP3A],color='dodgerblue',linestyle=':',linewidth=0.5)

MC_iterations = 0
for n in range(0,MC_iterations):
	
	val2 = numpy.random.normal(loc = mean_d13C_BF_MAP3A, scale = std_d13C_BF_MAP3A)
	
	ax1.plot((val2),t_vals[w_d13C_BF_MAP3A],color='indigo',marker = 'x',markersize=2.0,linewidth=0.0,alpha=0.002)

out_mat_fold = 1

ax1.plot(SL_data_combined[w_d13C_BF_MAP3A_c,1],mean_timing[w_d13C_BF_MAP3A_c],color='indigo',marker='o',linewidth=0.0,markersize=4.0,alpha=0.5,label="MAP3A $\delta$$^{13}$C$_{BF}$ data")

ax1.plot(mean_d13C_BF_MAP3A,t_vals[w_d13C_BF_MAP3A],color='indigo',linewidth=1.0)
ax1.plot([0,0],[0,0],color='indigo',linewidth=1.0,label="MAP3A $\delta$$^{13}$C$_{BF}$; $g_{4,1}(t)$")
ax1.plot(mean_d13C_BF_MAP3A + (std_d13C_BF_MAP3A),t_vals[w_d13C_BF_MAP3A],color='indigo',linestyle='--',linewidth=0.75)
ax1.plot(mean_d13C_BF_MAP3A - (std_d13C_BF_MAP3A),t_vals[w_d13C_BF_MAP3A],color='indigo',linestyle='--',linewidth=0.75)
ax1.plot(mean_d13C_BF_MAP3A + (2. * std_d13C_BF_MAP3A),t_vals[w_d13C_BF_MAP3A],color='indigo',linestyle=':',linewidth=0.5)
ax1.plot(mean_d13C_BF_MAP3A - (2. * std_d13C_BF_MAP3A),t_vals[w_d13C_BF_MAP3A],color='indigo',linestyle=':',linewidth=0.5)

ax1.set_ylim(60,51)
ax1.set_xlim(-4.5,2.)

ax1.set_title("MAP3A forams")
ax1.set_xlabel("$\delta$$^{13}$C")
ax1.set_ylabel("Depth (ft.)")
ax1.set_ylabel("Depth (ft.)")
ax1.legend(fontsize=8)

ax1 = plt.subplot(3,4,11)
###

MC_iterations = 0
for n in range(0,MC_iterations):
	
	val2 = numpy.random.normal(loc = mean_d13C_PF_MAP3B, scale = std_d13C_PF_MAP3B)
	
	ax1.plot((val2),t_vals[w_d13C_PF_MAP3B],color='dodgerblue',marker = 'x',markersize=2.0,linewidth=0.0,alpha=0.002)

out_mat_fold = 1

ax1.plot(SL_data_combined[w_d13C_PF_MAP3B_c,1],mean_timing[w_d13C_PF_MAP3B_c],color='dodgerblue',marker='o',linewidth=0.0,markersize=4.0,alpha=0.5,label="MAP3B $\delta$$^{13}$C$_{PF}$ data")

ax1.plot(mean_d13C_PF_MAP3B,t_vals[w_d13C_PF_MAP3B],color='dodgerblue',linewidth=1.0)
ax1.plot([0,0],[0,0],color='dodgerblue',linewidth=1.0,label="MAP3B $\delta$$^{13}$C$_{PF}$; $g_{3,1}(t)$")
ax1.plot(mean_d13C_PF_MAP3B + (std_d13C_PF_MAP3B),t_vals[w_d13C_PF_MAP3B],color='dodgerblue',linestyle='--',linewidth=0.75)
ax1.plot(mean_d13C_PF_MAP3B - (std_d13C_PF_MAP3B),t_vals[w_d13C_PF_MAP3B],color='dodgerblue',linestyle='--',linewidth=0.75)
ax1.plot(mean_d13C_PF_MAP3B + (2. * std_d13C_PF_MAP3B),t_vals[w_d13C_PF_MAP3B],color='dodgerblue',linestyle=':',linewidth=0.5)
ax1.plot(mean_d13C_PF_MAP3B - (2. * std_d13C_PF_MAP3B),t_vals[w_d13C_PF_MAP3B],color='dodgerblue',linestyle=':',linewidth=0.5)

MC_iterations = 0
for n in range(0,MC_iterations):
	
	val2 = numpy.random.normal(loc = mean_d13C_BF_MAP3B, scale = std_d13C_BF_MAP3B)
	
	ax1.plot((val2),t_vals[w_d13C_BF_MAP3B],color='indigo',marker = 'x',markersize=2.0,linewidth=0.0,alpha=0.002)

out_mat_fold = 1

ax1.plot(SL_data_combined[w_d13C_BF_MAP3B_c,1],mean_timing[w_d13C_BF_MAP3B_c],color='indigo',marker='o',linewidth=0.0,markersize=4.0,alpha=0.5,label="MAP3B $\delta$$^{13}$C$_{BF}$ data")

ax1.plot(mean_d13C_BF_MAP3B,t_vals[w_d13C_BF_MAP3B],color='indigo',linewidth=1.0)
ax1.plot([0,0],[0,0],color='indigo',linewidth=1.0,label="MAP3B $\delta$$^{13}$C$_{BF}$; $g_{4,1}(t)$")
ax1.plot(mean_d13C_BF_MAP3B + (std_d13C_BF_MAP3B),t_vals[w_d13C_BF_MAP3B],color='indigo',linestyle='--',linewidth=0.75)
ax1.plot(mean_d13C_BF_MAP3B - (std_d13C_BF_MAP3B),t_vals[w_d13C_BF_MAP3B],color='indigo',linestyle='--',linewidth=0.75)
ax1.plot(mean_d13C_BF_MAP3B + (2. * std_d13C_BF_MAP3B),t_vals[w_d13C_BF_MAP3B],color='indigo',linestyle=':',linewidth=0.5)
ax1.plot(mean_d13C_BF_MAP3B - (2. * std_d13C_BF_MAP3B),t_vals[w_d13C_BF_MAP3B],color='indigo',linestyle=':',linewidth=0.5)

ax1.set_ylim(60,51)
ax1.set_xlim(-4.5,2.)

ax1.set_title("MAP3B forams")
ax1.set_xlabel("$\delta$$^{13}$C")
ax1.set_ylabel("Depth (ft.)")
ax1.set_ylabel("Depth (ft.)")
ax1.legend(fontsize=8)

########################
########################
########################

ax1 = plt.subplot(3,4,8)
###

MC_iterations = 0
for n in range(0,MC_iterations):
	
	val2 = numpy.random.normal(loc = mean_TEX86_MAP3A, scale = std_TEX86_MAP3A)
	
	ax1.plot((val2),t_vals[w_TEX86_MAP3A],color='mediumvioletred',marker = 'x',markersize=2.0,linewidth=0.0,alpha=0.002)

out_mat_fold = 1

ax1.plot(SL_data_combined[w_TEX86_MAP3A_c,1],mean_timing[w_TEX86_MAP3A_c],color='mediumvioletred',marker='o',linewidth=0.0,markersize=4.0,alpha=0.5,label="MAP3A TEX86 data")

ax1.plot(mean_TEX86_MAP3A,t_vals[w_TEX86_MAP3A],color='mediumvioletred',linewidth=1.0)
ax1.plot([0,0],[0,0],color='mediumvioletred',linewidth=1.0,label="MAP3A TEX86; $h_{1}(t)$")
ax1.plot(mean_TEX86_MAP3A + (std_TEX86_MAP3A),t_vals[w_TEX86_MAP3A],color='mediumvioletred',linestyle='--',linewidth=0.75)
ax1.plot(mean_TEX86_MAP3A - (std_TEX86_MAP3A),t_vals[w_TEX86_MAP3A],color='mediumvioletred',linestyle='--',linewidth=0.75)
ax1.plot(mean_TEX86_MAP3A + (2. * std_TEX86_MAP3A),t_vals[w_TEX86_MAP3A],color='mediumvioletred',linestyle=':',linewidth=0.5)
ax1.plot(mean_TEX86_MAP3A - (2. * std_TEX86_MAP3A),t_vals[w_TEX86_MAP3A],color='mediumvioletred',linestyle=':',linewidth=0.5)
ax1.set_ylim(60,51)
ax1.set_xlim(26,38)

ax1.set_title("MAP3A temperature")
ax1.set_xlabel("Temperature ($\degree$C)")
ax1.set_ylabel("Depth (ft.)")
ax1.legend(fontsize=8)

ax1 = plt.subplot(3,4,12)
###

MC_iterations = 0
for n in range(0,MC_iterations):
	
	val2 = numpy.random.normal(loc = mean_TEX86_MAP3B, scale = std_TEX86_MAP3B)
	
	ax1.plot((val2),t_vals[w_TEX86_MAP3B],color='mediumvioletred',marker = 'x',markersize=2.0,linewidth=0.0,alpha=0.002)

out_mat_fold = 1

ax1.plot(SL_data_combined[w_TEX86_MAP3B_c,1],mean_timing[w_TEX86_MAP3B_c],color='mediumvioletred',marker='o',linewidth=0.0,markersize=4.0,alpha=0.5,label="MAP3B TEX86 data")

ax1.plot(mean_TEX86_MAP3B,t_vals[w_TEX86_MAP3B],color='mediumvioletred',linewidth=1.0)
ax1.plot([0,0],[0,0],color='mediumvioletred',linewidth=1.0,label="MAP3B TEX86; $h_{1}(t)$")
ax1.plot(mean_TEX86_MAP3B + (std_TEX86_MAP3B),t_vals[w_TEX86_MAP3B],color='mediumvioletred',linestyle='--',linewidth=0.75)
ax1.plot(mean_TEX86_MAP3B - (std_TEX86_MAP3B),t_vals[w_TEX86_MAP3B],color='mediumvioletred',linestyle='--',linewidth=0.75)
ax1.plot(mean_TEX86_MAP3B + (2. * std_TEX86_MAP3B),t_vals[w_TEX86_MAP3B],color='mediumvioletred',linestyle=':',linewidth=0.5)
ax1.plot(mean_TEX86_MAP3B - (2. * std_TEX86_MAP3B),t_vals[w_TEX86_MAP3B],color='mediumvioletred',linestyle=':',linewidth=0.5)
ax1.set_ylim(60,51)
ax1.set_xlim(26,38)

ax1.set_title("MAP3B temperature")
ax1.set_xlabel("Temperature ($\degree$C)")
ax1.set_ylabel("Depth (ft.)")
ax1.legend(fontsize=8)

plt.tight_layout()

pltname = wkspc +  'MAP3AB_d13C_Tex86_20230301.png'

plt.savefig(pltname, dpi = 300)

plt.close()

fig= plt.figure(2,figsize=(16.,16.5))

ax1 = plt.subplot(3,4,1)
###

MC_iterations = 0
for n in range(0,MC_iterations):
	
	val2 = numpy.random.normal(loc = mean_NJ_d13C_bulk, scale = std_NJ_d13C_bulk)
	
	ax1.plot((val2),t_vals[w_d13C_glob],color='tomato',marker = 'x',markersize=2.0,linewidth=0.0,alpha=0.002)

out_mat_fold = 1

ax1.plot(SL_data_combined[w_d13C_bulk_c,1],mean_timing[w_d13C_bulk_c],color='tomato',marker='o',linewidth=0.0,markersize=2.0,alpha=0.5,label="NJ $\delta$$^{13}$C$_{bulk}$ data")

ax1.plot(mean_NJ_d13C_bulk,t_vals[w_d13C_glob],color='tomato',linewidth=1.0)
ax1.plot([0,0],[0,0],color='tomato',linewidth=1.0,label="$\delta$$^{13}$C$_{bulk}$")
ax1.plot(mean_NJ_d13C_bulk + (std_NJ_d13C_bulk),t_vals[w_d13C_glob],color='tomato',linestyle='--',linewidth=0.75)
ax1.plot(mean_NJ_d13C_bulk - (std_NJ_d13C_bulk),t_vals[w_d13C_glob],color='tomato',linestyle='--',linewidth=0.75)
ax1.plot(mean_NJ_d13C_bulk + (2. * std_NJ_d13C_bulk),t_vals[w_d13C_glob],color='tomato',linestyle=':',linewidth=0.5)
ax1.plot(mean_NJ_d13C_bulk - (2. * std_NJ_d13C_bulk),t_vals[w_d13C_glob],color='tomato',linestyle=':',linewidth=0.5)

ax1.set_yticks(numpy.arange(51.,60.,0.25), minor=True)

ax1.set_ylim(60,51)
ax1.set_xlim(-6.,0.5)
ax1.legend(fontsize=8)

ax1.set_title("MAP3A/B bulk")
ax1.set_xlabel("$\delta$$^{13}$C")
ax1.set_ylabel("Depth (ft.)")
ax1 = plt.subplot(3,4,2)

###

MC_iterations = 0
for n in range(0,MC_iterations):
	
	val2 = numpy.random.normal(loc = mean_NJ_d13C_org, scale = std_NJ_d13C_org)
	
	ax1.plot((val2),t_vals[w_d13C_glob],color='forestgreen',marker = 'x',markersize=2.0,linewidth=0.0,alpha=0.002)

out_mat_fold = 1

ax1.plot(SL_data_combined[w_d13C_org_c,1],mean_timing[w_d13C_org_c],color='forestgreen',marker='o',linewidth=0.0,markersize=2.0,alpha=0.5,label="NJ $\delta$$^{13}$C$_{org}$ data")

ax1.plot(mean_NJ_d13C_org,t_vals[w_d13C_glob],color='forestgreen',linewidth=1.0)
ax1.plot([0,0],[0,0],color='forestgreen',linewidth=1.0,label="$\delta$$^{13}$C$_{org}$")
ax1.plot(mean_NJ_d13C_org + (std_NJ_d13C_org),t_vals[w_d13C_glob],color='forestgreen',linestyle='--',linewidth=0.75)
ax1.plot(mean_NJ_d13C_org - (std_NJ_d13C_org),t_vals[w_d13C_glob],color='forestgreen',linestyle='--',linewidth=0.75)
ax1.plot(mean_NJ_d13C_org + (2. * std_NJ_d13C_org),t_vals[w_d13C_glob],color='forestgreen',linestyle=':',linewidth=0.5)
ax1.plot(mean_NJ_d13C_org - (2. * std_NJ_d13C_org),t_vals[w_d13C_glob],color='forestgreen',linestyle=':',linewidth=0.5)

ax1.set_yticks(numpy.arange(51.,60.,0.25), minor=True)

ax1.set_ylim(60,51)

ax1.set_xlim(-31.,-24.5)

ax1.set_title("MAP3A/B organic")
ax1.set_xlabel("$\delta$$^{13}$C")
ax1.legend(fontsize=8)

ax1 = plt.subplot(3,4,3)
###

MC_iterations = 0
for n in range(0,MC_iterations):
	
	val2 = numpy.random.normal(loc = mean_NJ_d13C_PF, scale = std_NJ_d13C_PF)
	
	ax1.plot((val2),t_vals[w_d13C_glob],color='dodgerblue',marker = 'x',markersize=2.0,linewidth=0.0,alpha=0.002)

out_mat_fold = 1

ax1.plot(SL_data_combined[w_d13C_PF_c,1],mean_timing[w_d13C_PF_c],color='dodgerblue',marker='o',linewidth=0.0,markersize=2.0,alpha=0.5,label="NJ $\delta$$^{13}$C$_{PF}$ data")

ax1.plot(mean_NJ_d13C_PF,t_vals[w_d13C_glob],color='dodgerblue',linewidth=1.0)
ax1.plot([0,0],[0,0],color='dodgerblue',linewidth=1.0,label="$\delta$$^{13}$C$_{PF}$")
ax1.plot(mean_NJ_d13C_PF + (std_NJ_d13C_PF),t_vals[w_d13C_glob],color='dodgerblue',linestyle='--',linewidth=0.75)
ax1.plot(mean_NJ_d13C_PF - (std_NJ_d13C_PF),t_vals[w_d13C_glob],color='dodgerblue',linestyle='--',linewidth=0.75)
ax1.plot(mean_NJ_d13C_PF + (2. * std_NJ_d13C_PF),t_vals[w_d13C_glob],color='dodgerblue',linestyle=':',linewidth=0.5)
ax1.plot(mean_NJ_d13C_PF - (2. * std_NJ_d13C_PF),t_vals[w_d13C_glob],color='dodgerblue',linestyle=':',linewidth=0.5)

MC_iterations = 0
for n in range(0,MC_iterations):
	
	val2 = numpy.random.normal(loc = mean_NJ_d13C_BF, scale = std_NJ_d13C_BF)
	
	ax1.plot((val2),t_vals[w_d13C_glob],color='indigo',marker = 'x',markersize=2.0,linewidth=0.0,alpha=0.002)

out_mat_fold = 1

ax1.plot(SL_data_combined[w_d13C_BF_c,1],mean_timing[w_d13C_BF_c],color='indigo',marker='o',linewidth=0.0,markersize=2.0,alpha=0.5,label="NJ $\delta$$^{13}$C$_{BF}$ data")

ax1.plot(mean_NJ_d13C_BF,t_vals[w_d13C_glob],color='indigo',linewidth=1.0)
ax1.plot([0,0],[0,0],color='indigo',linewidth=1.0,label="$\delta$$^{13}$C$_{BF}$")
ax1.plot(mean_NJ_d13C_BF + (std_NJ_d13C_BF),t_vals[w_d13C_glob],color='indigo',linestyle='--',linewidth=0.75)
ax1.plot(mean_NJ_d13C_BF - (std_NJ_d13C_BF),t_vals[w_d13C_glob],color='indigo',linestyle='--',linewidth=0.75)
ax1.plot(mean_NJ_d13C_BF + (2. * std_NJ_d13C_BF),t_vals[w_d13C_glob],color='indigo',linestyle=':',linewidth=0.5)
ax1.plot(mean_NJ_d13C_BF - (2. * std_NJ_d13C_BF),t_vals[w_d13C_glob],color='indigo',linestyle=':',linewidth=0.5)

ax1.set_yticks(numpy.arange(51.,60.,0.25), minor=True)

ax1.set_ylim(60,51)

ax1.set_xlim(-4.5,2.)

ax1.set_title("MAP3A/B forams")
ax1.set_xlabel("$\delta$$^{13}$C")
ax1.legend(fontsize=8)

ax1 = plt.subplot(3,4,4)
###

MC_iterations = 0
for n in range(0,MC_iterations):
	
	val2 = numpy.random.normal(loc = mean_NJ_TEX86, scale = std_NJ_TEX86)
	
	ax1.plot((val2),t_vals[w_d13C_glob],color='mediumvioletred',marker = 'x',markersize=2.0,linewidth=0.0,alpha=0.002)

out_mat_fold = 1

ax1.plot(SL_data_combined[w_TEX86_c,1],mean_timing[w_TEX86_c],color='mediumvioletred',marker='o',linewidth=0.0,markersize=2.0,alpha=0.5,label="TEX86")

ax1.plot(mean_NJ_TEX86,t_vals[w_d13C_glob],color='mediumvioletred',linewidth=1.0)
ax1.plot([0,0],[0,0],color='mediumvioletred',linewidth=1.0,label="TEX86")
ax1.plot(mean_NJ_TEX86 + (std_NJ_TEX86),t_vals[w_d13C_glob],color='mediumvioletred',linestyle='--',linewidth=0.75)
ax1.plot(mean_NJ_TEX86 - (std_NJ_TEX86),t_vals[w_d13C_glob],color='mediumvioletred',linestyle='--',linewidth=0.75)
ax1.plot(mean_NJ_TEX86 + (2. * std_NJ_TEX86),t_vals[w_d13C_glob],color='mediumvioletred',linestyle=':',linewidth=0.5)
ax1.plot(mean_NJ_TEX86 - (2. * std_NJ_TEX86),t_vals[w_d13C_glob],color='mediumvioletred',linestyle=':',linewidth=0.5)

ax1.set_yticks(numpy.arange(51.,60.,0.25), minor=True)

ax1.set_ylim(60,51)

ax1.set_xlim(26,38)

ax1.set_title("MAP3A/B temperature")
ax1.set_xlabel("Temperature ($\degree$C)")
ax1.legend()

ax1 = plt.subplot(3,4,5)
###

out_matrix_BSL = numpy.zeros((len(output_matrix_6[:,0]),len(w_d13C_bulk)))
out_matrix2_BSL = numpy.zeros((len(output_matrix_6[:,0]),len(w_d13C_bulk)))

for n in range(0,len(output_matrix_6[:,0])):
	out_matrix_BSL[n,:] = numpy.gradient(output_matrix_6[n,w_d13C_bulk]*-1.,t_vals[w_d13C_bulk])
	out_matrix2_BSL[n,:] = numpy.gradient(out_matrix_BSL[n,:],t_vals[w_d13C_bulk])
	
mean_BSL_gradient_bulk = numpy.nanmean(out_matrix_BSL,axis=0)
std_BSL_gradient_bulk = numpy.nanstd(out_matrix_BSL,axis=0)

mean_BSL_gradient_bulk2 = numpy.nanmean(out_matrix2_BSL,axis=0)
std_BSL_gradient_bulk2 = numpy.nanstd(out_matrix2_BSL,axis=0)

MC_iterations = 0
for n in range(0,MC_iterations):
	
	val2 = numpy.random.normal(loc = mean_BSL_gradient_bulk, scale = std_BSL_gradient_bulk)
	
	ax1.plot((val2),t_vals[w_d13C_bulk],color='tomato',marker = 'x',markersize=2.0,linewidth=0.0,alpha=0.002)

out_mat_fold = 1

ax1.plot(mean_BSL_gradient_bulk,t_vals[w_d13C_bulk],color='tomato',linewidth=1.0)
ax1.plot([0,0],[0,0],color='tomato',linewidth=1.0,label="NJ $\Delta$$\delta$$^{13}$C$_{bulk}$")
ax1.plot(mean_BSL_gradient_bulk + (std_BSL_gradient_bulk),t_vals[w_d13C_bulk],color='tomato',linestyle='--',linewidth=0.75)
ax1.plot(mean_BSL_gradient_bulk - (std_BSL_gradient_bulk),t_vals[w_d13C_bulk],color='tomato',linestyle='--',linewidth=0.75)
ax1.plot(mean_BSL_gradient_bulk + (2. * std_BSL_gradient_bulk),t_vals[w_d13C_bulk],color='tomato',linestyle=':',linewidth=0.5)
ax1.plot(mean_BSL_gradient_bulk - (2. * std_BSL_gradient_bulk),t_vals[w_d13C_bulk],color='tomato',linestyle=':',linewidth=0.5)

ax1.set_yticks(numpy.arange(51.,60.,0.25), minor=True)

ax1.set_ylim(60,51)


ax1.set_title("MAP3A/B bulk")
ax1.set_xlabel("$\delta$$^{13}$C/Time unit")

ax1.legend()
ax1.set_ylabel("Depth (ft.)")
ax1 = plt.subplot(3,4,6)
###

out_matrix_BSL = numpy.zeros((len(output_matrix_6[:,0]),len(w_d13C_org)))
out_matrix2_BSL = numpy.zeros((len(output_matrix_6[:,0]),len(w_d13C_org)))

for n in range(0,len(output_matrix_6[:,0])):
	out_matrix_BSL[n,:] = numpy.gradient(output_matrix_6[n,w_d13C_org]*-1.,t_vals[w_d13C_org])
	out_matrix2_BSL[n,:] = numpy.gradient(out_matrix_BSL[n,:],t_vals[w_d13C_org])
	
mean_BSL_gradient_org = numpy.nanmean(out_matrix_BSL,axis=0)
std_BSL_gradient_org = numpy.nanstd(out_matrix_BSL,axis=0)

mean_BSL_gradient_org2 = numpy.nanmean(out_matrix2_BSL,axis=0)
std_BSL_gradient_org2 = numpy.nanstd(out_matrix2_BSL,axis=0)

MC_iterations = 0
for n in range(0,MC_iterations):
	
	val2 = numpy.random.normal(loc = mean_BSL_gradient_org, scale = std_BSL_gradient_org)
	
	ax1.plot((val2),t_vals[w_d13C_org],color='forestgreen',marker = 'x',markersize=2.0,linewidth=0.0,alpha=0.002)

out_mat_fold = 1

ax1.plot(mean_BSL_gradient_org,t_vals[w_d13C_org],color='forestgreen',linewidth=1.0)
ax1.plot([0,0],[0,0],color='forestgreen',linewidth=1.0,label="NJ $\Delta$$\delta$$^{13}$C$_{org}$")
ax1.plot(mean_BSL_gradient_org + (std_BSL_gradient_org),t_vals[w_d13C_org],color='forestgreen',linestyle='--',linewidth=0.75)
ax1.plot(mean_BSL_gradient_org - (std_BSL_gradient_org),t_vals[w_d13C_org],color='forestgreen',linestyle='--',linewidth=0.75)
ax1.plot(mean_BSL_gradient_org + (2. * std_BSL_gradient_org),t_vals[w_d13C_org],color='forestgreen',linestyle=':',linewidth=0.5)
ax1.plot(mean_BSL_gradient_org - (2. * std_BSL_gradient_org),t_vals[w_d13C_org],color='forestgreen',linestyle=':',linewidth=0.5)

ax1.set_yticks(numpy.arange(51.,60.,0.25), minor=True)

ax1.set_ylim(60,51)

ax1.set_title("MAP3A/B organic")
ax1.set_xlabel("$\delta$$^{13}$C/Time unit")

ax1.legend()

ax1 = plt.subplot(3,4,7)

out_matrix_BSL = numpy.zeros((len(output_matrix_6[:,0]),len(w_d13C_PF)))
out_matrix2_BSL = numpy.zeros((len(output_matrix_6[:,0]),len(w_d13C_PF)))

for n in range(0,len(output_matrix_6[:,0])):
	out_matrix_BSL[n,:] = numpy.gradient(output_matrix_6[n,w_d13C_PF]*-1.,t_vals[w_d13C_PF])
	out_matrix2_BSL[n,:] = numpy.gradient(out_matrix_BSL[n,:],t_vals[w_d13C_PF])
	
mean_BSL_gradient_PF = numpy.nanmean(out_matrix_BSL,axis=0)
std_BSL_gradient_PF = numpy.nanstd(out_matrix_BSL,axis=0)

mean_BSL_gradient_PF2 = numpy.nanmean(out_matrix2_BSL,axis=0)
std_BSL_gradient_PF2 = numpy.nanstd(out_matrix2_BSL,axis=0)

MC_iterations = 0
for n in range(0,MC_iterations):
	
	val2 = numpy.random.normal(loc = mean_BSL_gradient_PF, scale = std_BSL_gradient_PF)
	
	ax1.plot((val2),t_vals[w_d13C_PF],color='dodgerblue',marker = 'x',markersize=2.0,linewidth=0.0,alpha=0.002)

out_mat_fold = 1

ax1.plot(mean_BSL_gradient_PF,t_vals[w_d13C_PF],color='dodgerblue',linewidth=1.0)
ax1.plot([0,0],[0,0],color='dodgerblue',linewidth=1.0,label="NJ $\Delta$$\delta$$^{13}$C$_{PF}$")
ax1.plot(mean_BSL_gradient_PF + (std_BSL_gradient_PF),t_vals[w_d13C_PF],color='dodgerblue',linestyle='--',linewidth=0.75)
ax1.plot(mean_BSL_gradient_PF - (std_BSL_gradient_PF),t_vals[w_d13C_PF],color='dodgerblue',linestyle='--',linewidth=0.75)
ax1.plot(mean_BSL_gradient_PF + (2. * std_BSL_gradient_PF),t_vals[w_d13C_PF],color='dodgerblue',linestyle=':',linewidth=0.5)
ax1.plot(mean_BSL_gradient_PF - (2. * std_BSL_gradient_PF),t_vals[w_d13C_PF],color='dodgerblue',linestyle=':',linewidth=0.5)

ax1.set_yticks(numpy.arange(51.,60.,0.25), minor=True)

ax1.set_ylim(60,51)

out_matrix_BSL = numpy.zeros((len(output_matrix_6[:,0]),len(w_d13C_BF)))
out_matrix2_BSL = numpy.zeros((len(output_matrix_6[:,0]),len(w_d13C_BF)))

for n in range(0,len(output_matrix_6[:,0])):
	out_matrix_BSL[n,:] = numpy.gradient(output_matrix_6[n,w_d13C_BF]*-1.,t_vals[w_d13C_BF])
	out_matrix2_BSL[n,:] = numpy.gradient(out_matrix_BSL[n,:],t_vals[w_d13C_BF])
	
mean_BSL_gradient_BF = numpy.nanmean(out_matrix_BSL,axis=0)
std_BSL_gradient_BF = numpy.nanstd(out_matrix_BSL,axis=0)

mean_BSL_gradient_BF2 = numpy.nanmean(out_matrix2_BSL,axis=0)
std_BSL_gradient_BF2 = numpy.nanstd(out_matrix2_BSL,axis=0)

MC_iterations = 0
for n in range(0,MC_iterations):
	
	val2 = numpy.random.normal(loc = mean_BSL_gradient_BF, scale = std_BSL_gradient_BF)
	
	ax1.plot((val2),t_vals[w_d13C_BF],color='indigo',marker = 'x',markersize=2.0,linewidth=0.0,alpha=0.002)

out_mat_fold = 1

ax1.plot(mean_BSL_gradient_BF,t_vals[w_d13C_BF],color='indigo',linewidth=1.0)
ax1.plot([0,0],[0,0],color='indigo',linewidth=1.0,label="NJ $\Delta$$\delta$$^{13}$C$_{BF}$")

ax1.plot(mean_BSL_gradient_BF + (std_BSL_gradient_BF),t_vals[w_d13C_BF],color='indigo',linestyle='--',linewidth=0.75)
ax1.plot(mean_BSL_gradient_BF - (std_BSL_gradient_BF),t_vals[w_d13C_BF],color='indigo',linestyle='--',linewidth=0.75)
ax1.plot(mean_BSL_gradient_BF + (2. * std_BSL_gradient_BF),t_vals[w_d13C_BF],color='indigo',linestyle=':',linewidth=0.5)
ax1.plot(mean_BSL_gradient_BF - (2. * std_BSL_gradient_BF),t_vals[w_d13C_BF],color='indigo',linestyle=':',linewidth=0.5)

ax1.set_yticks(numpy.arange(51.,60.,0.25), minor=True)

ax1.set_ylim(60,51)

ax1.set_title("MAP3A/B forams")
ax1.set_xlabel("$\delta$$^{13}$C/Time unit")

ax1.legend()


ax1 = plt.subplot(3,4,8)
###

out_matrix_BSL = numpy.zeros((len(output_matrix_6[:,0]),len(w_TEX86)))
out_matrix2_BSL = numpy.zeros((len(output_matrix_6[:,0]),len(w_TEX86)))

for n in range(0,len(output_matrix_6[:,0])):
	out_matrix_BSL[n,:] = numpy.gradient(output_matrix_6[n,w_TEX86]*-1.,t_vals[w_TEX86])
	out_matrix2_BSL[n,:] = numpy.gradient(out_matrix_BSL[n,:],t_vals[w_TEX86])
	
mean_BSL_gradient_tex = numpy.nanmean(out_matrix_BSL,axis=0)
std_BSL_gradient_tex = numpy.nanstd(out_matrix_BSL,axis=0)

mean_BSL_gradient_tex2 = numpy.nanmean(out_matrix2_BSL,axis=0)
std_BSL_gradient_tex2 = numpy.nanstd(out_matrix2_BSL,axis=0)

MC_iterations = 0
for n in range(0,MC_iterations):
	
	val2 = numpy.random.normal(loc = mean_BSL_gradient_tex, scale = std_BSL_gradient_tex)
	
	ax1.plot((val2),t_vals[w_TEX86],color='mediumvioletred',marker = 'x',markersize=2.0,linewidth=0.0,alpha=0.002)

out_mat_fold = 1

ax1.plot(mean_BSL_gradient_tex,t_vals[w_TEX86],color='mediumvioletred',linewidth=1.0)
ax1.plot([0,0],[0,0],color='mediumvioletred',linewidth=1.0,label="NJ $\Delta$TEX86")
ax1.plot(mean_BSL_gradient_tex + (std_BSL_gradient_tex),t_vals[w_TEX86],color='mediumvioletred',linestyle='--',linewidth=0.75)
ax1.plot(mean_BSL_gradient_tex - (std_BSL_gradient_tex),t_vals[w_TEX86],color='mediumvioletred',linestyle='--',linewidth=0.75)
ax1.plot(mean_BSL_gradient_tex + (2. * std_BSL_gradient_tex),t_vals[w_TEX86],color='mediumvioletred',linestyle=':',linewidth=0.5)
ax1.plot(mean_BSL_gradient_tex - (2. * std_BSL_gradient_tex),t_vals[w_TEX86],color='mediumvioletred',linestyle=':',linewidth=0.5)

ax1.set_yticks(numpy.arange(51.,60.,0.25), minor=True)

ax1.set_ylim(60,51)

ax1.set_title("MAP3A/B temperature")
ax1.set_xlabel("$\degree$C/Time unit")

ax1.legend()

####

ax1 = plt.subplot(3,4,9)
###

out_matrix_BSL = numpy.zeros((len(output_matrix_6[:,0]),len(w_d13C_bulk)))
out_matrix2_BSL = numpy.zeros((len(output_matrix_6[:,0]),len(w_d13C_bulk)))

for n in range(0,len(output_matrix_6[:,0])):
	out_matrix_BSL[n,:] = numpy.gradient(output_matrix_6[n,w_d13C_bulk]*-1.,t_vals[w_d13C_bulk])
	out_matrix2_BSL[n,:] = numpy.gradient(out_matrix_BSL[n,:],t_vals[w_d13C_bulk])
	
mean_BSL_gradient_bulk = numpy.nanmean(out_matrix_BSL,axis=0)
std_BSL_gradient_bulk = numpy.nanstd(out_matrix_BSL,axis=0)

mean_BSL_gradient_bulk2 = numpy.nanmean(out_matrix2_BSL,axis=0)
std_BSL_gradient_bulk2 = numpy.nanstd(out_matrix2_BSL,axis=0)

MC_iterations = 0
for n in range(0,MC_iterations):
	
	val2 = numpy.random.normal(loc = mean_BSL_gradient_bulk2, scale = std_BSL_gradient_bulk2)
	
	ax1.plot((val2),t_vals[w_d13C_bulk],color='tomato',marker = 'x',markersize=2.0,linewidth=0.0,alpha=0.002)

out_mat_fold = 1

ax1.plot(mean_BSL_gradient_bulk2,t_vals[w_d13C_bulk],color='tomato',linewidth=1.0)
ax1.plot([0,0],[0,0],color='tomato',linewidth=1.0,label="NJ $\Delta$$^2$$\delta$$^{13}$C$_{bulk}$")
ax1.plot(mean_BSL_gradient_bulk2 + (std_BSL_gradient_bulk2),t_vals[w_d13C_bulk],color='tomato',linestyle='--',linewidth=0.75)
ax1.plot(mean_BSL_gradient_bulk2 - (std_BSL_gradient_bulk2),t_vals[w_d13C_bulk],color='tomato',linestyle='--',linewidth=0.75)
ax1.plot(mean_BSL_gradient_bulk2 + (2. * std_BSL_gradient_bulk2),t_vals[w_d13C_bulk],color='tomato',linestyle=':',linewidth=0.5)
ax1.plot(mean_BSL_gradient_bulk2 - (2. * std_BSL_gradient_bulk2),t_vals[w_d13C_bulk],color='tomato',linestyle=':',linewidth=0.5)

ax1.set_yticks(numpy.arange(51.,60.,0.25), minor=True)

ax1.set_ylim(60,51)

ax1.set_title("MAP3A/B bulk")
ax1.set_xlabel("$\delta$$^{13}$C/Time unit$^2$")

ax1.legend()
ax1.set_ylabel("Depth (ft.)")
ax1 = plt.subplot(3,4,10)
###

out_matrix_BSL = numpy.zeros((len(output_matrix_6[:,0]),len(w_d13C_org)))
out_matrix2_BSL = numpy.zeros((len(output_matrix_6[:,0]),len(w_d13C_org)))

for n in range(0,len(output_matrix_6[:,0])):
	out_matrix_BSL[n,:] = numpy.gradient(output_matrix_6[n,w_d13C_org]*-1.,t_vals[w_d13C_org])
	out_matrix2_BSL[n,:] = numpy.gradient(out_matrix_BSL[n,:],t_vals[w_d13C_org])
	
mean_BSL_gradient_org = numpy.nanmean(out_matrix_BSL,axis=0)
std_BSL_gradient_org = numpy.nanstd(out_matrix_BSL,axis=0)

mean_BSL_gradient_org2 = numpy.nanmean(out_matrix2_BSL,axis=0)
std_BSL_gradient_org2 = numpy.nanstd(out_matrix2_BSL,axis=0)

MC_iterations = 0
for n in range(0,MC_iterations):
	
	val2 = numpy.random.normal(loc = mean_BSL_gradient_org2, scale = std_BSL_gradient_org2)
	
	ax1.plot((val2),t_vals[w_d13C_org],color='forestgreen',marker = 'x',markersize=2.0,linewidth=0.0,alpha=0.002)

out_mat_fold = 1

ax1.plot(mean_BSL_gradient_org2,t_vals[w_d13C_org],color='forestgreen',linewidth=1.0)
ax1.plot([0,0],[0,0],color='forestgreen',linewidth=1.0,label="NJ $\Delta$$^2$$\delta$$^{13}$C$_{org}$")
ax1.plot(mean_BSL_gradient_org2 + (std_BSL_gradient_org2),t_vals[w_d13C_org],color='forestgreen',linestyle='--',linewidth=0.75)
ax1.plot(mean_BSL_gradient_org2 - (std_BSL_gradient_org2),t_vals[w_d13C_org],color='forestgreen',linestyle='--',linewidth=0.75)
ax1.plot(mean_BSL_gradient_org2 + (2. * std_BSL_gradient_org2),t_vals[w_d13C_org],color='forestgreen',linestyle=':',linewidth=0.5)
ax1.plot(mean_BSL_gradient_org2 - (2. * std_BSL_gradient_org2),t_vals[w_d13C_org],color='forestgreen',linestyle=':',linewidth=0.5)

ax1.set_yticks(numpy.arange(51.,60.,0.25), minor=True)

ax1.set_ylim(60,51)


ax1.set_title("MAP3A/B organic")
ax1.set_xlabel("$\delta$$^{13}$C/Time unit$^2$")

ax1.legend()

ax1 = plt.subplot(3,4,11)
###

out_matrix_BSL = numpy.zeros((len(output_matrix_6[:,0]),len(w_d13C_PF)))
out_matrix2_BSL = numpy.zeros((len(output_matrix_6[:,0]),len(w_d13C_PF)))

for n in range(0,len(output_matrix_6[:,0])):
	out_matrix_BSL[n,:] = numpy.gradient(output_matrix_6[n,w_d13C_PF]*-1.,t_vals[w_d13C_PF])
	out_matrix2_BSL[n,:] = numpy.gradient(out_matrix_BSL[n,:],t_vals[w_d13C_PF])
	
mean_BSL_gradient_PF = numpy.nanmean(out_matrix_BSL,axis=0)
std_BSL_gradient_PF = numpy.nanstd(out_matrix_BSL,axis=0)

mean_BSL_gradient_PF2 = numpy.nanmean(out_matrix2_BSL,axis=0)
std_BSL_gradient_PF2 = numpy.nanstd(out_matrix2_BSL,axis=0)

MC_iterations = 0
for n in range(0,MC_iterations):
	
	val2 = numpy.random.normal(loc = mean_BSL_gradient_PF2, scale = std_BSL_gradient_PF2)
	
	ax1.plot((val2),t_vals[w_d13C_PF],color='dodgerblue',marker = 'x',markersize=2.0,linewidth=0.0,alpha=0.002)

out_mat_fold = 1

ax1.plot(mean_BSL_gradient_PF2,t_vals[w_d13C_PF],color='dodgerblue',linewidth=1.0)
ax1.plot([0,0],[0,0],color='dodgerblue',linewidth=1.0,label="NJ $\Delta$$^2$$\delta$$^{13}$C$_{PF}$")
ax1.plot(mean_BSL_gradient_PF2 + (std_BSL_gradient_PF2),t_vals[w_d13C_PF],color='dodgerblue',linestyle='--',linewidth=0.75)
ax1.plot(mean_BSL_gradient_PF2 - (std_BSL_gradient_PF2),t_vals[w_d13C_PF],color='dodgerblue',linestyle='--',linewidth=0.75)
ax1.plot(mean_BSL_gradient_PF2 + (2. * std_BSL_gradient_PF2),t_vals[w_d13C_PF],color='dodgerblue',linestyle=':',linewidth=0.5)
ax1.plot(mean_BSL_gradient_PF2 - (2. * std_BSL_gradient_PF2),t_vals[w_d13C_PF],color='dodgerblue',linestyle=':',linewidth=0.5)

out_matrix_BSL = numpy.zeros((len(output_matrix_6[:,0]),len(w_d13C_BF)))
out_matrix2_BSL = numpy.zeros((len(output_matrix_6[:,0]),len(w_d13C_BF)))

for n in range(0,len(output_matrix_6[:,0])):
	out_matrix_BSL[n,:] = numpy.gradient(output_matrix_6[n,w_d13C_BF]*-1.,t_vals[w_d13C_BF])
	out_matrix2_BSL[n,:] = numpy.gradient(out_matrix_BSL[n,:],t_vals[w_d13C_BF])
	
mean_BSL_gradient_BF = numpy.nanmean(out_matrix_BSL,axis=0)
std_BSL_gradient_BF = numpy.nanstd(out_matrix_BSL,axis=0)

mean_BSL_gradient_BF2 = numpy.nanmean(out_matrix2_BSL,axis=0)
std_BSL_gradient_BF2 = numpy.nanstd(out_matrix2_BSL,axis=0)

MC_iterations = 0
for n in range(0,MC_iterations):
	
	val2 = numpy.random.normal(loc = mean_BSL_gradient_BF2, scale = std_BSL_gradient_BF2)
	
	ax1.plot((val2),t_vals[w_d13C_BF],color='indigo',marker = 'x',markersize=2.0,linewidth=0.0,alpha=0.002)

out_mat_fold = 1

ax1.plot(mean_BSL_gradient_BF2,t_vals[w_d13C_BF],color='indigo',linewidth=1.0)
ax1.plot([0,0],[0,0],color='indigo',linewidth=1.0,label="NJ $\Delta$$^2$$\delta$$^{13}$C$_{BF}$")
ax1.plot(mean_BSL_gradient_BF2 + (std_BSL_gradient_BF2),t_vals[w_d13C_BF],color='indigo',linestyle='--',linewidth=0.75)
ax1.plot(mean_BSL_gradient_BF2 - (std_BSL_gradient_BF2),t_vals[w_d13C_BF],color='indigo',linestyle='--',linewidth=0.75)
ax1.plot(mean_BSL_gradient_BF2 + (2. * std_BSL_gradient_BF2),t_vals[w_d13C_BF],color='indigo',linestyle=':',linewidth=0.5)
ax1.plot(mean_BSL_gradient_BF2 - (2. * std_BSL_gradient_BF2),t_vals[w_d13C_BF],color='indigo',linestyle=':',linewidth=0.5)

ax1.set_yticks(numpy.arange(51.,60.,0.25), minor=True)

ax1.set_ylim(60,51)

ax1.set_title("MAP3A/B forams")
ax1.set_xlabel("$\delta$$^{13}$C/Time unit$^2$")

ax1.legend()

ax1 = plt.subplot(3,4,12)
###

out_matrix_BSL = numpy.zeros((len(output_matrix_6[:,0]),len(w_TEX86)))
out_matrix2_BSL = numpy.zeros((len(output_matrix_6[:,0]),len(w_TEX86)))

for n in range(0,len(output_matrix_6[:,0])):
	out_matrix_BSL[n,:] = numpy.gradient(output_matrix_6[n,w_TEX86]*-1.,t_vals[w_TEX86])
	out_matrix2_BSL[n,:] = numpy.gradient(out_matrix_BSL[n,:],t_vals[w_TEX86])
	
mean_BSL_gradient_tex = numpy.nanmean(out_matrix_BSL,axis=0)
std_BSL_gradient_tex = numpy.nanstd(out_matrix_BSL,axis=0)

mean_BSL_gradient_tex2 = numpy.nanmean(out_matrix2_BSL,axis=0)
std_BSL_gradient_tex2 = numpy.nanstd(out_matrix2_BSL,axis=0)

MC_iterations = 0
for n in range(0,MC_iterations):
	
	val2 = numpy.random.normal(loc = mean_BSL_gradient_tex2, scale = std_BSL_gradient_tex2)
	
	ax1.plot((val2),t_vals[w_TEX86],color='mediumvioletred',marker = 'x',markersize=2.0,linewidth=0.0,alpha=0.002)

out_mat_fold = 1

ax1.plot(mean_BSL_gradient_tex2,t_vals[w_TEX86],color='mediumvioletred',linewidth=1.0)
ax1.plot([0,0],[0,0],color='mediumvioletred',linewidth=1.0,label="NJ $\Delta$$^2$TEX86")
ax1.plot(mean_BSL_gradient_tex2 + (std_BSL_gradient_tex2),t_vals[w_TEX86],color='mediumvioletred',linestyle='--',linewidth=0.75)
ax1.plot(mean_BSL_gradient_tex2 - (std_BSL_gradient_tex2),t_vals[w_TEX86],color='mediumvioletred',linestyle='--',linewidth=0.75)
ax1.plot(mean_BSL_gradient_tex2 + (2. * std_BSL_gradient_tex2),t_vals[w_TEX86],color='mediumvioletred',linestyle=':',linewidth=0.5)
ax1.plot(mean_BSL_gradient_tex2 - (2. * std_BSL_gradient_tex2),t_vals[w_TEX86],color='mediumvioletred',linestyle=':',linewidth=0.5)

ax1.set_yticks(numpy.arange(51.,60.,0.25), minor=True)

ax1.set_ylim(60,51)

ax1.set_title("MAP3A/B temperature")
ax1.set_xlabel("$\degree$C/Time unit$^2$")

ax1.legend()

plt.tight_layout()

pltname = wkspc +  'MAP3AB_d13C_Tex86_derivative_20230301.png'

plt.savefig(pltname, dpi = 300)