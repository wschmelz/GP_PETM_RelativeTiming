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

from mpl_toolkits.axes_grid.anchored_artists import AnchoredText

import glob

print (str(__file__))

print (time.strftime("%H:%M:%S"))

t1 = float(time.time())

backslash = '\\'

wkspc = str(os.getcwd()).replace(backslash,"/") + "/"
wkspc_data = wkspc + '01_Data/'

gpsdata = glob.glob(wkspc_data + "*.csv")

print (wkspc_data)

for n in range(0,len(gpsdata)):
	SL_table = gpsdata[n]
	SL_data = numpy.genfromtxt(SL_table,delimiter=',')
	
	print (SL_table.split("\\")[1][4:6])
	print (SL_table.split("\\")[1][8])
	
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
		w1 = numpy.where((SL_data_2[:,0]>48.07) & (SL_data_2[:,0]<65.))[0]	
	
	if float(SL_table.split("\\")[1][4:6]) >= 6.:
		w1 = numpy.where((SL_data_2[:,0]>=-188.757480-60.) & (SL_data_2[:,0]<=156.881250+60.))[0]			
		
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
		
		site_matrix_depth[n,:] = 1138.6,1158.2,1172.5,1184.8
		
		recovery_BR = site_matrix_depth[n,0]
		core_BR = site_matrix_depth[n,1]
		base_BR = site_matrix_depth[n,3]
		CIE_BR = site_matrix_depth[n,2]
		high_BR = CIE_BR - 51.5
		low_BR = CIE_BR + 12.64				
		
	if n == 1.:
		
		w1 = numpy.where((SL_data_combined[:,3] == n))[0]
		site_locations.append(w1,)
		
		site_matrix_depth[n,:] = 846.7,865.1,898.4,910.75
		
		recovery_MI = site_matrix_depth[n,0]
		core_MI = site_matrix_depth[n,1]
		base_MI = site_matrix_depth[n,3]
		CIE_MI = site_matrix_depth[n,2]
		high_MI = CIE_MI - 51.5
		low_MI = CIE_MI + 12.64				
		
	if n == 2.:
		
		w1 = numpy.where((SL_data_combined[:,3] == n))[0]
		site_locations.append(w1,)
		
		site_matrix_depth[n,:] = 521.86,542.64,563.08,574.74
		
		recovery_AN = site_matrix_depth[n,0]
		core_AN = site_matrix_depth[n,1]
		base_AN = site_matrix_depth[n,3]
		CIE_AN = site_matrix_depth[n,2]
		high_AN = CIE_AN - 51.5
		low_AN = CIE_AN + 12.64		
		
	if n == 3.:

		w1 = numpy.where((SL_data_combined[:,3] == n))[0]
		site_locations.append(w1,)
		
		site_matrix_depth[n,:] = 317.441,331.42,366.71,378.23
		
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
		
		depth_mod = 76.8916667 * sed_rate1
		depth_mod2 = (156.881250-76.8916667) * sed_rate1		
		
		site_matrix_depth[n,:] = (57.05-depth_mod) - depth_mod2,57.05-depth_mod,57.05,69.302		
		
		recovery_MAP3A = site_matrix_depth[n,0]
		core_MAP3A = site_matrix_depth[n,1]
		base_MAP3A = site_matrix_depth[n,3]
		CIE_MAP3A = 57.05
		high_MAP3A = CIE_MAP3A - 51.5
		low_MAP3A = CIE_MAP3A + 12.64

site_matrix_age = numpy.zeros((5,4))

CIE_time = 0.0

recovery_time = 156.881250
recovery_time_err = 0.0
body_time = 76.8916667
body_time_err = 0.0
CIE_time_2 = 0.0
CIE_time_2_err = 0.00
onset_time = -188.757480
onset_time_err = 0.0

for n in range(0,5):

	site_matrix_age[n,:] = 156.881250,76.8916667,4.79912332e-09,-188.757480

output_matrix_4_1 = wkspc + 'MC_probability_output_4.csv'	
output_matrix_4 = numpy.genfromtxt(output_matrix_4_1,delimiter=',')

w1 = numpy.where(numpy.mean(output_matrix_4,axis=1) ==0.0)[0]

output_matrix_4 = numpy.delete(output_matrix_4,w1,axis=0)

mean_timing = (numpy.nanmean(output_matrix_4,axis=0))
std_timing = numpy.nanstd(output_matrix_4,axis=0)

fig= plt.figure(1,figsize=(20.,5.))

w_d13C_BR_c = numpy.where((SL_data_combined[:,3]==0.0))[0]
w_d13C_MI_c = numpy.where((SL_data_combined[:,3]==1.0))[0]
w_d13C_AN_c = numpy.where((SL_data_combined[:,3]==2.0) )[0]
w_d13C_WL_c = numpy.where((SL_data_combined[:,3]==3.0))[0]
w_d13C_MAP3A_c = numpy.where((SL_data_combined[:,3]==4.0))[0]

ax1 = plt.subplot(155)

x_start = numpy.min(SL_data_combined[w_d13C_BR_c,0])
x_end = numpy.max(SL_data_combined[w_d13C_BR_c,0])

new_x_BR = numpy.linspace(x_start,x_end,100)

dt1 = new_x_BR[1] - new_x_BR[0]

old_x = numpy.ndarray.flatten(numpy.repeat(numpy.reshape(SL_data_combined[w_d13C_BR_c,0],(1,-1)),len(output_matrix_4[:,0]),axis=0))
old_y = numpy.ndarray.flatten(output_matrix_4[:,w_d13C_BR_c])

new_y_BR = (loess2D.loess_int(new_x_BR,new_x_BR*0.0,old_x,old_x*0.0,old_y,int(len(old_y)/25),1. * dt1,1000. * dt1)[2])
new_y_2 = (loess2D.loess_int(site_matrix_depth[0,:],site_matrix_depth[0,:]*0.0,old_x,old_x*0.0,old_y,int(len(old_y)/25),1. * dt1,1000. * dt1)[2])

print(new_y_2)

ax1.plot(new_y_BR,new_x_BR,color='k',linewidth=2.5,alpha=1.0)

ax1.plot([new_y_2[0],new_y_2[0]],[100000.,site_matrix_depth[0,0]],color='blue',linewidth=1.0,alpha=1.0,label="'CIE recovery'")
ax1.plot([-1000,new_y_2[0]],[site_matrix_depth[0,0],site_matrix_depth[0,0]],color='blue',linewidth=1.0,alpha=1.0)

ax1.plot([new_y_2[1],new_y_2[1]],[100000.,site_matrix_depth[0,1]],color='orange',linewidth=1.0,alpha=1.0,label="Top 'CIE core'")
ax1.plot([-1000,new_y_2[1]],[site_matrix_depth[0,1],site_matrix_depth[0,1]],color='orange',linewidth=1.0,alpha=1.0)

ax1.plot([new_y_2[2],new_y_2[2]],[100000.,site_matrix_depth[0,2]],color='r',linewidth=1.5,alpha=1.0,label="'CIE onset'")
ax1.plot([-1000,new_y_2[2]],[site_matrix_depth[0,2],site_matrix_depth[0,2]],color='r',linewidth=1.5,alpha=1.0)

ax1.plot([new_y_2[3],new_y_2[3]],[100000.,site_matrix_depth[0,3]],color='gray',linewidth=1.0,alpha=1.0,label="'Pre-CIE'")
ax1.plot([-1000,new_y_2[3]],[site_matrix_depth[0,3],site_matrix_depth[0,3]],color='gray',linewidth=1.0,alpha=1.0)

text_1 = str(round((site_matrix_depth[0,0] - site_matrix_depth[0,1]) / (new_y_2[0] - new_y_2[1]) * -1. * 30.48, 2)) + "cm \nper correlation unit"
ax1.text(new_y_2[0]-200.,site_matrix_depth[0,0] +5.5,text_1,color="blue")	

text_1 = str(round((site_matrix_depth[0,1] - site_matrix_depth[0,2]) / (new_y_2[1] - new_y_2[2]) * -1. * 30.48, 2)) + "cm \nper correlation unit"
ax1.text(new_y_2[1]-200.,site_matrix_depth[0,1] +5.5,text_1,color="orange")	

text_1 = str(round((site_matrix_depth[0,2] - site_matrix_depth[0,3]) / (new_y_2[2] - new_y_2[3]) * -1. * 30.48, 2)) + "cm \nper correlation unit"
ax1.text(new_y_2[2]-200.,site_matrix_depth[0,2] +5.5,text_1,color="r")	

ax1.set_ylim(low_BR+2.5,high_BR-2.5)
ax1.set_xlim((-215),(175))

ax1.grid(alpha=0.0,)

ax1.legend(loc=2)

ax1.set_title("Bass River")
ax1.set_xlabel("Correlation units (rel. to CIE)")

ax1 = plt.subplot(154)

x_start = numpy.min(SL_data_combined[w_d13C_MI_c,0])
x_end = numpy.max(SL_data_combined[w_d13C_MI_c,0])

new_x_MI = numpy.linspace(x_start,x_end,100)

dt1 = new_x_MI[1] - new_x_MI[0]

old_x = numpy.ndarray.flatten(numpy.repeat(numpy.reshape(SL_data_combined[w_d13C_MI_c,0],(1,-1)),len(output_matrix_4[:,0]),axis=0))
old_y = numpy.ndarray.flatten(output_matrix_4[:,w_d13C_MI_c])

new_y_MI = (loess2D.loess_int(new_x_MI,new_x_MI*0.0,old_x,old_x*0.0,old_y,int(len(old_y)/25),1. * dt1,1000. * dt1)[2])
new_y_2 = (loess2D.loess_int(site_matrix_depth[1,:],site_matrix_depth[1,:]*0.0,old_x,old_x*0.0,old_y,int(len(old_y)/25),1. * dt1,1000. * dt1)[2])

ax1.plot(new_y_MI,new_x_MI,color='k',linewidth=2.5,alpha=1.0)

ax1.plot([new_y_2[0],new_y_2[0]],[100000.,site_matrix_depth[1,0]],color='blue',linewidth=1.0,alpha=1.0)
ax1.plot([-1000,new_y_2[0]],[site_matrix_depth[1,0],site_matrix_depth[1,0]],color='blue',linewidth=1.0,alpha=1.0)

ax1.plot([new_y_2[1],new_y_2[1]],[100000.,site_matrix_depth[1,1]],color='orange',linewidth=1.0,alpha=1.0)
ax1.plot([-1000,new_y_2[1]],[site_matrix_depth[1,1],site_matrix_depth[1,1]],color='orange',linewidth=1.0,alpha=1.0)

ax1.plot([new_y_2[2],new_y_2[2]],[100000.,site_matrix_depth[1,2]],color='r',linewidth=1.5,alpha=1.0)
ax1.plot([-1000,new_y_2[2]],[site_matrix_depth[1,2],site_matrix_depth[1,2]],color='r',linewidth=1.5,alpha=1.0)

ax1.plot([new_y_2[3],new_y_2[3]],[100000.,site_matrix_depth[1,3]],color='gray',linewidth=1.0,alpha=1.0)
ax1.plot([-1000,new_y_2[3]],[site_matrix_depth[1,3],site_matrix_depth[1,3]],color='gray',linewidth=1.0,alpha=1.0)

text_1 = str(round((site_matrix_depth[1,0] - site_matrix_depth[1,1]) / (new_y_2[0] - new_y_2[1]) * -1. * 30.48, 2)) + "cm \nper correlation unit"
ax1.text(new_y_2[0]-200.,site_matrix_depth[1,0] +5.5,text_1,color="blue")	

text_1 = str(round((site_matrix_depth[1,1] - site_matrix_depth[1,2]) / (new_y_2[1] - new_y_2[2]) * -1. * 30.48, 2)) + "cm \nper correlation unit"
ax1.text(new_y_2[1]-200.,site_matrix_depth[1,1] +5.5,text_1,color="orange")	

text_1 = str(round((site_matrix_depth[1,2] - site_matrix_depth[1,3]) / (new_y_2[2] - new_y_2[3]) * -1. * 30.48, 2)) + "cm \nper correlation unit"
ax1.text(new_y_2[2]-200.,site_matrix_depth[1,2] +5.5,text_1,color="r")	

ax1.set_ylim(low_MI+2.5,high_MI-2.5)
ax1.set_xlim((-215),(175))

ax1.grid(alpha=0.0,)

ax1.set_title("Millville")
ax1.set_xlabel("Correlation units (rel. to CIE)")

ax1 = plt.subplot(153)

x_start = numpy.min(SL_data_combined[w_d13C_AN_c,0])
x_end = numpy.max(SL_data_combined[w_d13C_AN_c,0])

new_x_AN = numpy.linspace(x_start,x_end,100)

dt1 = new_x_AN[1] - new_x_AN[0]

old_x = numpy.ndarray.flatten(numpy.repeat(numpy.reshape(SL_data_combined[w_d13C_AN_c,0],(1,-1)),len(output_matrix_4[:,0]),axis=0))
old_y = numpy.ndarray.flatten(output_matrix_4[:,w_d13C_AN_c])

new_y_AN = (loess2D.loess_int(new_x_AN,new_x_AN*0.0,old_x,old_x*0.0,old_y,int(len(old_y)/25),1. * dt1,1000. * dt1)[2])
new_y_2 = (loess2D.loess_int(site_matrix_depth[2,:],site_matrix_depth[2,:]*0.0,old_x,old_x*0.0,old_y,int(len(old_y)/25),1. * dt1,1000. * dt1)[2])

print(new_y_2)

ax1.plot(new_y_AN,new_x_AN,color='k',linewidth=2.5,alpha=1.0)

ax1.plot([new_y_2[0],new_y_2[0]],[100000.,site_matrix_depth[2,0]],color='blue',linewidth=1.0,alpha=1.0)
ax1.plot([-1000,new_y_2[0]],[site_matrix_depth[2,0],site_matrix_depth[2,0]],color='blue',linewidth=1.0,alpha=1.0)

ax1.plot([new_y_2[1],new_y_2[1]],[100000.,site_matrix_depth[2,1]],color='orange',linewidth=1.0,alpha=1.0)
ax1.plot([-1000,new_y_2[1]],[site_matrix_depth[2,1],site_matrix_depth[2,1]],color='orange',linewidth=1.0,alpha=1.0)

ax1.plot([new_y_2[2],new_y_2[2]],[100000.,site_matrix_depth[2,2]],color='r',linewidth=1.5,alpha=1.0)
ax1.plot([-1000,new_y_2[2]],[site_matrix_depth[2,2],site_matrix_depth[2,2]],color='r',linewidth=1.5,alpha=1.0)

ax1.plot([new_y_2[3],new_y_2[3]],[100000.,site_matrix_depth[2,3]],color='gray',linewidth=1.0,alpha=1.0)
ax1.plot([-1000,new_y_2[3]],[site_matrix_depth[2,3],site_matrix_depth[2,3]],color='gray',linewidth=1.0,alpha=1.0)

text_1 = str(round((site_matrix_depth[2,0] - site_matrix_depth[2,1]) / (new_y_2[0] - new_y_2[1]) * -1. * 30.48, 2)) + "cm \nper correlation unit"
ax1.text(new_y_2[0]-200.,site_matrix_depth[2,0] +5.5,text_1,color="blue")	

text_1 = str(round((site_matrix_depth[2,1] - site_matrix_depth[2,2]) / (new_y_2[1] - new_y_2[2]) * -1. * 30.48, 2)) + "cm \nper correlation unit"
ax1.text(new_y_2[1]-200.,site_matrix_depth[2,1] +5.5,text_1,color="orange")	

text_1 = str(round((site_matrix_depth[2,2] - site_matrix_depth[2,3]) / (new_y_2[2] - new_y_2[3]) * -1. * 30.48, 2)) + "cm \nper correlation unit"
ax1.text(new_y_2[2]-200.,site_matrix_depth[2,2] +5.5,text_1,color="r")	

ax1.set_ylim(low_AN+2.5,high_AN-2.5)
ax1.set_xlim((-215),(175))

ax1.grid(alpha=0.0,)

ax1.set_title("Ancora")
ax1.set_xlabel("Correlation units (rel. to CIE)")

ax1 = plt.subplot(152)

x_start = numpy.min(SL_data_combined[w_d13C_WL_c,0])
x_end = numpy.max(SL_data_combined[w_d13C_WL_c,0])

new_x_WL = numpy.linspace(x_start,x_end,100)

dt1 = new_x_WL[1] - new_x_WL[0]

old_x = numpy.ndarray.flatten(numpy.repeat(numpy.reshape(SL_data_combined[w_d13C_WL_c,0],(1,-1)),len(output_matrix_4[:,0]),axis=0))
old_y = numpy.ndarray.flatten(output_matrix_4[:,w_d13C_WL_c])

new_y_WL = (loess2D.loess_int(new_x_WL,new_x_WL*0.0,old_x,old_x*0.0,old_y,int(len(old_y)/25),1. * dt1,1000. * dt1)[2])
new_y_2 = (loess2D.loess_int(site_matrix_depth[3,:],site_matrix_depth[3,:]*0.0,old_x,old_x*0.0,old_y,int(len(old_y)/25),1. * dt1,1000. * dt1)[2])

ax1.plot(new_y_WL,new_x_WL,color='k',linewidth=2.5,alpha=1.0)

ax1.plot([new_y_2[0],new_y_2[0]],[100000.,site_matrix_depth[3,0]],color='blue',linewidth=1.0,alpha=1.0)
ax1.plot([-1000,new_y_2[0]],[site_matrix_depth[3,0],site_matrix_depth[3,0]],color='blue',linewidth=1.0,alpha=1.0)

ax1.plot([new_y_2[1],new_y_2[1]],[100000.,site_matrix_depth[3,1]],color='orange',linewidth=1.0,alpha=1.0)
ax1.plot([-1000,new_y_2[1]],[site_matrix_depth[3,1],site_matrix_depth[3,1]],color='orange',linewidth=1.0,alpha=1.0)

ax1.plot([new_y_2[2],new_y_2[2]],[100000.,site_matrix_depth[3,2]],color='r',linewidth=1.5,alpha=1.0)
ax1.plot([-1000,new_y_2[2]],[site_matrix_depth[3,2],site_matrix_depth[3,2]],color='r',linewidth=1.5,alpha=1.0)

ax1.plot([new_y_2[3],new_y_2[3]],[100000.,site_matrix_depth[3,3]],color='gray',linewidth=1.0,alpha=1.0)
ax1.plot([-1000,new_y_2[3]],[site_matrix_depth[3,3],site_matrix_depth[3,3]],color='gray',linewidth=1.0,alpha=1.0)

text_1 = str(round((site_matrix_depth[3,0] - site_matrix_depth[3,1]) / (new_y_2[0] - new_y_2[1]) * -1. * 30.48, 2)) + "cm \nper correlation unit"
ax1.text(new_y_2[0]-200.,site_matrix_depth[3,0] +5.5,text_1,color="blue")	

text_1 = str(round((site_matrix_depth[3,1] - site_matrix_depth[3,2]) / (new_y_2[1] - new_y_2[2]) * -1. * 30.48, 2)) + "cm \nper correlation unit"
ax1.text(new_y_2[1]-200.,site_matrix_depth[3,1] +5.5,text_1,color="orange")	

text_1 = str(round((site_matrix_depth[3,2] - site_matrix_depth[3,3]) / (new_y_2[2] - new_y_2[3]) * -1. * 30.48, 2)) + "cm \nper correlation unit"
ax1.text(new_y_2[2]-200.,site_matrix_depth[3,2] +5.5,text_1,color="r")	

ax1.set_ylim(low_WL+2.5,high_WL-2.5)
ax1.set_xlim((-215),(175))

ax1.grid(alpha=0.0,)

ax1.set_title("Wilson Lake")
ax1.set_xlabel("Correlation units (rel. to CIE)")

ax1 = plt.subplot(151)

x_start = numpy.min(SL_data_combined[w_d13C_MAP3A_c,0])
x_end = numpy.max(SL_data_combined[w_d13C_MAP3A_c,0])

new_x_MAP3A = numpy.linspace(x_start,x_end,100)

dt1 = new_x_MAP3A[1] - new_x_MAP3A[0]

old_x = numpy.ndarray.flatten(numpy.repeat(numpy.reshape(SL_data_combined[w_d13C_MAP3A_c,0],(1,-1)),len(output_matrix_4[:,0]),axis=0))
old_y = numpy.ndarray.flatten(output_matrix_4[:,w_d13C_MAP3A_c])

new_y_MAP3A = (loess2D.loess_int(new_x_MAP3A,new_x_MAP3A*0.0,old_x,old_x*0.0,old_y,int(len(old_y)/25),1. * dt1,1000. * dt1)[2])
new_y_2 = (loess2D.loess_int(site_matrix_depth[4,:],site_matrix_depth[4,:]*0.0,old_x,old_x*0.0,old_y,int(len(old_y)/25),1. * dt1,1000. * dt1)[2])

ax1.plot(new_y_MAP3A,new_x_MAP3A,color='k',linewidth=2.5,alpha=1.0)

ax1.plot([new_y_2[2],new_y_2[2]],[100000.,site_matrix_depth[4,2]],color='r',linewidth=1.5,alpha=1.0,label="'CIE Onset'")
ax1.plot([-1000,new_y_2[2]],[site_matrix_depth[4,2],site_matrix_depth[4,2]],color='r',linewidth=1.5,alpha=1.0)

ax1.plot([new_y_2[3],new_y_2[3]],[100000.,site_matrix_depth[4,3]],color='gray',linewidth=1.0,alpha=1.0)
ax1.plot([-1000,new_y_2[3]],[site_matrix_depth[4,3],site_matrix_depth[4,3]],color='gray',linewidth=1.0,alpha=1.0)

text_1 = str(round((site_matrix_depth[4,1] - site_matrix_depth[4,2]) / (new_y_2[1] - new_y_2[2]) * -1. * 30.48, 2)) + "cm \nper correlation unit"
ax1.text(new_y_2[1]-275.,45.,text_1,color="orange")	

text_1 = str(round((site_matrix_depth[4,2] - site_matrix_depth[4,3]) / (new_y_2[2] - new_y_2[3]) * -1. * 30.48, 2)) + "cm \nper correlation unit"
ax1.text(new_y_2[2]-200.,site_matrix_depth[4,2] +5.5,text_1,color="r")	

ax1.set_ylim(low_MAP3A+2.5,high_MAP3A-2.5)
ax1.set_xlim((-215),(175))

ax1.grid(alpha=0.0,)

ax1.set_title("MAP 3A/B")
ax1.set_xlabel("Correlation units (rel. to CIE)")
ax1.set_ylabel("Depth (ft)")
plt.tight_layout()

pltname = wkspc +  'NJ_time_depth_20230314.png'
plt.savefig(pltname, dpi = 300)
pltname = wkspc +  'NJ_time_depth_20230314.pdf'
plt.savefig(pltname, dpi = 300)

plt.close()

mean_val =  0.0

w_d13C_bulk_BR_c = numpy.where((SL_data_combined[:,3]==0.0) & (SL_data_combined[:,4]==0.0))[0]
w_d13C_org_BR_c = numpy.where((SL_data_combined[:,3]==0.0) & (SL_data_combined[:,4]==1.0))[0]
w_TEX86_BR_c = numpy.where((SL_data_combined[:,3]==0.0) & (SL_data_combined[:,4]==2.0))[0]
w_d13C_PF_BR_c = numpy.where((SL_data_combined[:,3]==0.0) & (SL_data_combined[:,4]==3.0))[0]
w_d13C_BF_BR_c = numpy.where((SL_data_combined[:,3]==0.0) & (SL_data_combined[:,4]==4.0))[0]

w_d13C_bulk_MI_c = numpy.where((SL_data_combined[:,3]==1.0) & (SL_data_combined[:,4]==0.0))[0]
w_d13C_org_MI_c = numpy.where((SL_data_combined[:,3]==1.0) & (SL_data_combined[:,4]==1.0))[0]
w_TEX86_MI_c = numpy.where((SL_data_combined[:,3]==1.0) & (SL_data_combined[:,4]==2.0))[0]
w_d13C_PF_MI_c = numpy.where((SL_data_combined[:,3]==1.0) & (SL_data_combined[:,4]==3.0))[0]
w_d13C_BF_MI_c = numpy.where((SL_data_combined[:,3]==1.0) & (SL_data_combined[:,4]==4.0))[0]

w_d13C_bulk_AN_c = numpy.where((SL_data_combined[:,3]==2.0)  & (SL_data_combined[:,4]==0.0))[0]
w_d13C_org_AN_c = numpy.where((SL_data_combined[:,3]==2.0)  & (SL_data_combined[:,4]==1.0))[0]
w_TEX86_AN_c = numpy.where((SL_data_combined[:,3]==2.0)  & (SL_data_combined[:,4]==2.0))[0]
w_d13C_PF_AN_c = numpy.where((SL_data_combined[:,3]==2.0) & (SL_data_combined[:,4]==3.0))[0]
w_d13C_BF_AN_c = numpy.where((SL_data_combined[:,3]==2.0) & (SL_data_combined[:,4]==4.0))[0]

w_d13C_bulk_WL_c = numpy.where((SL_data_combined[:,3]==3.0) & (SL_data_combined[:,4]==0.0))[0]
w_d13C_org_WL_c = numpy.where((SL_data_combined[:,3]==3.0) & (SL_data_combined[:,4]==1.0))[0]
w_TEX86_WL_c = numpy.where((SL_data_combined[:,3]==3.0) & (SL_data_combined[:,4]==2.0))[0]
w_d13C_PF_WL_c = numpy.where((SL_data_combined[:,3]==3.0) & (SL_data_combined[:,4]==3.0))[0]
w_d13C_BF_WL_c = numpy.where((SL_data_combined[:,3]==3.0) & (SL_data_combined[:,4]==4.0))[0]

w_d13C_bulk_MAP3A_c = numpy.where((SL_data_combined[:,3]==4.0) & (SL_data_combined[:,4]==0.0))[0]
w_d13C_org_MAP3A_c = numpy.where((SL_data_combined[:,3]==4.0) & (SL_data_combined[:,4]==1.0))[0]
w_TEX86_MAP3A_c = numpy.where((SL_data_combined[:,3]==4.0) & (SL_data_combined[:,4]==2.0))[0]
w_d13C_PF_MAP3A_c = numpy.where((SL_data_combined[:,3]==4.0) & (SL_data_combined[:,4]==3.0))[0]
w_d13C_BF_MAP3A_c = numpy.where((SL_data_combined[:,3]==4.0) & (SL_data_combined[:,4]==4.0))[0]

w_d13C_bulk_c = numpy.where((SL_data_combined[:,4]==0.0))[0]
w_d13C_org_c = numpy.where((SL_data_combined[:,4]==1.0))[0]
w_d13C_PF_c = numpy.where((SL_data_combined[:,4]==3.0))[0]
w_d13C_BF_c = numpy.where((SL_data_combined[:,4]==4.0))[0]
w_TEX86_c = numpy.where((SL_data_combined[:,4]==2.0))[0]

w_d13C_glob_c = numpy.where(SL_data_combined[:,4]==5.0)[0]
w_d13C_glob_690  = numpy.where(SL_data_combined[:,3]==6.0)[0]
w_d13C_glob_1262 = numpy.where(SL_data_combined[:,3]==7.0)[0]
w_d13C_glob_1265 = numpy.where(SL_data_combined[:,3]==9.0)[0]
w_d13C_glob_1266 = numpy.where(SL_data_combined[:,3]==10.0)[0]

#####

MCMC_iterations = 0

per_year = 1.

ice_end = 150.

t_vals = numpy.arange(-150.,ice_end+ per_year,per_year)


s_vals1 = t_vals * 0.0

s_vals1_tmp =  (t_vals * 0.0) + 0.
s_vals1 = numpy.append(s_vals1,s_vals1_tmp)

s_vals1_tmp =  (t_vals * 0.0) + 0.
s_vals1 = numpy.append(s_vals1,s_vals1_tmp)

s_vals1_tmp =  (t_vals * 0.0) + 0.
s_vals1 = numpy.append(s_vals1,s_vals1_tmp)

s_vals1_tmp =  (t_vals * 0.0) + 0.
s_vals1 = numpy.append(s_vals1,s_vals1_tmp)

##

s_vals1_tmp =  (t_vals * 0.0) + 1.
s_vals1 = numpy.append(s_vals1,s_vals1_tmp)

s_vals1_tmp =  (t_vals * 0.0) + 1.
s_vals1 = numpy.append(s_vals1,s_vals1_tmp)

s_vals1_tmp =  (t_vals * 0.0) + 1.
s_vals1 = numpy.append(s_vals1,s_vals1_tmp)

s_vals1_tmp =  (t_vals * 0.0) + 1.
s_vals1 = numpy.append(s_vals1,s_vals1_tmp)

s_vals1_tmp =  (t_vals * 0.0) + 1.
s_vals1 = numpy.append(s_vals1,s_vals1_tmp)


##

s_vals1_tmp =  (t_vals * 0.0) + 2.
s_vals1 = numpy.append(s_vals1,s_vals1_tmp)

s_vals1_tmp =  (t_vals * 0.0) + 2.
s_vals1 = numpy.append(s_vals1,s_vals1_tmp)

s_vals1_tmp =  (t_vals * 0.0) + 2.
s_vals1 = numpy.append(s_vals1,s_vals1_tmp)

s_vals1_tmp =  (t_vals * 0.0) + 2.
s_vals1 = numpy.append(s_vals1,s_vals1_tmp)

s_vals1_tmp =  (t_vals * 0.0) + 2.
s_vals1 = numpy.append(s_vals1,s_vals1_tmp)

##

s_vals1_tmp =  (t_vals * 0.0) + 3.
s_vals1 = numpy.append(s_vals1,s_vals1_tmp)

s_vals1_tmp =  (t_vals * 0.0) + 3.
s_vals1 = numpy.append(s_vals1,s_vals1_tmp)

s_vals1_tmp =  (t_vals * 0.0) + 3.
s_vals1 = numpy.append(s_vals1,s_vals1_tmp)

s_vals1_tmp =  (t_vals * 0.0) + 3.
s_vals1 = numpy.append(s_vals1,s_vals1_tmp)

s_vals1_tmp =  (t_vals * 0.0) + 3.
s_vals1 = numpy.append(s_vals1,s_vals1_tmp)

##

s_vals1_tmp =  (t_vals * 0.0) + 4.
s_vals1 = numpy.append(s_vals1,s_vals1_tmp)

s_vals1_tmp =  (t_vals * 0.0) + 4.
s_vals1 = numpy.append(s_vals1,s_vals1_tmp)

s_vals1_tmp =  (t_vals * 0.0) + 4.
s_vals1 = numpy.append(s_vals1,s_vals1_tmp)

s_vals1_tmp =  (t_vals * 0.0) + 4.
s_vals1 = numpy.append(s_vals1,s_vals1_tmp)

s_vals1_tmp =  (t_vals * 0.0) + 4.
s_vals1 = numpy.append(s_vals1,s_vals1_tmp)

##

s_vals1_tmp =  (t_vals * 0.0) + 6.
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

##


s_vals2_tmp =  (t_vals * 0.0) + 5.
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
t_vals = numpy.append(t_vals,t_vals_tmp)
t_vals = numpy.append(t_vals,t_vals_tmp)
t_vals = numpy.append(t_vals,t_vals_tmp)
t_vals = numpy.append(t_vals,t_vals_tmp)

t_vals = numpy.append(t_vals,t_vals_tmp)

t_vals = (t_vals)

w1_tmp = numpy.where((t_vals<-150.) | (t_vals>153.28125))[0]

t_vals = numpy.delete(t_vals,w1_tmp,axis=0)
s_vals1 = numpy.delete(s_vals1,w1_tmp,axis=0)
s_vals2 = numpy.delete(s_vals2,w1_tmp,axis=0)

w_d13C_bulk_BR = numpy.where((s_vals1==0.0) & (s_vals2==0.0))[0]
w_d13C_org_BR = numpy.where((s_vals1==0.0) & (s_vals2==1.0))[0]
w_TEX86_BR = numpy.where((s_vals1==0.0) & (s_vals2==2.0))[0]
w_d13C_PF_BR = numpy.where((s_vals1==0.0) & (s_vals2==3.0))[0]
w_d13C_BF_BR = numpy.where((s_vals1==0.0) & (s_vals2==4.0))[0]


w_d13C_bulk_MI = numpy.where((s_vals1==1.0) & (s_vals2==0.0))[0]
w_d13C_org_MI = numpy.where((s_vals1==1.0) & (s_vals2==1.0))[0]
w_TEX86_MI = numpy.where((s_vals1==1.0) & (s_vals2==2.0))[0]
w_d13C_PF_MI = numpy.where((s_vals1==1.0) & (s_vals2==3.0))[0]
w_d13C_BF_MI = numpy.where((s_vals1==1.0) & (s_vals2==4.0))[0]

w_d13C_bulk_AN = numpy.where((s_vals1==2.0)  & (s_vals2==0.0))[0]
w_d13C_org_AN = numpy.where((s_vals1==2.0)  & (s_vals2==1.0))[0]
w_TEX86_AN = numpy.where((s_vals1==2.0)  & (s_vals2==2.0))[0]
w_d13C_PF_AN = numpy.where((s_vals1==2.0)  & (s_vals2==3.0))[0]
w_d13C_BF_AN = numpy.where((s_vals1==2.0)  & (s_vals2==4.0))[0]

w_d13C_bulk_WL = numpy.where((s_vals1==3.0) & (s_vals2==0.0))[0]
w_d13C_org_WL = numpy.where((s_vals1==3.0) & (s_vals2==1.0))[0]
w_TEX86_WL = numpy.where((s_vals1==3.0) & (s_vals2==2.0))[0]
w_d13C_PF_WL = numpy.where((s_vals1==3.0) & (s_vals2==3.0))[0]
w_d13C_BF_WL = numpy.where((s_vals1==3.0) & (s_vals2==4.0))[0]


w_d13C_bulk_MAP3A = numpy.where((s_vals1==4.0) & (s_vals2==0.0))[0]
w_d13C_org_MAP3A = numpy.where((s_vals1==4.0) & (s_vals2==1.0))[0]
w_TEX86_MAP3A = numpy.where((s_vals1==4.0) & (s_vals2==2.0))[0]
w_d13C_PF_MAP3A = numpy.where((s_vals1==4.0) & (s_vals2==3.0))[0]
w_d13C_BF_MAP3A = numpy.where((s_vals1==4.0) & (s_vals2==4.0))[0]

w_d13C_glob = numpy.where(s_vals1==6.0)[0]
w_d13C_bulk = numpy.where((s_vals1==4.0) & (s_vals2==0.0))[0]
w_d13C_org =  numpy.where((s_vals1==4.0) & (s_vals2==1.0))[0]
w_d13C_PF = numpy.where((s_vals1==4.0) & (s_vals2==3.0))[0]
w_d13C_BF = numpy.where((s_vals1==4.0) & (s_vals2==4.0))[0]
w_TEX86 = numpy.where((s_vals1==4.0) & (s_vals2==2.0))[0]

w_d13C_bulk1 = numpy.where((s_vals2==0.0)&(s_vals1<=5.0))[0]
w_d13C_org1 = numpy.where((s_vals2==1.0)&(s_vals1<=5.0))[0]
w_d13C_PF1 = numpy.where((s_vals2==3.0)&(s_vals1<=5.0))[0]
w_d13C_BF1 = numpy.where((s_vals2==4.0)&(s_vals1<=5.0))[0]
w_d13C_TEX861 = numpy.where((s_vals2==2.0)&(s_vals1<=5.0))[0]

#####

count = 0

output_matrix_1t = wkspc + 'MC_probability_output_1.csv'	
output_matrix = numpy.genfromtxt(output_matrix_1t,delimiter=',')

w1 = numpy.where(numpy.mean(output_matrix,axis=1) ==0.0)[0]

output_matrix = numpy.delete(output_matrix,w1,axis=0)

#local

#BR

mean_d13C_bulk_BR = numpy.nanmean(output_matrix[:,w_d13C_bulk_BR],axis=0)
std_d13C_bulk_BR = numpy.nanstd(output_matrix[:,w_d13C_bulk_BR],axis=0)

mean_d13C_org_BR = numpy.nanmean(output_matrix[:,w_d13C_org_BR],axis=0)
std_d13C_org_BR = numpy.nanstd(output_matrix[:,w_d13C_org_BR],axis=0)

mean_TEX86_BR = numpy.nanmean(output_matrix[:,w_TEX86_BR],axis=0)
std_TEX86_BR = numpy.nanstd(output_matrix[:,w_TEX86_BR],axis=0)

mean_d13C_PF_BR = numpy.nanmean(output_matrix[:,w_d13C_PF_BR],axis=0)
std_d13C_PF_BR = numpy.nanstd(output_matrix[:,w_d13C_PF_BR],axis=0)

mean_d13C_BF_BR = numpy.nanmean(output_matrix[:,w_d13C_BF_BR],axis=0)
std_d13C_BF_BR = numpy.nanstd(output_matrix[:,w_d13C_BF_BR],axis=0)

#MI

mean_d13C_bulk_MI = numpy.nanmean(output_matrix[:,w_d13C_bulk_MI],axis=0)
std_d13C_bulk_MI = numpy.nanstd(output_matrix[:,w_d13C_bulk_MI],axis=0)

mean_d13C_org_MI = numpy.nanmean(output_matrix[:,w_d13C_org_MI],axis=0)
std_d13C_org_MI = numpy.nanstd(output_matrix[:,w_d13C_org_MI],axis=0)

mean_TEX86_MI = numpy.nanmean(output_matrix[:,w_TEX86_MI],axis=0)
std_TEX86_MI = numpy.nanstd(output_matrix[:,w_TEX86_MI],axis=0)

mean_d13C_PF_MI = numpy.nanmean(output_matrix[:,w_d13C_PF_MI],axis=0)
std_d13C_PF_MI = numpy.nanstd(output_matrix[:,w_d13C_PF_MI],axis=0)

mean_d13C_BF_MI = numpy.nanmean(output_matrix[:,w_d13C_BF_MI],axis=0)
std_d13C_BF_MI = numpy.nanstd(output_matrix[:,w_d13C_BF_MI],axis=0)

#AN

mean_d13C_bulk_AN = numpy.nanmean(output_matrix[:,w_d13C_bulk_AN],axis=0)
std_d13C_bulk_AN = numpy.nanstd(output_matrix[:,w_d13C_bulk_AN],axis=0)

mean_d13C_org_AN = numpy.nanmean(output_matrix[:,w_d13C_org_AN],axis=0)
std_d13C_org_AN = numpy.nanstd(output_matrix[:,w_d13C_org_AN],axis=0)

mean_TEX86_AN = numpy.nanmean(output_matrix[:,w_TEX86_AN],axis=0)
std_TEX86_AN = numpy.nanstd(output_matrix[:,w_TEX86_AN],axis=0)

mean_d13C_PF_AN = numpy.nanmean(output_matrix[:,w_d13C_PF_AN],axis=0)
std_d13C_PF_AN = numpy.nanstd(output_matrix[:,w_d13C_PF_AN],axis=0)

mean_d13C_BF_AN = numpy.nanmean(output_matrix[:,w_d13C_BF_AN],axis=0)
std_d13C_BF_AN = numpy.nanstd(output_matrix[:,w_d13C_BF_AN],axis=0)

#WL

mean_d13C_bulk_WL = numpy.nanmean(output_matrix[:,w_d13C_bulk_WL],axis=0)
std_d13C_bulk_WL = numpy.nanstd(output_matrix[:,w_d13C_bulk_WL],axis=0)

mean_d13C_org_WL = numpy.nanmean(output_matrix[:,w_d13C_org_WL],axis=0)
std_d13C_org_WL = numpy.nanstd(output_matrix[:,w_d13C_org_WL],axis=0)

mean_TEX86_WL = numpy.nanmean(output_matrix[:,w_TEX86_WL],axis=0)
std_TEX86_WL = numpy.nanstd(output_matrix[:,w_TEX86_WL],axis=0)

mean_d13C_PF_WL = numpy.nanmean(output_matrix[:,w_d13C_PF_WL],axis=0)
std_d13C_PF_WL = numpy.nanstd(output_matrix[:,w_d13C_PF_WL],axis=0)

mean_d13C_BF_WL = numpy.nanmean(output_matrix[:,w_d13C_BF_WL],axis=0)
std_d13C_BF_WL = numpy.nanstd(output_matrix[:,w_d13C_BF_WL],axis=0)

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

###

mean_NJ_d13C = numpy.nanmean(output_matrix[:,w_d13C_bulk_BR],axis=0)
std_NJ_d13C = numpy.nanstd(output_matrix[:,w_d13C_bulk_BR],axis=0)

###

output_matrix_3_1 = wkspc + 'MC_probability_output_2.csv'	
output_matrix_3 = numpy.genfromtxt(output_matrix_3_1,delimiter=',')

w1 = numpy.where(numpy.mean(output_matrix_3,axis=1) ==0.0)[0]

output_matrix_3 = numpy.delete(output_matrix_3,w1,axis=0)

output_matrix_5_1 = wkspc + 'MC_probability_output_6.csv'	
output_matrix_5 = numpy.genfromtxt(output_matrix_5_1,delimiter=',')

w1 = numpy.where(numpy.mean(output_matrix_5,axis=1) ==0.0)[0]

output_matrix_5 = numpy.delete(output_matrix_5,w1,axis=0)

mean_global = numpy.nanmean(output_matrix_3[:,w_d13C_glob],axis=0)
std_global = numpy.nanstd(output_matrix_3[:,w_d13C_glob],axis=0)

mean_NJ_d13C_bulk = numpy.nanmean(output_matrix_3[:,w_d13C_bulk],axis=0)
std_NJ_d13C_bulk = numpy.nanstd(output_matrix_3[:,w_d13C_bulk],axis=0)

mean_NJ_d13C_org = numpy.nanmean(output_matrix_3[:,w_d13C_org],axis=0)
std_NJ_d13C_org = numpy.nanstd(output_matrix_3[:,w_d13C_org],axis=0)

mean_NJ_d13C_PF = numpy.nanmean(output_matrix_3[:,w_d13C_PF],axis=0)
std_NJ_d13C_PF = numpy.nanstd(output_matrix_3[:,w_d13C_PF],axis=0)

mean_NJ_d13C_BF = numpy.nanmean(output_matrix_3[:,w_d13C_BF],axis=0)
std_NJ_d13C_BF = numpy.nanstd(output_matrix_3[:,w_d13C_BF],axis=0)

mean_NJ_TEX86 = numpy.nanmean(output_matrix_3[:,w_TEX86],axis=0)
std_NJ_TEX86 = numpy.nanstd(output_matrix_3[:,w_TEX86],axis=0)

###

mean_timing = (numpy.nanmean(output_matrix_4,axis=0))
std_timing = numpy.nanstd(output_matrix_4,axis=0)*(1000000./1000.)

fig= plt.figure(2,figsize=(20.,27.5))

ax1 = plt.subplot(551)


MC_iterations = 0
for n in range(0,MC_iterations):
	
	val2 = numpy.random.normal(loc = mean_global, scale = std_global)
	
	ax1.plot((val2),t_vals[w_d13C_glob],color='k',marker = 'x',markersize=2.0,linewidth=0.0,alpha=0.002)

out_mat_fold = 1

ax1.plot(mean_global,t_vals[w_d13C_glob],'k',linewidth=1.0)
ax1.plot([0,0],[0,0],'k',linewidth=1.0,label="$\delta$$^{13}$C$_{NJ}$; $f(t)$")
ax1.plot(mean_global + (std_global),t_vals[w_d13C_glob],'k--',linewidth=0.75)
ax1.plot(mean_global - (std_global),t_vals[w_d13C_glob],'k--',linewidth=0.75)
ax1.plot(mean_global + (2. * std_global),t_vals[w_d13C_glob],'k:',linewidth=0.5)
ax1.plot(mean_global - (2. * std_global),t_vals[w_d13C_glob],'k:',linewidth=0.5)
ax1.set_ylim((-150),(150))
ax1.set_xlim(-5.,3.5)

ax1.grid(alpha=0.0,)
ax1.legend()

ax1.set_title("NJ shelf  $\delta$$^{13}$C")
ax1.set_xlabel("$\delta$$^{13}$C")
ax1.set_ylabel("Correlation units (rel. to CIE)")

ax1 = plt.subplot(552)


MC_iterations = 0
for n in range(0,MC_iterations):
	
	val2 = numpy.random.normal(loc = mean_NJ_d13C_bulk, scale = std_NJ_d13C_bulk)
	
	ax1.plot((val2),t_vals[w_d13C_bulk],color='tomato',marker = 'x',markersize=2.0,linewidth=0.0,alpha=0.002)

out_mat_fold = 1

ax1.plot(SL_data_combined[w_d13C_bulk_c,1],mean_timing[w_d13C_bulk_c],color='tomato',marker='o',linewidth=0.0,markersize=2.0,alpha=0.15,label="NJ $\delta$$^{13}$C$_{bulk}$ data")

ax1.plot(mean_NJ_d13C_bulk,t_vals[w_d13C_bulk],color='tomato',linewidth=1.0)
ax1.plot([0,0],[0,0],color='tomato',linewidth=1.0,label="$\delta$$^{13}$C$_{bulk}$; $f(t)$ + $\Delta$$_{1}(t)$")
ax1.plot(mean_NJ_d13C_bulk + (std_NJ_d13C_bulk),t_vals[w_d13C_bulk],color='tomato',linestyle='--',linewidth=0.75)
ax1.plot(mean_NJ_d13C_bulk - (std_NJ_d13C_bulk),t_vals[w_d13C_bulk],color='tomato',linestyle='--',linewidth=0.75)
ax1.plot(mean_NJ_d13C_bulk + (2. * std_NJ_d13C_bulk),t_vals[w_d13C_bulk],color='tomato',linestyle=':',linewidth=0.5)
ax1.plot(mean_NJ_d13C_bulk - (2. * std_NJ_d13C_bulk),t_vals[w_d13C_bulk],color='tomato',linestyle=':',linewidth=0.5)
ax1.set_ylim((-150),(150))
ax1.set_xlim(-5.,3.5)

ax1.grid(alpha=0.0,)
ax1.legend()

ax1.set_title("NJ shelf $\delta$$^{13}$C$_{bulk}$")
ax1.set_xlabel("$\delta$$^{13}$C")

ax1 = plt.subplot(553)

MC_iterations = 0
for n in range(0,MC_iterations):
	
	val2 = numpy.random.normal(loc = mean_NJ_d13C_org, scale = std_NJ_d13C_org)
	
	ax1.plot((val2),t_vals[w_d13C_org],color='forestgreen',marker = 'x',markersize=2.0,linewidth=0.0,alpha=0.002)

out_mat_fold = 1

ax1.plot(SL_data_combined[w_d13C_org_c,1],mean_timing[w_d13C_org_c],color='forestgreen',marker='o',linewidth=0.0,markersize=2.0,alpha=0.15,label="NJ $\delta$$^{13}$C$_{org}$ data")

ax1.plot(mean_NJ_d13C_org,t_vals[w_d13C_org],color='forestgreen',linewidth=1.0)
ax1.plot([0,0],[0,0],color='forestgreen',linewidth=1.0,label="$\delta$$^{13}$C$_{org}$; $f(t)$ + $\Delta$$_{2}(t)$")
ax1.plot(mean_NJ_d13C_org + (std_NJ_d13C_org),t_vals[w_d13C_org],color='forestgreen',linestyle='--',linewidth=0.75)
ax1.plot(mean_NJ_d13C_org - (std_NJ_d13C_org),t_vals[w_d13C_org],color='forestgreen',linestyle='--',linewidth=0.75)
ax1.plot(mean_NJ_d13C_org + (2. * std_NJ_d13C_org),t_vals[w_d13C_org],color='forestgreen',linestyle=':',linewidth=0.5)
ax1.plot(mean_NJ_d13C_org - (2. * std_NJ_d13C_org),t_vals[w_d13C_org],color='forestgreen',linestyle=':',linewidth=0.5)
ax1.set_ylim((-150),(150))
ax1.set_xlim(-31.,-22.)

ax1.set_title("NJ shelf $\delta$$^{13}$C$_{org}$")
ax1.set_xlabel("$\delta$$^{13}$C")

ax1.grid(alpha=0.0,)
ax1.legend()

ax1 = plt.subplot(554)


MC_iterations = 0
for n in range(0,MC_iterations):
	
	val2 = numpy.random.normal(loc = mean_NJ_d13C_PF, scale = std_NJ_d13C_PF)
	
	ax1.plot((val2),t_vals[w_d13C_PF],color='dodgerblue',marker = 'x',markersize=2.0,linewidth=0.0,alpha=0.002)

out_mat_fold = 1

ax1.plot(SL_data_combined[w_d13C_PF_c,1],mean_timing[w_d13C_PF_c],color='dodgerblue',marker='o',linewidth=0.0,markersize=2.0,alpha=0.15,label="NJ $\delta$$^{13}$C$_{PF}$ data")

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

ax1.plot(SL_data_combined[w_d13C_BF_c,1],mean_timing[w_d13C_BF_c],color='indigo',marker='o',linewidth=0.0,markersize=2.0,alpha=0.15,label="NJ $\delta$$^{13}$C$_{BF}$ data")

ax1.plot(mean_NJ_d13C_BF,t_vals[w_d13C_BF],color='indigo',linewidth=1.0)
ax1.plot([0,0],[0,0],color='indigo',linewidth=1.0,label="$\delta$$^{13}$C$_{BF}$; $f(t)$ + $\Delta$$_{4}(t)$")
ax1.plot(mean_NJ_d13C_BF + (std_NJ_d13C_BF),t_vals[w_d13C_BF],color='indigo',linestyle='--',linewidth=0.75)
ax1.plot(mean_NJ_d13C_BF - (std_NJ_d13C_BF),t_vals[w_d13C_BF],color='indigo',linestyle='--',linewidth=0.75)
ax1.plot(mean_NJ_d13C_BF + (2. * std_NJ_d13C_BF),t_vals[w_d13C_BF],color='indigo',linestyle=':',linewidth=0.5)
ax1.plot(mean_NJ_d13C_BF - (2. * std_NJ_d13C_BF),t_vals[w_d13C_BF],color='indigo',linestyle=':',linewidth=0.5)
ax1.set_ylim((-150),(150))
ax1.set_xlim(-5.5,4.)

ax1.set_title("NJ shelf forams")
ax1.set_xlabel("$\delta$$^{13}$C")

ax1.grid(alpha=0.0,)
ax1.legend()

ax1 = plt.subplot(555)


MC_iterations = 0
for n in range(0,MC_iterations):
	
	val2 = numpy.random.normal(loc = mean_NJ_TEX86, scale = std_NJ_TEX86)
	
	ax1.plot((val2),t_vals[w_TEX86],color='mediumvioletred',marker = 'x',markersize=2.0,linewidth=0.0,alpha=0.002)

out_mat_fold = 1

ax1.plot(SL_data_combined[w_TEX86_c,1],mean_timing[w_TEX86_c],color='mediumvioletred',marker='o',linewidth=0.0,markersize=2.0,alpha=0.15,label="TEX86 data")

ax1.plot(mean_NJ_TEX86,t_vals[w_TEX86],color='mediumvioletred',linewidth=1.0)
ax1.plot([0,0],[0,0],color='mediumvioletred',linewidth=1.0,label="TEX86; $n(t)$")
ax1.plot(mean_NJ_TEX86 + (std_NJ_TEX86),t_vals[w_TEX86],color='mediumvioletred',linestyle='--',linewidth=0.75)
ax1.plot(mean_NJ_TEX86 - (std_NJ_TEX86),t_vals[w_TEX86],color='mediumvioletred',linestyle='--',linewidth=0.75)
ax1.plot(mean_NJ_TEX86 + (2. * std_NJ_TEX86),t_vals[w_TEX86],color='mediumvioletred',linestyle=':',linewidth=0.5)
ax1.plot(mean_NJ_TEX86 - (2. * std_NJ_TEX86),t_vals[w_TEX86],color='mediumvioletred',linestyle=':',linewidth=0.5)
ax1.set_ylim((-150),(150))
ax1.set_xlim(26.75,37.51)

ax1.set_title("NJ shelf temperature")
ax1.set_xlabel("Temperature ($\degree$C)")

ax1.grid(alpha=0.0,)
ax1.legend()

ax1 = plt.subplot(5,5,10)


MC_iterations = 0
for n in range(0,MC_iterations):
	
	val2 = numpy.random.normal(loc = mean_d13C_bulk_BR, scale = std_d13C_bulk_BR)
	
	ax1.plot((val2),t_vals[w_d13C_bulk_BR],color='tomato',marker = 'x',markersize=2.0,linewidth=0.0,alpha=0.002)

out_mat_fold = 1

ax1.plot(SL_data_combined[w_d13C_bulk_BR_c,1],mean_timing[w_d13C_bulk_BR_c],color='tomato',marker='o',linewidth=0.0,markersize=4.0,alpha=0.5,label="BR $\delta$$^{13}$C$_{bulk}$ data")

ax1.plot(mean_d13C_bulk_BR,t_vals[w_d13C_bulk_BR],color='tomato',linewidth=1.0)
ax1.plot([0,0],[0,0],color='tomato',linewidth=1.0,label="BR $\delta$$^{13}$C$_{bulk}$; $g_{1,5}(t)$")
ax1.plot(mean_d13C_bulk_BR + (std_d13C_bulk_BR),t_vals[w_d13C_bulk_BR],color='tomato',linestyle='--',linewidth=0.75)
ax1.plot(mean_d13C_bulk_BR - (std_d13C_bulk_BR),t_vals[w_d13C_bulk_BR],color='tomato',linestyle='--',linewidth=0.75)
ax1.plot(mean_d13C_bulk_BR + (2. * std_d13C_bulk_BR),t_vals[w_d13C_bulk_BR],color='tomato',linestyle=':',linewidth=0.5)
ax1.plot(mean_d13C_bulk_BR - (2. * std_d13C_bulk_BR),t_vals[w_d13C_bulk_BR],color='tomato',linestyle=':',linewidth=0.5)
ax1.set_ylim((-150),(150))
ax1.set_xlim(-5.,3.5)

ax1.set_title("Bass River bulk")
ax1.set_xlabel("$\delta$$^{13}$C")


ax1.grid(alpha=0.0,)
ax1.legend()

ax1 = plt.subplot(5,5,9)


MC_iterations = 0
for n in range(0,MC_iterations):
	
	val2 = numpy.random.normal(loc = mean_d13C_bulk_MI, scale = std_d13C_bulk_MI)
	
	ax1.plot((val2),t_vals[w_d13C_bulk_MI],color='tomato',marker = 'x',markersize=2.0,linewidth=0.0,alpha=0.002)

out_mat_fold = 1

ax1.plot(SL_data_combined[w_d13C_bulk_MI_c,1],mean_timing[w_d13C_bulk_MI_c],color='tomato',marker='o',linewidth=0.0,markersize=4.0,alpha=0.5,label="MI $\delta$$^{13}$C$_{bulk}$ data")

ax1.plot(mean_d13C_bulk_MI,t_vals[w_d13C_bulk_MI],color='tomato',linewidth=1.0)
ax1.plot([0,0],[0,0],color='tomato',linewidth=1.0,label="MI $\delta$$^{13}$C$_{bulk}$; $g_{1,4}(t)$")
ax1.plot(mean_d13C_bulk_MI + (std_d13C_bulk_MI),t_vals[w_d13C_bulk_MI],color='tomato',linestyle='--',linewidth=0.75)
ax1.plot(mean_d13C_bulk_MI - (std_d13C_bulk_MI),t_vals[w_d13C_bulk_MI],color='tomato',linestyle='--',linewidth=0.75)
ax1.plot(mean_d13C_bulk_MI + (2. * std_d13C_bulk_MI),t_vals[w_d13C_bulk_MI],color='tomato',linestyle=':',linewidth=0.5)
ax1.plot(mean_d13C_bulk_MI - (2. * std_d13C_bulk_MI),t_vals[w_d13C_bulk_MI],color='tomato',linestyle=':',linewidth=0.5)
ax1.set_ylim((-150),(150))
ax1.set_xlim(-5.,3.5)

ax1.set_title("Millville bulk")
ax1.set_xlabel("$\delta$$^{13}$C")

ax1.grid(alpha=0.0,)
ax1.legend()

ax1 = plt.subplot(5,5,8)


MC_iterations = 0
for n in range(0,MC_iterations):
	
	val2 = numpy.random.normal(loc = mean_d13C_bulk_AN, scale = std_d13C_bulk_AN)
	
	ax1.plot((val2),t_vals[w_d13C_bulk_AN],color='tomato',marker = 'x',markersize=2.0,linewidth=0.0,alpha=0.002)

out_mat_fold = 1

ax1.plot(SL_data_combined[w_d13C_bulk_AN_c,1],mean_timing[w_d13C_bulk_AN_c],color='tomato',marker='o',linewidth=0.0,markersize=4.0,alpha=0.5,label="AN $\delta$$^{13}$C$_{bulk}$ data")

ax1.plot(mean_d13C_bulk_AN,t_vals[w_d13C_bulk_AN],color='tomato',linewidth=1.0)
ax1.plot([0,0],[0,0],color='tomato',linewidth=1.0,label="AN $\delta$$^{13}$C$_{bulk}$; $g_{1,3}(t)$")
ax1.plot(mean_d13C_bulk_AN + (std_d13C_bulk_AN),t_vals[w_d13C_bulk_AN],color='tomato',linestyle='--',linewidth=0.75)
ax1.plot(mean_d13C_bulk_AN - (std_d13C_bulk_AN),t_vals[w_d13C_bulk_AN],color='tomato',linestyle='--',linewidth=0.75)
ax1.plot(mean_d13C_bulk_AN + (2. * std_d13C_bulk_AN),t_vals[w_d13C_bulk_AN],color='tomato',linestyle=':',linewidth=0.5)
ax1.plot(mean_d13C_bulk_AN - (2. * std_d13C_bulk_AN),t_vals[w_d13C_bulk_AN],color='tomato',linestyle=':',linewidth=0.5)
ax1.set_ylim((-150),(150))
ax1.set_xlim(-5.,3.5)

ax1.set_title("Ancora bulk")
ax1.set_xlabel("$\delta$$^{13}$C")

ax1.grid(alpha=0.0,)
ax1.legend()

ax1 = plt.subplot(5,5,7)


MC_iterations = 0
for n in range(0,MC_iterations):
	
	val2 = numpy.random.normal(loc = mean_d13C_bulk_WL, scale = std_d13C_bulk_WL)
	
	ax1.plot((val2),t_vals[w_d13C_bulk_WL],color='tomato',marker = 'x',markersize=2.0,linewidth=0.0,alpha=0.002)

out_mat_fold = 1

ax1.plot(SL_data_combined[w_d13C_bulk_WL_c,1],mean_timing[w_d13C_bulk_WL_c],color='tomato',marker='o',linewidth=0.0,markersize=4.0,alpha=0.5,label="WL $\delta$$^{13}$C$_{bulk}$ data")

ax1.plot(mean_d13C_bulk_WL,t_vals[w_d13C_bulk_WL],color='tomato',linewidth=1.0)
ax1.plot([0,0],[0,0],color='tomato',linewidth=1.0,label="WL $\delta$$^{13}$C$_{bulk}$; $g_{1,2}(t)$")
ax1.plot(mean_d13C_bulk_WL + (std_d13C_bulk_WL),t_vals[w_d13C_bulk_WL],color='tomato',linestyle='--',linewidth=0.75)
ax1.plot(mean_d13C_bulk_WL - (std_d13C_bulk_WL),t_vals[w_d13C_bulk_WL],color='tomato',linestyle='--',linewidth=0.75)
ax1.plot(mean_d13C_bulk_WL + (2. * std_d13C_bulk_WL),t_vals[w_d13C_bulk_WL],color='tomato',linestyle=':',linewidth=0.5)
ax1.plot(mean_d13C_bulk_WL - (2. * std_d13C_bulk_WL),t_vals[w_d13C_bulk_WL],color='tomato',linestyle=':',linewidth=0.5)
ax1.set_ylim((-150),(150))
ax1.set_xlim(-5.,3.5)

ax1.set_title("Wilson Lake bulk")
ax1.set_xlabel("$\delta$$^{13}$C")

ax1.grid(alpha=0.0,)
ax1.legend()

ax1 = plt.subplot(5,5,6)


MC_iterations = 0
for n in range(0,MC_iterations):
	
	val2 = numpy.random.normal(loc = mean_d13C_bulk_MAP3A, scale = std_d13C_bulk_MAP3A)
	
	ax1.plot((val2),t_vals[w_d13C_bulk_MAP3A],color='tomato',marker = 'x',markersize=2.0,linewidth=0.0,alpha=0.002)

out_mat_fold = 1

ax1.plot(SL_data_combined[w_d13C_bulk_MAP3A_c,1],mean_timing[w_d13C_bulk_MAP3A_c],color='tomato',marker='o',linewidth=0.0,markersize=4.0,alpha=0.5,label="MAP3A/B $\delta$$^{13}$C$_{bulk}$ data")

ax1.plot(mean_d13C_bulk_MAP3A,t_vals[w_d13C_bulk_MAP3A],color='tomato',linewidth=1.0)
ax1.plot([0,0],[0,0],color='tomato',linewidth=1.0,label="MAP3A/B $\delta$$^{13}$C$_{bulk}$; $g_{1,1}(t)$")
ax1.plot(mean_d13C_bulk_MAP3A + (std_d13C_bulk_MAP3A),t_vals[w_d13C_bulk_MAP3A],color='tomato',linestyle='--',linewidth=0.75)
ax1.plot(mean_d13C_bulk_MAP3A - (std_d13C_bulk_MAP3A),t_vals[w_d13C_bulk_MAP3A],color='tomato',linestyle='--',linewidth=0.75)
ax1.plot(mean_d13C_bulk_MAP3A + (2. * std_d13C_bulk_MAP3A),t_vals[w_d13C_bulk_MAP3A],color='tomato',linestyle=':',linewidth=0.5)
ax1.plot(mean_d13C_bulk_MAP3A - (2. * std_d13C_bulk_MAP3A),t_vals[w_d13C_bulk_MAP3A],color='tomato',linestyle=':',linewidth=0.5)
ax1.set_ylim((-150),(150))
ax1.set_xlim(-7.5,1.)

ax1.set_title("MAP3A/B bulk")
ax1.set_xlabel("$\delta$$^{13}$C")
ax1.set_ylabel("Correlation units (rel. to CIE)")
ax1.grid(alpha=0.0,)
ax1.legend(loc=1)

########################
########################
########################

ax1 = plt.subplot(5,5,15)


MC_iterations = 0
for n in range(0,MC_iterations):
	
	val2 = numpy.random.normal(loc = mean_d13C_org_BR, scale = std_d13C_org_BR)
	
	ax1.plot((val2),t_vals[w_d13C_org_BR],color='forestgreen',marker = 'x',markersize=2.0,linewidth=0.0,alpha=0.002)

out_mat_fold = 1

ax1.plot(SL_data_combined[w_d13C_org_BR_c,1],mean_timing[w_d13C_org_BR_c],color='forestgreen',marker='o',linewidth=0.0,markersize=4.0,alpha=0.5,label="BR $\delta$$^{13}$C$_{org}$ data")

ax1.plot(mean_d13C_org_BR,t_vals[w_d13C_org_BR],color='forestgreen',linewidth=1.0)
ax1.plot([0,0],[0,0],color='forestgreen',linewidth=1.0,label="BR $\delta$$^{13}$C$_{org}$; $g_{2,5}(t)$")
ax1.plot(mean_d13C_org_BR + (std_d13C_org_BR),t_vals[w_d13C_org_BR],color='forestgreen',linestyle='--',linewidth=0.75)
ax1.plot(mean_d13C_org_BR - (std_d13C_org_BR),t_vals[w_d13C_org_BR],color='forestgreen',linestyle='--',linewidth=0.75)
ax1.plot(mean_d13C_org_BR + (2. * std_d13C_org_BR),t_vals[w_d13C_org_BR],color='forestgreen',linestyle=':',linewidth=0.5)
ax1.plot(mean_d13C_org_BR - (2. * std_d13C_org_BR),t_vals[w_d13C_org_BR],color='forestgreen',linestyle=':',linewidth=0.5)
ax1.set_ylim((-150),(150))
ax1.set_xlim(-31.,-22.)

ax1.set_title("Bass River organic")
ax1.set_xlabel("$\delta$$^{13}$C")


ax1.grid(alpha=0.0,)
ax1.legend()

ax1 = plt.subplot(5,5,14)


MC_iterations = 0
for n in range(0,MC_iterations):
	
	val2 = numpy.random.normal(loc = mean_d13C_org_MI, scale = std_d13C_org_MI)
	
	ax1.plot((val2),t_vals[w_d13C_org_MI],color='forestgreen',marker = 'x',markersize=2.0,linewidth=0.0,alpha=0.002)

out_mat_fold = 1

ax1.plot(SL_data_combined[w_d13C_org_MI_c,1],mean_timing[w_d13C_org_MI_c],color='forestgreen',marker='o',linewidth=0.0,markersize=4.0,alpha=0.5,label="MI $\delta$$^{13}$C$_{org}$ data")

ax1.plot(mean_d13C_org_MI,t_vals[w_d13C_org_MI],color='forestgreen',linewidth=1.0)
ax1.plot([0,0],[0,0],color='forestgreen',linewidth=1.0,label="MI $\delta$$^{13}$C$_{org}$; $g_{2,4}(t)$")
ax1.plot(mean_d13C_org_MI + (std_d13C_org_MI),t_vals[w_d13C_org_MI],color='forestgreen',linestyle='--',linewidth=0.75)
ax1.plot(mean_d13C_org_MI - (std_d13C_org_MI),t_vals[w_d13C_org_MI],color='forestgreen',linestyle='--',linewidth=0.75)
ax1.plot(mean_d13C_org_MI + (2. * std_d13C_org_MI),t_vals[w_d13C_org_MI],color='forestgreen',linestyle=':',linewidth=0.5)
ax1.plot(mean_d13C_org_MI - (2. * std_d13C_org_MI),t_vals[w_d13C_org_MI],color='forestgreen',linestyle=':',linewidth=0.5)
ax1.set_ylim((-150),(150))
ax1.set_xlim(-31.,-22.)

ax1.set_title("Millville organic")
ax1.set_xlabel("$\delta$$^{13}$C")

ax1.grid(alpha=0.0,)
ax1.legend()

ax1 = plt.subplot(5,5,13)


MC_iterations = 0
for n in range(0,MC_iterations):
	
	val2 = numpy.random.normal(loc = mean_d13C_org_AN, scale = std_d13C_org_AN)
	
	ax1.plot((val2),t_vals[w_d13C_org_AN],color='forestgreen',marker = 'x',markersize=2.0,linewidth=0.0,alpha=0.002)

out_mat_fold = 1

ax1.plot(SL_data_combined[w_d13C_org_AN_c,1],mean_timing[w_d13C_org_AN_c],color='forestgreen',marker='o',linewidth=0.0,markersize=4.0,alpha=0.5,label="AN $\delta$$^{13}$C$_{org}$ data")

ax1.plot(mean_d13C_org_AN,t_vals[w_d13C_org_AN],color='forestgreen',linewidth=1.0)
ax1.plot([0,0],[0,0],color='forestgreen',linewidth=1.0,label="AN $\delta$$^{13}$C$_{org}$; $g_{2,3}(t)$")
ax1.plot(mean_d13C_org_AN + (std_d13C_org_AN),t_vals[w_d13C_org_AN],color='forestgreen',linestyle='--',linewidth=0.75)
ax1.plot(mean_d13C_org_AN - (std_d13C_org_AN),t_vals[w_d13C_org_AN],color='forestgreen',linestyle='--',linewidth=0.75)
ax1.plot(mean_d13C_org_AN + (2. * std_d13C_org_AN),t_vals[w_d13C_org_AN],color='forestgreen',linestyle=':',linewidth=0.5)
ax1.plot(mean_d13C_org_AN - (2. * std_d13C_org_AN),t_vals[w_d13C_org_AN],color='forestgreen',linestyle=':',linewidth=0.5)
ax1.set_ylim((-150),(150))
ax1.set_xlim(-31.,-22.)

ax1.set_title("Ancora organic")
ax1.set_xlabel("$\delta$$^{13}$C")

ax1.grid(alpha=0.0,)
ax1.legend()

ax1 = plt.subplot(5,5,12)


MC_iterations = 0
for n in range(0,MC_iterations):
	
	val2 = numpy.random.normal(loc = mean_d13C_org_WL, scale = std_d13C_org_WL)
	
	ax1.plot((val2),t_vals[w_d13C_org_WL],color='forestgreen',marker = 'x',markersize=2.0,linewidth=0.0,alpha=0.002)

out_mat_fold = 1

ax1.plot(SL_data_combined[w_d13C_org_WL_c,1],mean_timing[w_d13C_org_WL_c],color='forestgreen',marker='o',linewidth=0.0,markersize=4.0,alpha=0.5,label="WL $\delta$$^{13}$C$_{org}$ data")

ax1.plot(mean_d13C_org_WL,t_vals[w_d13C_org_WL],color='forestgreen',linewidth=1.0)
ax1.plot([0,0],[0,0],color='forestgreen',linewidth=1.0,label="WL $\delta$$^{13}$C$_{org}$; $g_{2,2}(t)$")
ax1.plot(mean_d13C_org_WL + (std_d13C_org_WL),t_vals[w_d13C_org_WL],color='forestgreen',linestyle='--',linewidth=0.75)
ax1.plot(mean_d13C_org_WL - (std_d13C_org_WL),t_vals[w_d13C_org_WL],color='forestgreen',linestyle='--',linewidth=0.75)
ax1.plot(mean_d13C_org_WL + (2. * std_d13C_org_WL),t_vals[w_d13C_org_WL],color='forestgreen',linestyle=':',linewidth=0.5)
ax1.plot(mean_d13C_org_WL - (2. * std_d13C_org_WL),t_vals[w_d13C_org_WL],color='forestgreen',linestyle=':',linewidth=0.5)
ax1.set_ylim((-150),(150))
ax1.set_xlim(-31.,-22.)

ax1.set_title("Wilson Lake organic")
ax1.set_xlabel("$\delta$$^{13}$C")

ax1.grid(alpha=0.0,)
ax1.legend()

ax1 = plt.subplot(5,5,11)


MC_iterations = 0
for n in range(0,MC_iterations):
	
	val2 = numpy.random.normal(loc = mean_d13C_org_MAP3A, scale = std_d13C_org_MAP3A)
	
	ax1.plot((val2),t_vals[w_d13C_org_MAP3A],color='forestgreen',marker = 'x',markersize=2.0,linewidth=0.0,alpha=0.002)

out_mat_fold = 1

ax1.plot(SL_data_combined[w_d13C_org_MAP3A_c,1],mean_timing[w_d13C_org_MAP3A_c],color='forestgreen',marker='o',linewidth=0.0,markersize=4.0,alpha=0.5,label="MAP3A/B $\delta$$^{13}$C$_{org}$ data")

ax1.plot(mean_d13C_org_MAP3A,t_vals[w_d13C_org_MAP3A],color='forestgreen',linewidth=1.0)
ax1.plot([0,0],[0,0],color='forestgreen',linewidth=1.0,label="MAP3A/B $\delta$$^{13}$C$_{org}$; $g_{2,1}(t)$")
ax1.plot(mean_d13C_org_MAP3A + (std_d13C_org_MAP3A),t_vals[w_d13C_org_MAP3A],color='forestgreen',linestyle='--',linewidth=0.75)
ax1.plot(mean_d13C_org_MAP3A - (std_d13C_org_MAP3A),t_vals[w_d13C_org_MAP3A],color='forestgreen',linestyle='--',linewidth=0.75)
ax1.plot(mean_d13C_org_MAP3A + (2. * std_d13C_org_MAP3A),t_vals[w_d13C_org_MAP3A],color='forestgreen',linestyle=':',linewidth=0.5)
ax1.plot(mean_d13C_org_MAP3A - (2. * std_d13C_org_MAP3A),t_vals[w_d13C_org_MAP3A],color='forestgreen',linestyle=':',linewidth=0.5)
ax1.set_ylim((-150),(150))
ax1.set_xlim(-31.,-22.)

ax1.set_title("MAP3A/B organic")
ax1.set_xlabel("$\delta$$^{13}$C")
ax1.set_ylabel("Correlation units (rel. to CIE)")
ax1.grid(alpha=0.0,)
ax1.legend(loc=1)

########################
########################
########################

ax1 = plt.subplot(5,5,20)


MC_iterations = 0
for n in range(0,MC_iterations):
	
	val2 = numpy.random.normal(loc = mean_d13C_PF_BR, scale = std_d13C_PF_BR)
	
	ax1.plot((val2),t_vals[w_d13C_PF_BR],color='dodgerblue',marker = 'x',markersize=2.0,linewidth=0.0,alpha=0.002)

out_mat_fold = 1

ax1.plot(SL_data_combined[w_d13C_PF_BR_c,1],mean_timing[w_d13C_PF_BR_c],color='dodgerblue',marker='o',linewidth=0.0,markersize=4.0,alpha=0.5,label="BR $\delta$$^{13}$C$_{PF}$ data")

ax1.plot(mean_d13C_PF_BR,t_vals[w_d13C_PF_BR],color='dodgerblue',linewidth=1.0)
ax1.plot([0,0],[0,0],color='dodgerblue',linewidth=1.0,label="BR $\delta$$^{13}$C$_{PF}$; $g_{3,5}(t)$")
ax1.plot(mean_d13C_PF_BR + (std_d13C_PF_BR),t_vals[w_d13C_PF_BR],color='dodgerblue',linestyle='--',linewidth=0.75)
ax1.plot(mean_d13C_PF_BR - (std_d13C_PF_BR),t_vals[w_d13C_PF_BR],color='dodgerblue',linestyle='--',linewidth=0.75)
ax1.plot(mean_d13C_PF_BR + (2. * std_d13C_PF_BR),t_vals[w_d13C_PF_BR],color='dodgerblue',linestyle=':',linewidth=0.5)
ax1.plot(mean_d13C_PF_BR - (2. * std_d13C_PF_BR),t_vals[w_d13C_PF_BR],color='dodgerblue',linestyle=':',linewidth=0.5)

MC_iterations = 0
for n in range(0,MC_iterations):
	
	val2 = numpy.random.normal(loc = mean_d13C_BF_BR, scale = std_d13C_BF_BR)
	
	ax1.plot((val2),t_vals[w_d13C_BF_BR],color='indigo',marker = 'x',markersize=2.0,linewidth=0.0,alpha=0.002)

out_mat_fold = 1

ax1.plot(SL_data_combined[w_d13C_BF_BR_c,1],mean_timing[w_d13C_BF_BR_c],color='indigo',marker='o',linewidth=0.0,markersize=4.0,alpha=0.5,label="BR $\delta$$^{13}$C$_{BF}$ data")

ax1.plot(mean_d13C_BF_BR,t_vals[w_d13C_BF_BR],color='indigo',linewidth=1.0)
ax1.plot([0,0],[0,0],color='indigo',linewidth=1.0,label="BR $\delta$$^{13}$C$_{BF}$; $g_{4,5}(t)$")
ax1.plot(mean_d13C_BF_BR + (std_d13C_BF_BR),t_vals[w_d13C_BF_BR],color='indigo',linestyle='--',linewidth=0.75)
ax1.plot(mean_d13C_BF_BR - (std_d13C_BF_BR),t_vals[w_d13C_BF_BR],color='indigo',linestyle='--',linewidth=0.75)
ax1.plot(mean_d13C_BF_BR + (2. * std_d13C_BF_BR),t_vals[w_d13C_BF_BR],color='indigo',linestyle=':',linewidth=0.5)
ax1.plot(mean_d13C_BF_BR - (2. * std_d13C_BF_BR),t_vals[w_d13C_BF_BR],color='indigo',linestyle=':',linewidth=0.5)

ax1.set_ylim((-150),(150))
ax1.set_xlim(-5.5,4.)

ax1.set_title("Bass River forams")
ax1.set_xlabel("$\delta$$^{13}$C")

ax1.grid(alpha=0.0,)
ax1.legend()

ax1 = plt.subplot(5,5,19)


MC_iterations = 0
for n in range(0,MC_iterations):
	
	val2 = numpy.random.normal(loc = mean_d13C_PF_MI, scale = std_d13C_PF_MI)
	
	ax1.plot((val2),t_vals[w_d13C_PF_MI],color='dodgerblue',marker = 'x',markersize=2.0,linewidth=0.0,alpha=0.002)

out_mat_fold = 1

ax1.plot(SL_data_combined[w_d13C_PF_MI_c,1],mean_timing[w_d13C_PF_MI_c],color='dodgerblue',marker='o',linewidth=0.0,markersize=4.0,alpha=0.5,label="MI $\delta$$^{13}$C$_{PF}$ data")

ax1.plot(mean_d13C_PF_MI,t_vals[w_d13C_PF_MI],color='dodgerblue',linewidth=1.0)
ax1.plot([0,0],[0,0],color='dodgerblue',linewidth=1.0,label="MI $\delta$$^{13}$C$_{PF}$; $g_{3,4}(t)$")
ax1.plot(mean_d13C_PF_MI + (std_d13C_PF_MI),t_vals[w_d13C_PF_MI],color='dodgerblue',linestyle='--',linewidth=0.75)
ax1.plot(mean_d13C_PF_MI - (std_d13C_PF_MI),t_vals[w_d13C_PF_MI],color='dodgerblue',linestyle='--',linewidth=0.75)
ax1.plot(mean_d13C_PF_MI + (2. * std_d13C_PF_MI),t_vals[w_d13C_PF_MI],color='dodgerblue',linestyle=':',linewidth=0.5)
ax1.plot(mean_d13C_PF_MI - (2. * std_d13C_PF_MI),t_vals[w_d13C_PF_MI],color='dodgerblue',linestyle=':',linewidth=0.5)

MC_iterations = 0
for n in range(0,MC_iterations):
	
	val2 = numpy.random.normal(loc = mean_d13C_BF_MI, scale = std_d13C_BF_MI)
	
	ax1.plot((val2),t_vals[w_d13C_BF_MI],color='indigo',marker = 'x',markersize=2.0,linewidth=0.0,alpha=0.002)

out_mat_fold = 1

ax1.plot(SL_data_combined[w_d13C_BF_MI_c,1],mean_timing[w_d13C_BF_MI_c],color='indigo',marker='o',linewidth=0.0,markersize=4.0,alpha=0.5,label="MI $\delta$$^{13}$C$_{BF}$ data")

ax1.plot(mean_d13C_BF_MI,t_vals[w_d13C_BF_MI],color='indigo',linewidth=1.0)
ax1.plot([0,0],[0,0],color='indigo',linewidth=1.0,label="MI $\delta$$^{13}$C$_{BF}$; $g_{4,4}(t)$")
ax1.plot(mean_d13C_BF_MI + (std_d13C_BF_MI),t_vals[w_d13C_BF_MI],color='indigo',linestyle='--',linewidth=0.75)
ax1.plot(mean_d13C_BF_MI - (std_d13C_BF_MI),t_vals[w_d13C_BF_MI],color='indigo',linestyle='--',linewidth=0.75)
ax1.plot(mean_d13C_BF_MI + (2. * std_d13C_BF_MI),t_vals[w_d13C_BF_MI],color='indigo',linestyle=':',linewidth=0.5)
ax1.plot(mean_d13C_BF_MI - (2. * std_d13C_BF_MI),t_vals[w_d13C_BF_MI],color='indigo',linestyle=':',linewidth=0.5)

ax1.set_ylim((-150),(150))
ax1.set_xlim(-5.5,4.)

ax1.set_title("Millville forams")
ax1.set_xlabel("$\delta$$^{13}$C")

ax1.grid(alpha=0.0,)
ax1.legend()

ax1 = plt.subplot(5,5,18)


MC_iterations = 0
for n in range(0,MC_iterations):
	
	val2 = numpy.random.normal(loc = mean_d13C_PF_AN, scale = std_d13C_PF_AN)
	
	ax1.plot((val2),t_vals[w_d13C_PF_AN],color='dodgerblue',marker = 'x',markersize=2.0,linewidth=0.0,alpha=0.002)

out_mat_fold = 1

ax1.plot(SL_data_combined[w_d13C_PF_AN_c,1],mean_timing[w_d13C_PF_AN_c],color='dodgerblue',marker='o',linewidth=0.0,markersize=4.0,alpha=0.5,label="AN $\delta$$^{13}$C$_{PF}$ data")

ax1.plot(mean_d13C_PF_AN,t_vals[w_d13C_PF_AN],color='dodgerblue',linewidth=1.0)
ax1.plot([0,0],[0,0],color='dodgerblue',linewidth=1.0,label="AN $\delta$$^{13}$C$_{PF}$; $g_{3,3}(t)$")
ax1.plot(mean_d13C_PF_AN + (std_d13C_PF_AN),t_vals[w_d13C_PF_AN],color='dodgerblue',linestyle='--',linewidth=0.75)
ax1.plot(mean_d13C_PF_AN - (std_d13C_PF_AN),t_vals[w_d13C_PF_AN],color='dodgerblue',linestyle='--',linewidth=0.75)
ax1.plot(mean_d13C_PF_AN + (2. * std_d13C_PF_AN),t_vals[w_d13C_PF_AN],color='dodgerblue',linestyle=':',linewidth=0.5)
ax1.plot(mean_d13C_PF_AN - (2. * std_d13C_PF_AN),t_vals[w_d13C_PF_AN],color='dodgerblue',linestyle=':',linewidth=0.5)

MC_iterations = 0
for n in range(0,MC_iterations):
	
	val2 = numpy.random.normal(loc = mean_d13C_BF_AN, scale = std_d13C_BF_AN)
	
	ax1.plot((val2),t_vals[w_d13C_BF_AN],color='indigo',marker = 'x',markersize=2.0,linewidth=0.0,alpha=0.002)

out_mat_fold = 1

ax1.plot(SL_data_combined[w_d13C_BF_AN_c,1],mean_timing[w_d13C_BF_AN_c],color='indigo',marker='o',linewidth=0.0,markersize=4.0,alpha=0.5,label="AN $\delta$$^{13}$C$_{BF}$ data")

ax1.plot(mean_d13C_BF_AN,t_vals[w_d13C_BF_AN],color='indigo',linewidth=1.0)
ax1.plot([0,0],[0,0],color='indigo',linewidth=1.0,label="AN $\delta$$^{13}$C$_{BF}$; $g_{4,3}(t)$")
ax1.plot(mean_d13C_BF_AN + (std_d13C_BF_AN),t_vals[w_d13C_BF_AN],color='indigo',linestyle='--',linewidth=0.75)
ax1.plot(mean_d13C_BF_AN - (std_d13C_BF_AN),t_vals[w_d13C_BF_AN],color='indigo',linestyle='--',linewidth=0.75)
ax1.plot(mean_d13C_BF_AN + (2. * std_d13C_BF_AN),t_vals[w_d13C_BF_AN],color='indigo',linestyle=':',linewidth=0.5)
ax1.plot(mean_d13C_BF_AN - (2. * std_d13C_BF_AN),t_vals[w_d13C_BF_AN],color='indigo',linestyle=':',linewidth=0.5)

ax1.set_title("Ancora forams")
ax1.set_xlabel("$\delta$$^{13}$C")

ax1.set_ylim((-150),(150))
ax1.set_xlim(-5.5,4.)

ax1.grid(alpha=0.0,)
ax1.legend()

ax1 = plt.subplot(5,5,17)


MC_iterations = 0
for n in range(0,MC_iterations):
	
	val2 = numpy.random.normal(loc = mean_d13C_PF_WL, scale = std_d13C_PF_WL)
	
	ax1.plot((val2),t_vals[w_d13C_PF_WL],color='dodgerblue',marker = 'x',markersize=2.0,linewidth=0.0,alpha=0.002)

out_mat_fold = 1

ax1.plot(SL_data_combined[w_d13C_PF_WL_c,1],mean_timing[w_d13C_PF_WL_c],color='dodgerblue',marker='o',linewidth=0.0,markersize=4.0,alpha=0.5,label="WL $\delta$$^{13}$C$_{PF}$ data")

ax1.plot(mean_d13C_PF_WL,t_vals[w_d13C_PF_WL],color='dodgerblue',linewidth=1.0)
ax1.plot([0,0],[0,0],color='dodgerblue',linewidth=1.0,label="WL $\delta$$^{13}$C$_{PF}$; $g_{3,2}(t)$")
ax1.plot(mean_d13C_PF_WL + (std_d13C_PF_WL),t_vals[w_d13C_PF_WL],color='dodgerblue',linestyle='--',linewidth=0.75)
ax1.plot(mean_d13C_PF_WL - (std_d13C_PF_WL),t_vals[w_d13C_PF_WL],color='dodgerblue',linestyle='--',linewidth=0.75)
ax1.plot(mean_d13C_PF_WL + (2. * std_d13C_PF_WL),t_vals[w_d13C_PF_WL],color='dodgerblue',linestyle=':',linewidth=0.5)
ax1.plot(mean_d13C_PF_WL - (2. * std_d13C_PF_WL),t_vals[w_d13C_PF_WL],color='dodgerblue',linestyle=':',linewidth=0.5)

MC_iterations = 0
for n in range(0,MC_iterations):
	
	val2 = numpy.random.normal(loc = mean_d13C_BF_WL, scale = std_d13C_BF_WL)
	
	ax1.plot((val2),t_vals[w_d13C_BF_WL],color='indigo',marker = 'x',markersize=2.0,linewidth=0.0,alpha=0.002)

out_mat_fold = 1

ax1.plot(SL_data_combined[w_d13C_BF_WL_c,1],mean_timing[w_d13C_BF_WL_c],color='indigo',marker='o',linewidth=0.0,markersize=4.0,alpha=0.5,label="WL $\delta$$^{13}$C$_{BF}$ data")

ax1.plot(mean_d13C_BF_WL,t_vals[w_d13C_BF_WL],color='indigo',linewidth=1.0)
ax1.plot([0,0],[0,0],color='indigo',linewidth=1.0,label="WL $\delta$$^{13}$C$_{BF}$; $g_{4,2}(t)$")
ax1.plot(mean_d13C_BF_WL + (std_d13C_BF_WL),t_vals[w_d13C_BF_WL],color='indigo',linestyle='--',linewidth=0.75)
ax1.plot(mean_d13C_BF_WL - (std_d13C_BF_WL),t_vals[w_d13C_BF_WL],color='indigo',linestyle='--',linewidth=0.75)
ax1.plot(mean_d13C_BF_WL + (2. * std_d13C_BF_WL),t_vals[w_d13C_BF_WL],color='indigo',linestyle=':',linewidth=0.5)
ax1.plot(mean_d13C_BF_WL - (2. * std_d13C_BF_WL),t_vals[w_d13C_BF_WL],color='indigo',linestyle=':',linewidth=0.5)

ax1.set_ylim((-150),(150))
ax1.set_xlim(-5.5,4.)

ax1.set_title("Wilson Lake forams")
ax1.set_xlabel("$\delta$$^{13}$C")

ax1.grid(alpha=0.0,)
ax1.legend()

ax1 = plt.subplot(5,5,16)


MC_iterations = 0
for n in range(0,MC_iterations):
	
	val2 = numpy.random.normal(loc = mean_d13C_PF_MAP3A, scale = std_d13C_PF_MAP3A)
	
	ax1.plot((val2),t_vals[w_d13C_PF_MAP3A],color='dodgerblue',marker = 'x',markersize=2.0,linewidth=0.0,alpha=0.002)

out_mat_fold = 1

ax1.plot(SL_data_combined[w_d13C_PF_MAP3A_c,1],mean_timing[w_d13C_PF_MAP3A_c],color='dodgerblue',marker='o',linewidth=0.0,markersize=4.0,alpha=0.5,label="MAP3A/B $\delta$$^{13}$C$_{PF}$ data")

ax1.plot(mean_d13C_PF_MAP3A,t_vals[w_d13C_PF_MAP3A],color='dodgerblue',linewidth=1.0)
ax1.plot([0,0],[0,0],color='dodgerblue',linewidth=1.0,label="MAP3A/B $\delta$$^{13}$C$_{PF}$; $g_{3,1}(t)$")
ax1.plot(mean_d13C_PF_MAP3A + (std_d13C_PF_MAP3A),t_vals[w_d13C_PF_MAP3A],color='dodgerblue',linestyle='--',linewidth=0.75)
ax1.plot(mean_d13C_PF_MAP3A - (std_d13C_PF_MAP3A),t_vals[w_d13C_PF_MAP3A],color='dodgerblue',linestyle='--',linewidth=0.75)
ax1.plot(mean_d13C_PF_MAP3A + (2. * std_d13C_PF_MAP3A),t_vals[w_d13C_PF_MAP3A],color='dodgerblue',linestyle=':',linewidth=0.5)
ax1.plot(mean_d13C_PF_MAP3A - (2. * std_d13C_PF_MAP3A),t_vals[w_d13C_PF_MAP3A],color='dodgerblue',linestyle=':',linewidth=0.5)

MC_iterations = 0
for n in range(0,MC_iterations):
	
	val2 = numpy.random.normal(loc = mean_d13C_BF_MAP3A, scale = std_d13C_BF_MAP3A)
	
	ax1.plot((val2),t_vals[w_d13C_BF_MAP3A],color='indigo',marker = 'x',markersize=2.0,linewidth=0.0,alpha=0.002)

out_mat_fold = 1

ax1.plot(SL_data_combined[w_d13C_BF_MAP3A_c,1],mean_timing[w_d13C_BF_MAP3A_c],color='indigo',marker='o',linewidth=0.0,markersize=4.0,alpha=0.5,label="MAP3A/B $\delta$$^{13}$C$_{BF}$ data")

ax1.plot(mean_d13C_BF_MAP3A,t_vals[w_d13C_BF_MAP3A],color='indigo',linewidth=1.0)
ax1.plot([0,0],[0,0],color='indigo',linewidth=1.0,label="MAP3A/B $\delta$$^{13}$C$_{BF}$; $g_{4,1}(t)$")
ax1.plot(mean_d13C_BF_MAP3A + (std_d13C_BF_MAP3A),t_vals[w_d13C_BF_MAP3A],color='indigo',linestyle='--',linewidth=0.75)
ax1.plot(mean_d13C_BF_MAP3A - (std_d13C_BF_MAP3A),t_vals[w_d13C_BF_MAP3A],color='indigo',linestyle='--',linewidth=0.75)
ax1.plot(mean_d13C_BF_MAP3A + (2. * std_d13C_BF_MAP3A),t_vals[w_d13C_BF_MAP3A],color='indigo',linestyle=':',linewidth=0.5)
ax1.plot(mean_d13C_BF_MAP3A - (2. * std_d13C_BF_MAP3A),t_vals[w_d13C_BF_MAP3A],color='indigo',linestyle=':',linewidth=0.5)

ax1.set_ylim((-150),(150))
ax1.set_xlim(-5.5,4.)

ax1.set_title("MAP3A/B forams")
ax1.set_xlabel("$\delta$$^{13}$C")
ax1.set_ylabel("Correlation units (rel. to CIE)")
ax1.set_ylabel("Correlation units (rel. to CIE)")

ax1.grid(alpha=0.0,)
ax1.legend(loc=3)


########################
########################
########################

ax1 = plt.subplot(5,5,25)


MC_iterations = 0
for n in range(0,MC_iterations):
	
	val2 = numpy.random.normal(loc = mean_TEX86_BR, scale = std_TEX86_BR)
	
	ax1.plot((val2),t_vals[w_TEX86_BR],color='mediumvioletred',marker = 'x',markersize=2.0,linewidth=0.0,alpha=0.002)

out_mat_fold = 1

ax1.plot(SL_data_combined[w_TEX86_BR_c,1],mean_timing[w_TEX86_BR_c],color='mediumvioletred',marker='o',linewidth=0.0,markersize=4.0,alpha=0.5,label="BR TEX86 data")

ax1.plot(mean_TEX86_BR,t_vals[w_TEX86_BR],color='mediumvioletred',linewidth=1.0)
ax1.plot([0,0],[0,0],color='mediumvioletred',linewidth=1.0,label="BR TEX86; $h_{5}(t)$")
ax1.plot(mean_TEX86_BR + (std_TEX86_BR),t_vals[w_TEX86_BR],color='mediumvioletred',linestyle='--',linewidth=0.75)
ax1.plot(mean_TEX86_BR - (std_TEX86_BR),t_vals[w_TEX86_BR],color='mediumvioletred',linestyle='--',linewidth=0.75)
ax1.plot(mean_TEX86_BR + (2. * std_TEX86_BR),t_vals[w_TEX86_BR],color='mediumvioletred',linestyle=':',linewidth=0.5)
ax1.plot(mean_TEX86_BR - (2. * std_TEX86_BR),t_vals[w_TEX86_BR],color='mediumvioletred',linestyle=':',linewidth=0.5)
ax1.set_ylim((-150),(150))
ax1.set_xlim(26.75,37.51)

ax1.grid(alpha=0.0,)
ax1.legend()

ax1.set_title("Bass River temperature")
ax1.set_xlabel("Temperature ($\degree$C)")

ax1 = plt.subplot(5,5,24)


MC_iterations = 0
for n in range(0,MC_iterations):
	
	val2 = numpy.random.normal(loc = mean_TEX86_MI, scale = std_TEX86_MI)
	
	ax1.plot((val2),t_vals[w_TEX86_MI],color='mediumvioletred',marker = 'x',markersize=2.0,linewidth=0.0,alpha=0.002)

out_mat_fold = 1

ax1.plot(SL_data_combined[w_TEX86_MI_c,1],mean_timing[w_TEX86_MI_c],color='mediumvioletred',marker='o',linewidth=0.0,markersize=4.0,alpha=0.5,label="MI TEX86 data")

ax1.plot(mean_TEX86_MI,t_vals[w_TEX86_MI],color='mediumvioletred',linewidth=1.0)
ax1.plot([0,0],[0,0],color='mediumvioletred',linewidth=1.0,label="MI TEX86; $h_{4}(t)$")
ax1.plot(mean_TEX86_MI + (std_TEX86_MI),t_vals[w_TEX86_MI],color='mediumvioletred',linestyle='--',linewidth=0.75)
ax1.plot(mean_TEX86_MI - (std_TEX86_MI),t_vals[w_TEX86_MI],color='mediumvioletred',linestyle='--',linewidth=0.75)
ax1.plot(mean_TEX86_MI + (2. * std_TEX86_MI),t_vals[w_TEX86_MI],color='mediumvioletred',linestyle=':',linewidth=0.5)
ax1.plot(mean_TEX86_MI - (2. * std_TEX86_MI),t_vals[w_TEX86_MI],color='mediumvioletred',linestyle=':',linewidth=0.5)
ax1.set_ylim((-150),(150))
ax1.set_xlim(26.75,37.51)

ax1.set_title("Millville temperature")
ax1.set_xlabel("Temperature ($\degree$C)")

ax1.grid(alpha=0.0,)
ax1.legend()

ax1 = plt.subplot(5,5,23)


MC_iterations = 0
for n in range(0,MC_iterations):
	
	val2 = numpy.random.normal(loc = mean_TEX86_AN, scale = std_TEX86_AN)
	
	ax1.plot((val2),t_vals[w_TEX86_AN],color='mediumvioletred',marker = 'x',markersize=2.0,linewidth=0.0,alpha=0.002)

out_mat_fold = 1

ax1.plot(SL_data_combined[w_TEX86_AN_c,1],mean_timing[w_TEX86_AN_c],color='mediumvioletred',marker='o',linewidth=0.0,markersize=4.0,alpha=0.5,label="AN TEX86 data")

ax1.plot(mean_TEX86_AN,t_vals[w_TEX86_AN],color='mediumvioletred',linewidth=1.0)
ax1.plot([0,0],[0,0],color='mediumvioletred',linewidth=1.0,label="AN TEX86; $h_{3}(t)$")
ax1.plot(mean_TEX86_AN + (std_TEX86_AN),t_vals[w_TEX86_AN],color='mediumvioletred',linestyle='--',linewidth=0.75)
ax1.plot(mean_TEX86_AN - (std_TEX86_AN),t_vals[w_TEX86_AN],color='mediumvioletred',linestyle='--',linewidth=0.75)
ax1.plot(mean_TEX86_AN + (2. * std_TEX86_AN),t_vals[w_TEX86_AN],color='mediumvioletred',linestyle=':',linewidth=0.5)
ax1.plot(mean_TEX86_AN - (2. * std_TEX86_AN),t_vals[w_TEX86_AN],color='mediumvioletred',linestyle=':',linewidth=0.5)
ax1.set_ylim((-150),(150))
ax1.set_xlim(26.75,37.51)

ax1.set_title("Ancora temperature")
ax1.set_xlabel("Temperature ($\degree$C)")

ax1.grid(alpha=0.0,)
ax1.legend()

ax1 = plt.subplot(5,5,22)


MC_iterations = 0
for n in range(0,MC_iterations):
	
	val2 = numpy.random.normal(loc = mean_TEX86_WL, scale = std_TEX86_WL)
	
	ax1.plot((val2),t_vals[w_TEX86_WL],color='mediumvioletred',marker = 'x',markersize=2.0,linewidth=0.0,alpha=0.002)

out_mat_fold = 1

ax1.plot(SL_data_combined[w_TEX86_WL_c,1],mean_timing[w_TEX86_WL_c],color='mediumvioletred',marker='o',linewidth=0.0,markersize=4.0,alpha=0.5,label="WL TEX86 data")

ax1.plot(mean_TEX86_WL,t_vals[w_TEX86_WL],color='mediumvioletred',linewidth=1.0)
ax1.plot([0,0],[0,0],color='mediumvioletred',linewidth=1.0,label="WL TEX86; $h_{2}(t)$")
ax1.plot(mean_TEX86_WL + (std_TEX86_WL),t_vals[w_TEX86_WL],color='mediumvioletred',linestyle='--',linewidth=0.75)
ax1.plot(mean_TEX86_WL - (std_TEX86_WL),t_vals[w_TEX86_WL],color='mediumvioletred',linestyle='--',linewidth=0.75)
ax1.plot(mean_TEX86_WL + (2. * std_TEX86_WL),t_vals[w_TEX86_WL],color='mediumvioletred',linestyle=':',linewidth=0.5)
ax1.plot(mean_TEX86_WL - (2. * std_TEX86_WL),t_vals[w_TEX86_WL],color='mediumvioletred',linestyle=':',linewidth=0.5)
ax1.set_ylim((-150),(150))
ax1.set_xlim(26.75,37.51)

ax1.set_title("Wilson Lake temperature")
ax1.set_xlabel("Temperature ($\degree$C)")

ax1.grid(alpha=0.0,)
ax1.legend()

ax1 = plt.subplot(5,5,21)


MC_iterations = 0
for n in range(0,MC_iterations):
	
	val2 = numpy.random.normal(loc = mean_TEX86_MAP3A, scale = std_TEX86_MAP3A)
	
	ax1.plot((val2),t_vals[w_TEX86_MAP3A],color='mediumvioletred',marker = 'x',markersize=2.0,linewidth=0.0,alpha=0.002)

out_mat_fold = 1

ax1.plot(SL_data_combined[w_TEX86_MAP3A_c,1],mean_timing[w_TEX86_MAP3A_c],color='mediumvioletred',marker='o',linewidth=0.0,markersize=4.0,alpha=0.5,label="MAP3A/B TEX86 data")

ax1.plot(mean_TEX86_MAP3A,t_vals[w_TEX86_MAP3A],color='mediumvioletred',linewidth=1.0)
ax1.plot([0,0],[0,0],color='mediumvioletred',linewidth=1.0,label="MAP3A/B TEX86; $h_{1}(t)$")
ax1.plot(mean_TEX86_MAP3A + (std_TEX86_MAP3A),t_vals[w_TEX86_MAP3A],color='mediumvioletred',linestyle='--',linewidth=0.75)
ax1.plot(mean_TEX86_MAP3A - (std_TEX86_MAP3A),t_vals[w_TEX86_MAP3A],color='mediumvioletred',linestyle='--',linewidth=0.75)
ax1.plot(mean_TEX86_MAP3A + (2. * std_TEX86_MAP3A),t_vals[w_TEX86_MAP3A],color='mediumvioletred',linestyle=':',linewidth=0.5)
ax1.plot(mean_TEX86_MAP3A - (2. * std_TEX86_MAP3A),t_vals[w_TEX86_MAP3A],color='mediumvioletred',linestyle=':',linewidth=0.5)
ax1.set_ylim((-150),(150))
ax1.set_xlim(26.75,37.51)

ax1.set_title("MAP3A/B temperature")
ax1.set_xlabel("Temperature ($\degree$C)")
ax1.set_ylabel("Correlation units (rel. to CIE)")

ax1.grid(alpha=0.0,)
ax1.legend()

plt.tight_layout()

pltname = wkspc +  'NJ_d13C_20210604.png'
plt.savefig(pltname, dpi = 300)
pltname = wkspc +  'NJ_d13C_20230314.pdf'
plt.savefig(pltname, dpi = 300)

plt.close()

fig= plt.figure(13,figsize=(20.,16.5))

ax1 = plt.subplot(3,5,1)

out_mat_fold = 1

ax1.plot(mean_global,t_vals[w_d13C_glob],'k',linewidth=1.0)
ax1.plot([0,0],[0,0],'k',linewidth=1.0,label="NJ shelf $\delta$$^{13}$C")
ax1.plot(mean_global + (std_global),t_vals[w_d13C_glob],'k--',linewidth=0.75)
ax1.plot(mean_global - (std_global),t_vals[w_d13C_glob],'k--',linewidth=0.75)
ax1.plot(mean_global + (2. * std_global),t_vals[w_d13C_glob],'k:',linewidth=0.5)
ax1.plot(mean_global - (2. * std_global),t_vals[w_d13C_glob],'k:',linewidth=0.5)

ax1.set_yticks(numpy.arange(-30,30,2.), minor=True)

ax1.set_ylim(-30,30)

ax1.set_xlim(-5.,3.5)

ax1.grid(alpha=1.0,which='both')
ax1.legend()

ax1.set_title("NJ shelf  $\delta$$^{13}$C")
ax1.set_xlabel("$\delta$$^{13}$C")
ax1.set_ylabel("Correlation units (rel. to CIE)")

ax1 = plt.subplot(3,5,2)

out_mat_fold = 1

ax1.plot(SL_data_combined[w_d13C_bulk_c,1],mean_timing[w_d13C_bulk_c],color='tomato',marker='o',linewidth=0.0,markersize=2.0,alpha=0.15,label="NJ $\delta$$^{13}$C$_{bulk}$ data")

ax1.plot(mean_NJ_d13C_bulk,t_vals[w_d13C_glob],color='tomato',linewidth=1.0)
ax1.plot([0,0],[0,0],color='tomato',linewidth=1.0,label="$\delta$$^{13}$C$_{bulk}$")
ax1.plot(mean_NJ_d13C_bulk + (std_NJ_d13C_bulk),t_vals[w_d13C_glob],color='tomato',linestyle='--',linewidth=0.75)
ax1.plot(mean_NJ_d13C_bulk - (std_NJ_d13C_bulk),t_vals[w_d13C_glob],color='tomato',linestyle='--',linewidth=0.75)
ax1.plot(mean_NJ_d13C_bulk + (2. * std_NJ_d13C_bulk),t_vals[w_d13C_glob],color='tomato',linestyle=':',linewidth=0.5)
ax1.plot(mean_NJ_d13C_bulk - (2. * std_NJ_d13C_bulk),t_vals[w_d13C_glob],color='tomato',linestyle=':',linewidth=0.5)

ax1.set_yticks(numpy.arange(-30,30,2.), minor=True)

ax1.set_ylim(-30,30)

ax1.set_xlim(-5.,3.5)

ax1.grid(alpha=1.0,which='both')
ax1.legend()

ax1.set_title("NJ shelf $\delta$$^{13}$C$_{bulk}$")
ax1.set_xlabel("$\delta$$^{13}$C")

ax1 = plt.subplot(3,5,3)

out_mat_fold = 1

ax1.plot(SL_data_combined[w_d13C_org_c,1],mean_timing[w_d13C_org_c],color='forestgreen',marker='o',linewidth=0.0,markersize=2.0,alpha=0.15,label="NJ $\delta$$^{13}$C$_{org}$ data")

ax1.plot(mean_NJ_d13C_org,t_vals[w_d13C_glob],color='forestgreen',linewidth=1.0)
ax1.plot([0,0],[0,0],color='forestgreen',linewidth=1.0,label="$\delta$$^{13}$C$_{org}$")
ax1.plot(mean_NJ_d13C_org + (std_NJ_d13C_org),t_vals[w_d13C_glob],color='forestgreen',linestyle='--',linewidth=0.75)
ax1.plot(mean_NJ_d13C_org - (std_NJ_d13C_org),t_vals[w_d13C_glob],color='forestgreen',linestyle='--',linewidth=0.75)
ax1.plot(mean_NJ_d13C_org + (2. * std_NJ_d13C_org),t_vals[w_d13C_glob],color='forestgreen',linestyle=':',linewidth=0.5)
ax1.plot(mean_NJ_d13C_org - (2. * std_NJ_d13C_org),t_vals[w_d13C_glob],color='forestgreen',linestyle=':',linewidth=0.5)

ax1.set_yticks(numpy.arange(-30,30,2.), minor=True)

ax1.set_ylim(-30,30)

ax1.set_xlim(-31.,-22.)

ax1.set_title("NJ shelf $\delta$$^{13}$C$_{org}$")
ax1.set_xlabel("$\delta$$^{13}$C")

ax1.grid(alpha=1.0,which='both')
ax1.legend()

ax1 = plt.subplot(3,5,4)

out_mat_fold = 1

ax1.plot(SL_data_combined[w_d13C_PF_c,1],mean_timing[w_d13C_PF_c],color='dodgerblue',marker='o',linewidth=0.0,markersize=2.0,alpha=0.15,label="NJ $\delta$$^{13}$C$_{PF}$ data")

ax1.plot(mean_NJ_d13C_PF,t_vals[w_d13C_glob],color='dodgerblue',linewidth=1.0)
ax1.plot([0,0],[0,0],color='dodgerblue',linewidth=1.0,label="$\delta$$^{13}$C$_{PF}$")
ax1.plot(mean_NJ_d13C_PF + (std_NJ_d13C_PF),t_vals[w_d13C_glob],color='dodgerblue',linestyle='--',linewidth=0.75)
ax1.plot(mean_NJ_d13C_PF - (std_NJ_d13C_PF),t_vals[w_d13C_glob],color='dodgerblue',linestyle='--',linewidth=0.75)
ax1.plot(mean_NJ_d13C_PF + (2. * std_NJ_d13C_PF),t_vals[w_d13C_glob],color='dodgerblue',linestyle=':',linewidth=0.5)
ax1.plot(mean_NJ_d13C_PF - (2. * std_NJ_d13C_PF),t_vals[w_d13C_glob],color='dodgerblue',linestyle=':',linewidth=0.5)

out_mat_fold = 1

ax1.plot(SL_data_combined[w_d13C_BF_c,1],mean_timing[w_d13C_BF_c],color='indigo',marker='o',linewidth=0.0,markersize=2.0,alpha=0.15,label="NJ $\delta$$^{13}$C$_{BF}$ data")

ax1.plot(mean_NJ_d13C_BF,t_vals[w_d13C_glob],color='indigo',linewidth=1.0)
ax1.plot([0,0],[0,0],color='indigo',linewidth=1.0,label="$\delta$$^{13}$C$_{BF}$")
ax1.plot(mean_NJ_d13C_BF + (std_NJ_d13C_BF),t_vals[w_d13C_glob],color='indigo',linestyle='--',linewidth=0.75)
ax1.plot(mean_NJ_d13C_BF - (std_NJ_d13C_BF),t_vals[w_d13C_glob],color='indigo',linestyle='--',linewidth=0.75)
ax1.plot(mean_NJ_d13C_BF + (2. * std_NJ_d13C_BF),t_vals[w_d13C_glob],color='indigo',linestyle=':',linewidth=0.5)
ax1.plot(mean_NJ_d13C_BF - (2. * std_NJ_d13C_BF),t_vals[w_d13C_glob],color='indigo',linestyle=':',linewidth=0.5)

ax1.set_yticks(numpy.arange(-30,30,2.), minor=True)

ax1.set_ylim(-30,30)

ax1.set_xlim(-5.5,4.)

ax1.set_title("NJ shelf forams")
ax1.set_xlabel("$\delta$$^{13}$C")

ax1.grid(alpha=1.0,which='both')
ax1.legend()

ax1 = plt.subplot(3,5,5)

out_mat_fold = 1

ax1.plot(SL_data_combined[w_TEX86_c,1],mean_timing[w_TEX86_c],color='mediumvioletred',marker='o',linewidth=0.0,markersize=2.0,alpha=0.15,label="TEX86")

ax1.plot(mean_NJ_TEX86,t_vals[w_d13C_glob],color='mediumvioletred',linewidth=1.0)
ax1.plot([0,0],[0,0],color='mediumvioletred',linewidth=1.0,label="TEX86")
ax1.plot(mean_NJ_TEX86 + (std_NJ_TEX86),t_vals[w_d13C_glob],color='mediumvioletred',linestyle='--',linewidth=0.75)
ax1.plot(mean_NJ_TEX86 - (std_NJ_TEX86),t_vals[w_d13C_glob],color='mediumvioletred',linestyle='--',linewidth=0.75)
ax1.plot(mean_NJ_TEX86 + (2. * std_NJ_TEX86),t_vals[w_d13C_glob],color='mediumvioletred',linestyle=':',linewidth=0.5)
ax1.plot(mean_NJ_TEX86 - (2. * std_NJ_TEX86),t_vals[w_d13C_glob],color='mediumvioletred',linestyle=':',linewidth=0.5)

ax1.set_yticks(numpy.arange(-30,30,2.), minor=True)

ax1.set_ylim(-30,30)

ax1.set_xlim(26.75,37.51)

ax1.set_title("NJ shelf temperature")
ax1.set_xlabel("Temperature ($\degree$C)")

ax1.grid(alpha=1.0,which='both')
ax1.legend()


ax1 = plt.subplot(3,5,6)


out_matrix_BSL = numpy.zeros((len(output_matrix_5[:,0]),len(w_d13C_glob)))
out_matrix2_BSL = numpy.zeros((len(output_matrix_5[:,0]),len(w_d13C_glob)))

for n in range(0,len(output_matrix_5[:,0])):
	out_matrix_BSL[n,:] = numpy.gradient(output_matrix_5[n,w_d13C_glob],t_vals[w_d13C_glob])
	out_matrix2_BSL[n,:] = numpy.gradient(out_matrix_BSL[n,:],t_vals[w_d13C_glob])
	
mean_BSL_gradient = numpy.nanmean(out_matrix_BSL,axis=0)
std_BSL_gradient = numpy.nanstd(out_matrix_BSL,axis=0)

mean_BSL_gradient2 = numpy.nanmean(out_matrix2_BSL,axis=0)
std_BSL_gradient2 = numpy.nanstd(out_matrix2_BSL,axis=0)

out_mat_fold = 1

ax1.plot(mean_BSL_gradient,t_vals[w_d13C_glob],color='k',linewidth=1.0)
ax1.plot([0,0],[0,0],color='k',linewidth=1.0,label="d(NJ shelf $\delta$$^{13}$C")
###'''
ax1.plot(mean_BSL_gradient + (std_BSL_gradient),t_vals[w_d13C_glob],color='k',linestyle='--',linewidth=0.75)
ax1.plot(mean_BSL_gradient - (std_BSL_gradient),t_vals[w_d13C_glob],color='k',linestyle='--',linewidth=0.75)
ax1.plot(mean_BSL_gradient + (2. * std_BSL_gradient),t_vals[w_d13C_glob],color='k',linestyle=':',linewidth=0.5)
ax1.plot(mean_BSL_gradient - (2. * std_BSL_gradient),t_vals[w_d13C_glob],color='k',linestyle=':',linewidth=0.5)
###'''

ax1.set_yticks(numpy.arange(-30,30,2.), minor=True)

ax1.set_ylim(-30,30)

ax1.set_title("NJ shelf  $\delta$$^{13}$C")
ax1.set_xlabel("$\delta$$^{13}$C/Correlation unit")
ax1.set_ylabel("Correlation units (rel. to CIE)")


ax1.grid(alpha=1.0,which='both')
ax1.legend()


ax1 = plt.subplot(3,5,7)


out_matrix_BSL = numpy.zeros((len(output_matrix_5[:,0]),len(w_d13C_bulk)))
out_matrix2_BSL = numpy.zeros((len(output_matrix_5[:,0]),len(w_d13C_bulk)))

for n in range(0,len(output_matrix_5[:,0])):
	out_matrix_BSL[n,:] = numpy.gradient(output_matrix_5[n,w_d13C_bulk],t_vals[w_d13C_bulk])
	out_matrix2_BSL[n,:] = numpy.gradient(out_matrix_BSL[n,:],t_vals[w_d13C_bulk])
	
mean_BSL_gradient_bulk = numpy.nanmean(out_matrix_BSL,axis=0)
std_BSL_gradient_bulk = numpy.nanstd(out_matrix_BSL,axis=0)

mean_BSL_gradient_bulk2 = numpy.nanmean(out_matrix2_BSL,axis=0)
std_BSL_gradient_bulk2 = numpy.nanstd(out_matrix2_BSL,axis=0)

out_mat_fold = 1

ax1.plot(mean_BSL_gradient_bulk,t_vals[w_d13C_bulk],color='tomato',linewidth=1.0)
ax1.plot([0,0],[0,0],color='tomato',linewidth=1.0,label="NJ $\Delta$$\delta$$^{13}$C$_{bulk}$")

ax1.plot(mean_BSL_gradient_bulk + (std_BSL_gradient_bulk),t_vals[w_d13C_bulk],color='tomato',linestyle='--',linewidth=0.75)
ax1.plot(mean_BSL_gradient_bulk - (std_BSL_gradient_bulk),t_vals[w_d13C_bulk],color='tomato',linestyle='--',linewidth=0.75)
ax1.plot(mean_BSL_gradient_bulk + (2. * std_BSL_gradient_bulk),t_vals[w_d13C_bulk],color='tomato',linestyle=':',linewidth=0.5)
ax1.plot(mean_BSL_gradient_bulk - (2. * std_BSL_gradient_bulk),t_vals[w_d13C_bulk],color='tomato',linestyle=':',linewidth=0.5)

ax1.set_yticks(numpy.arange(-30,30,2.), minor=True)

ax1.set_ylim(-30,30)


ax1.set_title("NJ shelf $\delta$$^{13}$C$_{bulk}$")
ax1.set_xlabel("$\delta$$^{13}$C/Correlation unit")

ax1.grid(alpha=1.0,which='both')
ax1.legend()

ax1 = plt.subplot(3,5,8)


out_matrix_BSL = numpy.zeros((len(output_matrix_5[:,0]),len(w_d13C_org)))
out_matrix2_BSL = numpy.zeros((len(output_matrix_5[:,0]),len(w_d13C_org)))

for n in range(0,len(output_matrix_5[:,0])):
	out_matrix_BSL[n,:] = numpy.gradient(output_matrix_5[n,w_d13C_org],t_vals[w_d13C_org])
	out_matrix2_BSL[n,:] = numpy.gradient(out_matrix_BSL[n,:],t_vals[w_d13C_org])
	
mean_BSL_gradient_org = numpy.nanmean(out_matrix_BSL,axis=0)
std_BSL_gradient_org = numpy.nanstd(out_matrix_BSL,axis=0)

mean_BSL_gradient_org2 = numpy.nanmean(out_matrix2_BSL,axis=0)
std_BSL_gradient_org2 = numpy.nanstd(out_matrix2_BSL,axis=0)

out_mat_fold = 1

ax1.plot(mean_BSL_gradient_org,t_vals[w_d13C_org],color='forestgreen',linewidth=1.0)
ax1.plot([0,0],[0,0],color='forestgreen',linewidth=1.0,label="NJ $\Delta$$\delta$$^{13}$C$_{org}$")

ax1.plot(mean_BSL_gradient_org + (std_BSL_gradient_org),t_vals[w_d13C_org],color='forestgreen',linestyle='--',linewidth=0.75)
ax1.plot(mean_BSL_gradient_org - (std_BSL_gradient_org),t_vals[w_d13C_org],color='forestgreen',linestyle='--',linewidth=0.75)
ax1.plot(mean_BSL_gradient_org + (2. * std_BSL_gradient_org),t_vals[w_d13C_org],color='forestgreen',linestyle=':',linewidth=0.5)
ax1.plot(mean_BSL_gradient_org - (2. * std_BSL_gradient_org),t_vals[w_d13C_org],color='forestgreen',linestyle=':',linewidth=0.5)


ax1.set_yticks(numpy.arange(-30,30,2.), minor=True)

ax1.set_ylim(-30,30)


ax1.set_title("NJ shelf $\delta$$^{13}$C$_{org}$")
ax1.set_xlabel("$\delta$$^{13}$C/Correlation unit")

ax1.grid(alpha=1.0,which='both')
ax1.legend()

ax1 = plt.subplot(3,5,9)


out_matrix_BSL = numpy.zeros((len(output_matrix_5[:,0]),len(w_d13C_PF)))
out_matrix2_BSL = numpy.zeros((len(output_matrix_5[:,0]),len(w_d13C_PF)))

for n in range(0,len(output_matrix_5[:,0])):
	out_matrix_BSL[n,:] = numpy.gradient(output_matrix_5[n,w_d13C_PF],t_vals[w_d13C_PF])
	out_matrix2_BSL[n,:] = numpy.gradient(out_matrix_BSL[n,:],t_vals[w_d13C_PF])
	
mean_BSL_gradient_PF = numpy.nanmean(out_matrix_BSL,axis=0)
std_BSL_gradient_PF = numpy.nanstd(out_matrix_BSL,axis=0)

mean_BSL_gradient_PF2 = numpy.nanmean(out_matrix2_BSL,axis=0)
std_BSL_gradient_PF2 = numpy.nanstd(out_matrix2_BSL,axis=0)

out_mat_fold = 1

ax1.plot(mean_BSL_gradient_PF,t_vals[w_d13C_PF],color='dodgerblue',linewidth=1.0)
ax1.plot([0,0],[0,0],color='dodgerblue',linewidth=1.0,label="NJ $\Delta$$\delta$$^{13}$C$_{PF}$")

ax1.plot(mean_BSL_gradient_PF + (std_BSL_gradient_PF),t_vals[w_d13C_PF],color='dodgerblue',linestyle='--',linewidth=0.75)
ax1.plot(mean_BSL_gradient_PF - (std_BSL_gradient_PF),t_vals[w_d13C_PF],color='dodgerblue',linestyle='--',linewidth=0.75)
ax1.plot(mean_BSL_gradient_PF + (2. * std_BSL_gradient_PF),t_vals[w_d13C_PF],color='dodgerblue',linestyle=':',linewidth=0.5)
ax1.plot(mean_BSL_gradient_PF - (2. * std_BSL_gradient_PF),t_vals[w_d13C_PF],color='dodgerblue',linestyle=':',linewidth=0.5)


ax1.set_yticks(numpy.arange(-30,30,2.), minor=True)

ax1.set_ylim(-30,30)

out_matrix_BSL = numpy.zeros((len(output_matrix_5[:,0]),len(w_d13C_BF)))
out_matrix2_BSL = numpy.zeros((len(output_matrix_5[:,0]),len(w_d13C_BF)))

for n in range(0,len(output_matrix_5[:,0])):
	out_matrix_BSL[n,:] = numpy.gradient(output_matrix_5[n,w_d13C_BF],t_vals[w_d13C_BF])
	out_matrix2_BSL[n,:] = numpy.gradient(out_matrix_BSL[n,:],t_vals[w_d13C_BF])
	
mean_BSL_gradient_BF = numpy.nanmean(out_matrix_BSL,axis=0)
std_BSL_gradient_BF = numpy.nanstd(out_matrix_BSL,axis=0)

mean_BSL_gradient_BF2 = numpy.nanmean(out_matrix2_BSL,axis=0)
std_BSL_gradient_BF2 = numpy.nanstd(out_matrix2_BSL,axis=0)

out_mat_fold = 1

ax1.plot(mean_BSL_gradient_BF,t_vals[w_d13C_BF],color='indigo',linewidth=1.0)
ax1.plot([0,0],[0,0],color='indigo',linewidth=1.0,label="NJ $\Delta$$\delta$$^{13}$C$_{BF}$")

ax1.plot(mean_BSL_gradient_BF + (std_BSL_gradient_BF),t_vals[w_d13C_BF],color='indigo',linestyle='--',linewidth=0.75)
ax1.plot(mean_BSL_gradient_BF - (std_BSL_gradient_BF),t_vals[w_d13C_BF],color='indigo',linestyle='--',linewidth=0.75)
ax1.plot(mean_BSL_gradient_BF + (2. * std_BSL_gradient_BF),t_vals[w_d13C_BF],color='indigo',linestyle=':',linewidth=0.5)
ax1.plot(mean_BSL_gradient_BF - (2. * std_BSL_gradient_BF),t_vals[w_d13C_BF],color='indigo',linestyle=':',linewidth=0.5)

ax1.set_yticks(numpy.arange(-30,30,2.), minor=True)

ax1.set_ylim(-30,30)


ax1.set_title("NJ shelf forams")
ax1.set_xlabel("$\delta$$^{13}$C/Correlation unit")

ax1.grid(alpha=1.0,which='both')
ax1.legend()


ax1 = plt.subplot(3,5,10)


out_matrix_BSL = numpy.zeros((len(output_matrix_5[:,0]),len(w_TEX86)))
out_matrix2_BSL = numpy.zeros((len(output_matrix_5[:,0]),len(w_TEX86)))

for n in range(0,len(output_matrix_5[:,0])):
	out_matrix_BSL[n,:] = numpy.gradient(output_matrix_5[n,w_TEX86],t_vals[w_TEX86])
	out_matrix2_BSL[n,:] = numpy.gradient(out_matrix_BSL[n,:],t_vals[w_TEX86])
	
mean_BSL_gradient_tex = numpy.nanmean(out_matrix_BSL,axis=0)
std_BSL_gradient_tex = numpy.nanstd(out_matrix_BSL,axis=0)

mean_BSL_gradient_tex2 = numpy.nanmean(out_matrix2_BSL,axis=0)
std_BSL_gradient_tex2 = numpy.nanstd(out_matrix2_BSL,axis=0)

out_mat_fold = 1

ax1.plot(mean_BSL_gradient_tex,t_vals[w_TEX86],color='mediumvioletred',linewidth=1.0)
ax1.plot([0,0],[0,0],color='mediumvioletred',linewidth=1.0,label="NJ $\Delta$TEX86")

ax1.plot(mean_BSL_gradient_tex + (std_BSL_gradient_tex),t_vals[w_TEX86],color='mediumvioletred',linestyle='--',linewidth=0.75)
ax1.plot(mean_BSL_gradient_tex - (std_BSL_gradient_tex),t_vals[w_TEX86],color='mediumvioletred',linestyle='--',linewidth=0.75)
ax1.plot(mean_BSL_gradient_tex + (2. * std_BSL_gradient_tex),t_vals[w_TEX86],color='mediumvioletred',linestyle=':',linewidth=0.5)
ax1.plot(mean_BSL_gradient_tex - (2. * std_BSL_gradient_tex),t_vals[w_TEX86],color='mediumvioletred',linestyle=':',linewidth=0.5)


ax1.set_yticks(numpy.arange(-30,30,2.), minor=True)

ax1.set_ylim(-30,30)

ax1.set_title("NJ shelf temperature")
ax1.set_xlabel("$\degree$C/Correlation unit")

ax1.grid(alpha=1.0,which='both')
ax1.legend()

####

ax1 = plt.subplot(3,5,11)


out_matrix_BSL = numpy.zeros((len(output_matrix_5[:,0]),len(w_d13C_glob)))
out_matrix2_BSL = numpy.zeros((len(output_matrix_5[:,0]),len(w_d13C_glob)))

for n in range(0,len(output_matrix_5[:,0])):
	out_matrix_BSL[n,:] = numpy.gradient(output_matrix_5[n,w_d13C_glob],t_vals[w_d13C_glob])
	out_matrix2_BSL[n,:] = numpy.gradient(out_matrix_BSL[n,:],t_vals[w_d13C_glob])
	
mean_BSL_gradient = numpy.nanmean(out_matrix_BSL,axis=0)
std_BSL_gradient = numpy.nanstd(out_matrix_BSL,axis=0)

mean_BSL_gradient2 = numpy.nanmean(out_matrix2_BSL,axis=0)
std_BSL_gradient2 = numpy.nanstd(out_matrix2_BSL,axis=0)
'''
MC_iterations = 0
for n in range(0,MC_iterations):
	
	val2 = numpy.random.normal(loc = mean_BSL_gradient2, scale = std_BSL_gradient2)
	
	ax1.plot((val2),t_vals[w_d13C_glob],color='k',marker = 'x',markersize=2.0,linewidth=0.0,alpha=0.002)
'''
out_mat_fold = 1

ax1.plot(mean_BSL_gradient2,t_vals[w_d13C_glob],color='k',linewidth=1.0)
ax1.plot([0,0],[0,0],color='k',linewidth=1.0,label="NJ shelf $\Delta$$^2$$\delta$$^{13}$C")

ax1.plot(mean_BSL_gradient2 + (std_BSL_gradient2),t_vals[w_d13C_glob],color='k',linestyle='--',linewidth=0.75)
ax1.plot(mean_BSL_gradient2 - (std_BSL_gradient2),t_vals[w_d13C_glob],color='k',linestyle='--',linewidth=0.75)
ax1.plot(mean_BSL_gradient2 + (2. * std_BSL_gradient2),t_vals[w_d13C_glob],color='k',linestyle=':',linewidth=0.5)
ax1.plot(mean_BSL_gradient2 - (2. * std_BSL_gradient2),t_vals[w_d13C_glob],color='k',linestyle=':',linewidth=0.5)


ax1.set_yticks(numpy.arange(-30,30,2.), minor=True)

ax1.set_ylim(-30,30)

ax1.set_title("NJ shelf  $\delta$$^{13}$C")
ax1.set_xlabel("$\delta$$^{13}$C/Correlation unit$^2$")
ax1.set_ylabel("Correlation units (rel. to CIE)")


ax1.grid(alpha=1.0,which='both')
ax1.legend()


ax1 = plt.subplot(3,5,12)


out_matrix_BSL = numpy.zeros((len(output_matrix_5[:,0]),len(w_d13C_bulk)))
out_matrix2_BSL = numpy.zeros((len(output_matrix_5[:,0]),len(w_d13C_bulk)))

for n in range(0,len(output_matrix_5[:,0])):
	out_matrix_BSL[n,:] = numpy.gradient(output_matrix_5[n,w_d13C_bulk],t_vals[w_d13C_bulk])
	out_matrix2_BSL[n,:] = numpy.gradient(out_matrix_BSL[n,:],t_vals[w_d13C_bulk])
	
mean_BSL_gradient_bulk = numpy.nanmean(out_matrix_BSL,axis=0)
std_BSL_gradient_bulk = numpy.nanstd(out_matrix_BSL,axis=0)

mean_BSL_gradient_bulk2 = numpy.nanmean(out_matrix2_BSL,axis=0)
std_BSL_gradient_bulk2 = numpy.nanstd(out_matrix2_BSL,axis=0)

out_mat_fold = 1

ax1.plot(mean_BSL_gradient_bulk2,t_vals[w_d13C_bulk],color='tomato',linewidth=1.0)
ax1.plot([0,0],[0,0],color='tomato',linewidth=1.0,label="NJ $\Delta$$^2$$\delta$$^{13}$C$_{bulk}$")

ax1.plot(mean_BSL_gradient_bulk2 + (std_BSL_gradient_bulk2),t_vals[w_d13C_bulk],color='tomato',linestyle='--',linewidth=0.75)
ax1.plot(mean_BSL_gradient_bulk2 - (std_BSL_gradient_bulk2),t_vals[w_d13C_bulk],color='tomato',linestyle='--',linewidth=0.75)
ax1.plot(mean_BSL_gradient_bulk2 + (2. * std_BSL_gradient_bulk2),t_vals[w_d13C_bulk],color='tomato',linestyle=':',linewidth=0.5)
ax1.plot(mean_BSL_gradient_bulk2 - (2. * std_BSL_gradient_bulk2),t_vals[w_d13C_bulk],color='tomato',linestyle=':',linewidth=0.5)


ax1.set_yticks(numpy.arange(-30,30,2.), minor=True)

ax1.set_ylim(-30,30)


ax1.set_title("NJ shelf $\delta$$^{13}$C$_{bulk}$")
ax1.set_xlabel("$\delta$$^{13}$C/Correlation unit$^2$")

ax1.grid(alpha=1.0,which='both')
ax1.legend()

ax1 = plt.subplot(3,5,13)


out_matrix_BSL = numpy.zeros((len(output_matrix_5[:,0]),len(w_d13C_org)))
out_matrix2_BSL = numpy.zeros((len(output_matrix_5[:,0]),len(w_d13C_org)))

for n in range(0,len(output_matrix_5[:,0])):
	out_matrix_BSL[n,:] = numpy.gradient(output_matrix_5[n,w_d13C_org],t_vals[w_d13C_org])
	out_matrix2_BSL[n,:] = numpy.gradient(out_matrix_BSL[n,:],t_vals[w_d13C_org])
	
mean_BSL_gradient_org = numpy.nanmean(out_matrix_BSL,axis=0)
std_BSL_gradient_org = numpy.nanstd(out_matrix_BSL,axis=0)

mean_BSL_gradient_org2 = numpy.nanmean(out_matrix2_BSL,axis=0)
std_BSL_gradient_org2 = numpy.nanstd(out_matrix2_BSL,axis=0)

out_mat_fold = 1

ax1.plot(mean_BSL_gradient_org2,t_vals[w_d13C_org],color='forestgreen',linewidth=1.0)
ax1.plot([0,0],[0,0],color='forestgreen',linewidth=1.0,label="NJ $\Delta$$^2$$\delta$$^{13}$C$_{org}$")

ax1.plot(mean_BSL_gradient_org2 + (std_BSL_gradient_org2),t_vals[w_d13C_org],color='forestgreen',linestyle='--',linewidth=0.75)
ax1.plot(mean_BSL_gradient_org2 - (std_BSL_gradient_org2),t_vals[w_d13C_org],color='forestgreen',linestyle='--',linewidth=0.75)
ax1.plot(mean_BSL_gradient_org2 + (2. * std_BSL_gradient_org2),t_vals[w_d13C_org],color='forestgreen',linestyle=':',linewidth=0.5)
ax1.plot(mean_BSL_gradient_org2 - (2. * std_BSL_gradient_org2),t_vals[w_d13C_org],color='forestgreen',linestyle=':',linewidth=0.5)


ax1.set_yticks(numpy.arange(-30,30,2.), minor=True)

ax1.set_ylim(-30,30)


ax1.set_title("NJ shelf $\delta$$^{13}$C$_{org}$")
ax1.set_xlabel("$\delta$$^{13}$C/Correlation unit$^2$")

ax1.grid(alpha=1.0,which='both')
ax1.legend()

ax1 = plt.subplot(3,5,14)


out_matrix_BSL = numpy.zeros((len(output_matrix_5[:,0]),len(w_d13C_PF)))
out_matrix2_BSL = numpy.zeros((len(output_matrix_5[:,0]),len(w_d13C_PF)))

for n in range(0,len(output_matrix_5[:,0])):
	out_matrix_BSL[n,:] = numpy.gradient(output_matrix_5[n,w_d13C_PF],t_vals[w_d13C_PF])
	out_matrix2_BSL[n,:] = numpy.gradient(out_matrix_BSL[n,:],t_vals[w_d13C_PF])
	
mean_BSL_gradient_PF = numpy.nanmean(out_matrix_BSL,axis=0)
std_BSL_gradient_PF = numpy.nanstd(out_matrix_BSL,axis=0)

mean_BSL_gradient_PF2 = numpy.nanmean(out_matrix2_BSL,axis=0)
std_BSL_gradient_PF2 = numpy.nanstd(out_matrix2_BSL,axis=0)

out_mat_fold = 1

ax1.plot(mean_BSL_gradient_PF2,t_vals[w_d13C_PF],color='dodgerblue',linewidth=1.0)
ax1.plot([0,0],[0,0],color='dodgerblue',linewidth=1.0,label="NJ $\Delta$$^2$$\delta$$^{13}$C$_{PF}$")

ax1.plot(mean_BSL_gradient_PF2 + (std_BSL_gradient_PF2),t_vals[w_d13C_PF],color='dodgerblue',linestyle='--',linewidth=0.75)
ax1.plot(mean_BSL_gradient_PF2 - (std_BSL_gradient_PF2),t_vals[w_d13C_PF],color='dodgerblue',linestyle='--',linewidth=0.75)
ax1.plot(mean_BSL_gradient_PF2 + (2. * std_BSL_gradient_PF2),t_vals[w_d13C_PF],color='dodgerblue',linestyle=':',linewidth=0.5)
ax1.plot(mean_BSL_gradient_PF2 - (2. * std_BSL_gradient_PF2),t_vals[w_d13C_PF],color='dodgerblue',linestyle=':',linewidth=0.5)

out_matrix_BSL = numpy.zeros((len(output_matrix_5[:,0]),len(w_d13C_BF)))
out_matrix2_BSL = numpy.zeros((len(output_matrix_5[:,0]),len(w_d13C_BF)))

for n in range(0,len(output_matrix_5[:,0])):
	out_matrix_BSL[n,:] = numpy.gradient(output_matrix_5[n,w_d13C_BF],t_vals[w_d13C_BF])
	out_matrix2_BSL[n,:] = numpy.gradient(out_matrix_BSL[n,:],t_vals[w_d13C_BF])
	
mean_BSL_gradient_BF = numpy.nanmean(out_matrix_BSL,axis=0)
std_BSL_gradient_BF = numpy.nanstd(out_matrix_BSL,axis=0)

mean_BSL_gradient_BF2 = numpy.nanmean(out_matrix2_BSL,axis=0)
std_BSL_gradient_BF2 = numpy.nanstd(out_matrix2_BSL,axis=0)

out_mat_fold = 1

ax1.plot(mean_BSL_gradient_BF2,t_vals[w_d13C_BF],color='indigo',linewidth=1.0)
ax1.plot([0,0],[0,0],color='indigo',linewidth=1.0,label="NJ $\Delta$$^2$$\delta$$^{13}$C$_{BF}$")
###'''
ax1.plot(mean_BSL_gradient_BF2 + (std_BSL_gradient_BF2),t_vals[w_d13C_BF],color='indigo',linestyle='--',linewidth=0.75)
ax1.plot(mean_BSL_gradient_BF2 - (std_BSL_gradient_BF2),t_vals[w_d13C_BF],color='indigo',linestyle='--',linewidth=0.75)
ax1.plot(mean_BSL_gradient_BF2 + (2. * std_BSL_gradient_BF2),t_vals[w_d13C_BF],color='indigo',linestyle=':',linewidth=0.5)
ax1.plot(mean_BSL_gradient_BF2 - (2. * std_BSL_gradient_BF2),t_vals[w_d13C_BF],color='indigo',linestyle=':',linewidth=0.5)
###'''

ax1.set_yticks(numpy.arange(-30,30,2.), minor=True)

ax1.set_ylim(-30,30)


ax1.set_title("NJ shelf forams")
ax1.set_xlabel("$\delta$$^{13}$C/Correlation unit$^2$")

ax1.grid(alpha=1.0,which='both')
ax1.legend()

ax1 = plt.subplot(3,5,15)


out_matrix_BSL = numpy.zeros((len(output_matrix_5[:,0]),len(w_TEX86)))
out_matrix2_BSL = numpy.zeros((len(output_matrix_5[:,0]),len(w_TEX86)))

for n in range(0,len(output_matrix_5[:,0])):
	out_matrix_BSL[n,:] = numpy.gradient(output_matrix_5[n,w_TEX86],t_vals[w_TEX86])
	out_matrix2_BSL[n,:] = numpy.gradient(out_matrix_BSL[n,:],t_vals[w_TEX86])
	
mean_BSL_gradient_tex = numpy.nanmean(out_matrix_BSL,axis=0)
std_BSL_gradient_tex = numpy.nanstd(out_matrix_BSL,axis=0)

mean_BSL_gradient_tex2 = numpy.nanmean(out_matrix2_BSL,axis=0)
std_BSL_gradient_tex2 = numpy.nanstd(out_matrix2_BSL,axis=0)

out_mat_fold = 1

ax1.plot(mean_BSL_gradient_tex2,t_vals[w_TEX86],color='mediumvioletred',linewidth=1.0)
ax1.plot([0,0],[0,0],color='mediumvioletred',linewidth=1.0,label="NJ $\Delta$$^2$TEX86")

ax1.plot(mean_BSL_gradient_tex2 + (std_BSL_gradient_tex2),t_vals[w_TEX86],color='mediumvioletred',linestyle='--',linewidth=0.75)
ax1.plot(mean_BSL_gradient_tex2 - (std_BSL_gradient_tex2),t_vals[w_TEX86],color='mediumvioletred',linestyle='--',linewidth=0.75)
ax1.plot(mean_BSL_gradient_tex2 + (2. * std_BSL_gradient_tex2),t_vals[w_TEX86],color='mediumvioletred',linestyle=':',linewidth=0.5)
ax1.plot(mean_BSL_gradient_tex2 - (2. * std_BSL_gradient_tex2),t_vals[w_TEX86],color='mediumvioletred',linestyle=':',linewidth=0.5)


ax1.set_yticks(numpy.arange(-30,30,2.), minor=True)

ax1.set_ylim(-30,30)


ax1.set_title("NJ shelf temperature")
ax1.set_xlabel("$\degree$C/Correlation unit$^2$")

ax1.grid(alpha=1.0,which='both')
ax1.legend(loc=2)

plt.tight_layout()

pltname = wkspc +  'NJ_d13C_derivative_20210604_1.png'

plt.savefig(pltname, dpi = 300)
plt.close()

fig= plt.figure(3,figsize=(20.,16.5))

ax1 = plt.subplot(3,5,1)

out_mat_fold = 1

ax1.plot(mean_global,t_vals[w_d13C_glob],'k',linewidth=1.0)
ax1.plot([0,0],[0,0],'k',linewidth=1.0,label="NJ shelf $\delta$$^{13}$C")
ax1.plot(mean_global + (std_global),t_vals[w_d13C_glob],'k--',linewidth=0.75)
ax1.plot(mean_global - (std_global),t_vals[w_d13C_glob],'k--',linewidth=0.75)
ax1.plot(mean_global + (2. * std_global),t_vals[w_d13C_glob],'k:',linewidth=0.5)
ax1.plot(mean_global - (2. * std_global),t_vals[w_d13C_glob],'k:',linewidth=0.5)

ax1.set_yticks(numpy.arange(-30,30,2.), minor=True)

ax1.set_ylim(-30,30)

ax1.set_xlim(-5.,3.5)

ax1.grid(alpha=0.0,which='both')
ax1.legend()

ax1.set_title("NJ shelf  $\delta$$^{13}$C")
ax1.set_xlabel("$\delta$$^{13}$C")
ax1.set_ylabel("Correlation units (rel. to CIE)")

ax1 = plt.subplot(3,5,2)


out_mat_fold = 1

ax1.plot(SL_data_combined[w_d13C_bulk_c,1],mean_timing[w_d13C_bulk_c],color='tomato',marker='o',linewidth=0.0,markersize=2.0,alpha=0.15,label="NJ $\delta$$^{13}$C$_{bulk}$ data")

ax1.plot(mean_NJ_d13C_bulk,t_vals[w_d13C_glob],color='tomato',linewidth=1.0)
ax1.plot([0,0],[0,0],color='tomato',linewidth=1.0,label="$\delta$$^{13}$C$_{bulk}$")
ax1.plot(mean_NJ_d13C_bulk + (std_NJ_d13C_bulk),t_vals[w_d13C_glob],color='tomato',linestyle='--',linewidth=0.75)
ax1.plot(mean_NJ_d13C_bulk - (std_NJ_d13C_bulk),t_vals[w_d13C_glob],color='tomato',linestyle='--',linewidth=0.75)
ax1.plot(mean_NJ_d13C_bulk + (2. * std_NJ_d13C_bulk),t_vals[w_d13C_glob],color='tomato',linestyle=':',linewidth=0.5)
ax1.plot(mean_NJ_d13C_bulk - (2. * std_NJ_d13C_bulk),t_vals[w_d13C_glob],color='tomato',linestyle=':',linewidth=0.5)

ax1.set_yticks(numpy.arange(-30,30,2.), minor=True)

ax1.set_ylim(-30,30)

ax1.set_xlim(-5.,3.5)

ax1.grid(alpha=0.0,which='both')
ax1.legend()

ax1.set_title("NJ shelf $\delta$$^{13}$C$_{bulk}$")
ax1.set_xlabel("$\delta$$^{13}$C")

ax1 = plt.subplot(3,5,3)


out_mat_fold = 1

ax1.plot(SL_data_combined[w_d13C_org_c,1],mean_timing[w_d13C_org_c],color='forestgreen',marker='o',linewidth=0.0,markersize=2.0,alpha=0.15,label="NJ $\delta$$^{13}$C$_{org}$ data")

ax1.plot(mean_NJ_d13C_org,t_vals[w_d13C_glob],color='forestgreen',linewidth=1.0)
ax1.plot([0,0],[0,0],color='forestgreen',linewidth=1.0,label="$\delta$$^{13}$C$_{org}$")
ax1.plot(mean_NJ_d13C_org + (std_NJ_d13C_org),t_vals[w_d13C_glob],color='forestgreen',linestyle='--',linewidth=0.75)
ax1.plot(mean_NJ_d13C_org - (std_NJ_d13C_org),t_vals[w_d13C_glob],color='forestgreen',linestyle='--',linewidth=0.75)
ax1.plot(mean_NJ_d13C_org + (2. * std_NJ_d13C_org),t_vals[w_d13C_glob],color='forestgreen',linestyle=':',linewidth=0.5)
ax1.plot(mean_NJ_d13C_org - (2. * std_NJ_d13C_org),t_vals[w_d13C_glob],color='forestgreen',linestyle=':',linewidth=0.5)

ax1.set_yticks(numpy.arange(-30,30,2.), minor=True)

ax1.set_ylim(-30,30)

ax1.set_xlim(-31.,-22.)

ax1.set_title("NJ shelf $\delta$$^{13}$C$_{org}$")
ax1.set_xlabel("$\delta$$^{13}$C")

ax1.grid(alpha=0.0,which='both')
ax1.legend()

ax1 = plt.subplot(3,5,4)


out_mat_fold = 1

ax1.plot(SL_data_combined[w_d13C_PF_c,1],mean_timing[w_d13C_PF_c],color='dodgerblue',marker='o',linewidth=0.0,markersize=2.0,alpha=0.15,label="NJ $\delta$$^{13}$C$_{PF}$ data")

ax1.plot(mean_NJ_d13C_PF,t_vals[w_d13C_glob],color='dodgerblue',linewidth=1.0)
ax1.plot([0,0],[0,0],color='dodgerblue',linewidth=1.0,label="$\delta$$^{13}$C$_{PF}$")
ax1.plot(mean_NJ_d13C_PF + (std_NJ_d13C_PF),t_vals[w_d13C_glob],color='dodgerblue',linestyle='--',linewidth=0.75)
ax1.plot(mean_NJ_d13C_PF - (std_NJ_d13C_PF),t_vals[w_d13C_glob],color='dodgerblue',linestyle='--',linewidth=0.75)
ax1.plot(mean_NJ_d13C_PF + (2. * std_NJ_d13C_PF),t_vals[w_d13C_glob],color='dodgerblue',linestyle=':',linewidth=0.5)
ax1.plot(mean_NJ_d13C_PF - (2. * std_NJ_d13C_PF),t_vals[w_d13C_glob],color='dodgerblue',linestyle=':',linewidth=0.5)

out_mat_fold = 1

ax1.plot(SL_data_combined[w_d13C_BF_c,1],mean_timing[w_d13C_BF_c],color='indigo',marker='o',linewidth=0.0,markersize=2.0,alpha=0.15,label="NJ $\delta$$^{13}$C$_{BF}$ data")

ax1.plot(mean_NJ_d13C_BF,t_vals[w_d13C_glob],color='indigo',linewidth=1.0)
ax1.plot([0,0],[0,0],color='indigo',linewidth=1.0,label="$\delta$$^{13}$C$_{BF}$")
ax1.plot(mean_NJ_d13C_BF + (std_NJ_d13C_BF),t_vals[w_d13C_glob],color='indigo',linestyle='--',linewidth=0.75)
ax1.plot(mean_NJ_d13C_BF - (std_NJ_d13C_BF),t_vals[w_d13C_glob],color='indigo',linestyle='--',linewidth=0.75)
ax1.plot(mean_NJ_d13C_BF + (2. * std_NJ_d13C_BF),t_vals[w_d13C_glob],color='indigo',linestyle=':',linewidth=0.5)
ax1.plot(mean_NJ_d13C_BF - (2. * std_NJ_d13C_BF),t_vals[w_d13C_glob],color='indigo',linestyle=':',linewidth=0.5)

ax1.set_yticks(numpy.arange(-30,30,2.), minor=True)

ax1.set_ylim(-30,30)

ax1.set_xlim(-5.5,4.)

ax1.set_title("NJ shelf forams")
ax1.set_xlabel("$\delta$$^{13}$C")

ax1.grid(alpha=0.0,which='both')
ax1.legend()

ax1 = plt.subplot(3,5,5)

out_mat_fold = 1

ax1.plot(SL_data_combined[w_TEX86_c,1],mean_timing[w_TEX86_c],color='mediumvioletred',marker='o',linewidth=0.0,markersize=2.0,alpha=0.15,label="TEX86")

ax1.plot(mean_NJ_TEX86,t_vals[w_d13C_glob],color='mediumvioletred',linewidth=1.0)
ax1.plot([0,0],[0,0],color='mediumvioletred',linewidth=1.0,label="TEX86")
ax1.plot(mean_NJ_TEX86 + (std_NJ_TEX86),t_vals[w_d13C_glob],color='mediumvioletred',linestyle='--',linewidth=0.75)
ax1.plot(mean_NJ_TEX86 - (std_NJ_TEX86),t_vals[w_d13C_glob],color='mediumvioletred',linestyle='--',linewidth=0.75)
ax1.plot(mean_NJ_TEX86 + (2. * std_NJ_TEX86),t_vals[w_d13C_glob],color='mediumvioletred',linestyle=':',linewidth=0.5)
ax1.plot(mean_NJ_TEX86 - (2. * std_NJ_TEX86),t_vals[w_d13C_glob],color='mediumvioletred',linestyle=':',linewidth=0.5)

ax1.set_yticks(numpy.arange(-30,30,2.), minor=True)

ax1.set_ylim(-30,30)

ax1.set_xlim(26.75,37.51)

ax1.set_title("NJ shelf temperature")
ax1.set_xlabel("Temperature ($\degree$C)")

ax1.grid(alpha=0.0,which='both')
ax1.legend()


ax1 = plt.subplot(3,5,6)


out_matrix_BSL = numpy.zeros((len(output_matrix_5[:,0]),len(w_d13C_glob)))
out_matrix2_BSL = numpy.zeros((len(output_matrix_5[:,0]),len(w_d13C_glob)))

for n in range(0,len(output_matrix_5[:,0])):
	out_matrix_BSL[n,:] = numpy.gradient(output_matrix_5[n,w_d13C_glob],t_vals[w_d13C_glob])
	out_matrix2_BSL[n,:] = numpy.gradient(out_matrix_BSL[n,:],t_vals[w_d13C_glob])
	
mean_BSL_gradient = numpy.nanmean(out_matrix_BSL,axis=0)
std_BSL_gradient = numpy.nanstd(out_matrix_BSL,axis=0)

mean_BSL_gradient2 = numpy.nanmean(out_matrix2_BSL,axis=0)
std_BSL_gradient2 = numpy.nanstd(out_matrix2_BSL,axis=0)

out_mat_fold = 1

ax1.plot(mean_BSL_gradient,t_vals[w_d13C_glob],color='k',linewidth=1.0)
ax1.plot([0,0],[0,0],color='k',linewidth=1.0,label="d(NJ shelf $\delta$$^{13}$C")

ax1.plot(mean_BSL_gradient + (std_BSL_gradient),t_vals[w_d13C_glob],color='k',linestyle='--',linewidth=0.75)
ax1.plot(mean_BSL_gradient - (std_BSL_gradient),t_vals[w_d13C_glob],color='k',linestyle='--',linewidth=0.75)
ax1.plot(mean_BSL_gradient + (2. * std_BSL_gradient),t_vals[w_d13C_glob],color='k',linestyle=':',linewidth=0.5)
ax1.plot(mean_BSL_gradient - (2. * std_BSL_gradient),t_vals[w_d13C_glob],color='k',linestyle=':',linewidth=0.5)


ax1.set_yticks(numpy.arange(-30,30,2.), minor=True)

ax1.set_ylim(-30,30)

#ax1.set_xlim(-300,150)

ax1.set_title("NJ shelf  $\delta$$^{13}$C")
ax1.set_xlabel("$\delta$$^{13}$C/Correlation unit")
ax1.set_ylabel("Correlation units (rel. to CIE)")


ax1.grid(alpha=0.0,which='both')
ax1.legend()


ax1 = plt.subplot(3,5,7)


out_matrix_BSL = numpy.zeros((len(output_matrix_5[:,0]),len(w_d13C_bulk)))
out_matrix2_BSL = numpy.zeros((len(output_matrix_5[:,0]),len(w_d13C_bulk)))

for n in range(0,len(output_matrix_5[:,0])):
	out_matrix_BSL[n,:] = numpy.gradient(output_matrix_5[n,w_d13C_bulk],t_vals[w_d13C_bulk])
	out_matrix2_BSL[n,:] = numpy.gradient(out_matrix_BSL[n,:],t_vals[w_d13C_bulk])
	
mean_BSL_gradient_bulk = numpy.nanmean(out_matrix_BSL,axis=0)
std_BSL_gradient_bulk = numpy.nanstd(out_matrix_BSL,axis=0)

mean_BSL_gradient_bulk2 = numpy.nanmean(out_matrix2_BSL,axis=0)
std_BSL_gradient_bulk2 = numpy.nanstd(out_matrix2_BSL,axis=0)

out_mat_fold = 1

ax1.plot(mean_BSL_gradient_bulk,t_vals[w_d13C_bulk],color='tomato',linewidth=1.0)
ax1.plot([0,0],[0,0],color='tomato',linewidth=1.0,label="NJ $\Delta$$\delta$$^{13}$C$_{bulk}$")

ax1.plot(mean_BSL_gradient_bulk + (std_BSL_gradient_bulk),t_vals[w_d13C_bulk],color='tomato',linestyle='--',linewidth=0.75)
ax1.plot(mean_BSL_gradient_bulk - (std_BSL_gradient_bulk),t_vals[w_d13C_bulk],color='tomato',linestyle='--',linewidth=0.75)
ax1.plot(mean_BSL_gradient_bulk + (2. * std_BSL_gradient_bulk),t_vals[w_d13C_bulk],color='tomato',linestyle=':',linewidth=0.5)
ax1.plot(mean_BSL_gradient_bulk - (2. * std_BSL_gradient_bulk),t_vals[w_d13C_bulk],color='tomato',linestyle=':',linewidth=0.5)


ax1.set_yticks(numpy.arange(-30,30,2.), minor=True)

ax1.set_ylim(-30,30)


ax1.set_title("NJ shelf $\delta$$^{13}$C$_{bulk}$")
ax1.set_xlabel("$\delta$$^{13}$C/Correlation unit")

ax1.grid(alpha=0.0,which='both')
ax1.legend()

ax1 = plt.subplot(3,5,8)


out_matrix_BSL = numpy.zeros((len(output_matrix_5[:,0]),len(w_d13C_org)))
out_matrix2_BSL = numpy.zeros((len(output_matrix_5[:,0]),len(w_d13C_org)))

for n in range(0,len(output_matrix_5[:,0])):
	out_matrix_BSL[n,:] = numpy.gradient(output_matrix_5[n,w_d13C_org],t_vals[w_d13C_org])
	out_matrix2_BSL[n,:] = numpy.gradient(out_matrix_BSL[n,:],t_vals[w_d13C_org])
	
mean_BSL_gradient_org = numpy.nanmean(out_matrix_BSL,axis=0)
std_BSL_gradient_org = numpy.nanstd(out_matrix_BSL,axis=0)

mean_BSL_gradient_org2 = numpy.nanmean(out_matrix2_BSL,axis=0)
std_BSL_gradient_org2 = numpy.nanstd(out_matrix2_BSL,axis=0)

out_mat_fold = 1

ax1.plot(mean_BSL_gradient_org,t_vals[w_d13C_org],color='forestgreen',linewidth=1.0)
ax1.plot([0,0],[0,0],color='forestgreen',linewidth=1.0,label="NJ $\Delta$$\delta$$^{13}$C$_{org}$")

ax1.plot(mean_BSL_gradient_org + (std_BSL_gradient_org),t_vals[w_d13C_org],color='forestgreen',linestyle='--',linewidth=0.75)
ax1.plot(mean_BSL_gradient_org - (std_BSL_gradient_org),t_vals[w_d13C_org],color='forestgreen',linestyle='--',linewidth=0.75)
ax1.plot(mean_BSL_gradient_org + (2. * std_BSL_gradient_org),t_vals[w_d13C_org],color='forestgreen',linestyle=':',linewidth=0.5)
ax1.plot(mean_BSL_gradient_org - (2. * std_BSL_gradient_org),t_vals[w_d13C_org],color='forestgreen',linestyle=':',linewidth=0.5)


ax1.set_yticks(numpy.arange(-30,30,2.), minor=True)

ax1.set_ylim(-30,30)


ax1.set_title("NJ shelf $\delta$$^{13}$C$_{org}$")
ax1.set_xlabel("$\delta$$^{13}$C/Correlation unit")

ax1.grid(alpha=0.0,which='both')
ax1.legend()

ax1 = plt.subplot(3,5,9)


out_matrix_BSL = numpy.zeros((len(output_matrix_5[:,0]),len(w_d13C_PF)))
out_matrix2_BSL = numpy.zeros((len(output_matrix_5[:,0]),len(w_d13C_PF)))

for n in range(0,len(output_matrix_5[:,0])):
	out_matrix_BSL[n,:] = numpy.gradient(output_matrix_5[n,w_d13C_PF],t_vals[w_d13C_PF])
	out_matrix2_BSL[n,:] = numpy.gradient(out_matrix_BSL[n,:],t_vals[w_d13C_PF])
	
mean_BSL_gradient_PF = numpy.nanmean(out_matrix_BSL,axis=0)
std_BSL_gradient_PF = numpy.nanstd(out_matrix_BSL,axis=0)

mean_BSL_gradient_PF2 = numpy.nanmean(out_matrix2_BSL,axis=0)
std_BSL_gradient_PF2 = numpy.nanstd(out_matrix2_BSL,axis=0)

out_mat_fold = 1

ax1.plot(mean_BSL_gradient_PF,t_vals[w_d13C_PF],color='dodgerblue',linewidth=1.0)
ax1.plot([0,0],[0,0],color='dodgerblue',linewidth=1.0,label="NJ $\Delta$$\delta$$^{13}$C$_{PF}$")

ax1.plot(mean_BSL_gradient_PF + (std_BSL_gradient_PF),t_vals[w_d13C_PF],color='dodgerblue',linestyle='--',linewidth=0.75)
ax1.plot(mean_BSL_gradient_PF - (std_BSL_gradient_PF),t_vals[w_d13C_PF],color='dodgerblue',linestyle='--',linewidth=0.75)
ax1.plot(mean_BSL_gradient_PF + (2. * std_BSL_gradient_PF),t_vals[w_d13C_PF],color='dodgerblue',linestyle=':',linewidth=0.5)
ax1.plot(mean_BSL_gradient_PF - (2. * std_BSL_gradient_PF),t_vals[w_d13C_PF],color='dodgerblue',linestyle=':',linewidth=0.5)


ax1.set_yticks(numpy.arange(-30,30,2.), minor=True)

ax1.set_ylim(-30,30)



out_matrix_BSL = numpy.zeros((len(output_matrix_5[:,0]),len(w_d13C_BF)))
out_matrix2_BSL = numpy.zeros((len(output_matrix_5[:,0]),len(w_d13C_BF)))

for n in range(0,len(output_matrix_5[:,0])):
	out_matrix_BSL[n,:] = numpy.gradient(output_matrix_5[n,w_d13C_BF],t_vals[w_d13C_BF])
	out_matrix2_BSL[n,:] = numpy.gradient(out_matrix_BSL[n,:],t_vals[w_d13C_BF])
	
mean_BSL_gradient_BF = numpy.nanmean(out_matrix_BSL,axis=0)
std_BSL_gradient_BF = numpy.nanstd(out_matrix_BSL,axis=0)

mean_BSL_gradient_BF2 = numpy.nanmean(out_matrix2_BSL,axis=0)
std_BSL_gradient_BF2 = numpy.nanstd(out_matrix2_BSL,axis=0)

out_mat_fold = 1

ax1.plot(mean_BSL_gradient_BF,t_vals[w_d13C_BF],color='indigo',linewidth=1.0)
ax1.plot([0,0],[0,0],color='indigo',linewidth=1.0,label="NJ $\Delta$$\delta$$^{13}$C$_{BF}$")
###'''
ax1.plot(mean_BSL_gradient_BF + (std_BSL_gradient_BF),t_vals[w_d13C_BF],color='indigo',linestyle='--',linewidth=0.75)
ax1.plot(mean_BSL_gradient_BF - (std_BSL_gradient_BF),t_vals[w_d13C_BF],color='indigo',linestyle='--',linewidth=0.75)
ax1.plot(mean_BSL_gradient_BF + (2. * std_BSL_gradient_BF),t_vals[w_d13C_BF],color='indigo',linestyle=':',linewidth=0.5)
ax1.plot(mean_BSL_gradient_BF - (2. * std_BSL_gradient_BF),t_vals[w_d13C_BF],color='indigo',linestyle=':',linewidth=0.5)
###'''

ax1.set_yticks(numpy.arange(-30,30,2.), minor=True)

ax1.set_ylim(-30,30)

ax1.set_title("NJ shelf forams")
ax1.set_xlabel("$\delta$$^{13}$C/Correlation unit")

ax1.grid(alpha=0.0,which='both')
ax1.legend()


ax1 = plt.subplot(3,5,10)


out_matrix_BSL = numpy.zeros((len(output_matrix_5[:,0]),len(w_TEX86)))
out_matrix2_BSL = numpy.zeros((len(output_matrix_5[:,0]),len(w_TEX86)))

for n in range(0,len(output_matrix_5[:,0])):
	out_matrix_BSL[n,:] = numpy.gradient(output_matrix_5[n,w_TEX86],t_vals[w_TEX86])
	out_matrix2_BSL[n,:] = numpy.gradient(out_matrix_BSL[n,:],t_vals[w_TEX86])
	
mean_BSL_gradient_tex = numpy.nanmean(out_matrix_BSL,axis=0)
std_BSL_gradient_tex = numpy.nanstd(out_matrix_BSL,axis=0)

mean_BSL_gradient_tex2 = numpy.nanmean(out_matrix2_BSL,axis=0)
std_BSL_gradient_tex2 = numpy.nanstd(out_matrix2_BSL,axis=0)

out_mat_fold = 1

ax1.plot(mean_BSL_gradient_tex,t_vals[w_TEX86],color='mediumvioletred',linewidth=1.0)
ax1.plot([0,0],[0,0],color='mediumvioletred',linewidth=1.0,label="NJ $\Delta$TEX86")

ax1.plot(mean_BSL_gradient_tex + (std_BSL_gradient_tex),t_vals[w_TEX86],color='mediumvioletred',linestyle='--',linewidth=0.75)
ax1.plot(mean_BSL_gradient_tex - (std_BSL_gradient_tex),t_vals[w_TEX86],color='mediumvioletred',linestyle='--',linewidth=0.75)
ax1.plot(mean_BSL_gradient_tex + (2. * std_BSL_gradient_tex),t_vals[w_TEX86],color='mediumvioletred',linestyle=':',linewidth=0.5)
ax1.plot(mean_BSL_gradient_tex - (2. * std_BSL_gradient_tex),t_vals[w_TEX86],color='mediumvioletred',linestyle=':',linewidth=0.5)


ax1.set_yticks(numpy.arange(-30,30,2.), minor=True)

ax1.set_ylim(-30,30)

ax1.set_title("NJ shelf temperature")
ax1.set_xlabel("$\degree$C/Correlation unit")

ax1.grid(alpha=0.0,which='both')
ax1.legend()

####

ax1 = plt.subplot(3,5,11)


out_matrix_BSL = numpy.zeros((len(output_matrix_5[:,0]),len(w_d13C_glob)))
out_matrix2_BSL = numpy.zeros((len(output_matrix_5[:,0]),len(w_d13C_glob)))

for n in range(0,len(output_matrix_5[:,0])):
	out_matrix_BSL[n,:] = numpy.gradient(output_matrix_5[n,w_d13C_glob],t_vals[w_d13C_glob])
	out_matrix2_BSL[n,:] = numpy.gradient(out_matrix_BSL[n,:],t_vals[w_d13C_glob])
	
mean_BSL_gradient = numpy.nanmean(out_matrix_BSL,axis=0)
std_BSL_gradient = numpy.nanstd(out_matrix_BSL,axis=0)

mean_BSL_gradient2 = numpy.nanmean(out_matrix2_BSL,axis=0)
std_BSL_gradient2 = numpy.nanstd(out_matrix2_BSL,axis=0)

out_mat_fold = 1

ax1.plot(mean_BSL_gradient2,t_vals[w_d13C_glob],color='k',linewidth=1.0)
ax1.plot([0,0],[0,0],color='k',linewidth=1.0,label="NJ shelf $\Delta$$^2$$\delta$$^{13}$C")

ax1.plot(mean_BSL_gradient2 + (std_BSL_gradient2),t_vals[w_d13C_glob],color='k',linestyle='--',linewidth=0.75)
ax1.plot(mean_BSL_gradient2 - (std_BSL_gradient2),t_vals[w_d13C_glob],color='k',linestyle='--',linewidth=0.75)
ax1.plot(mean_BSL_gradient2 + (2. * std_BSL_gradient2),t_vals[w_d13C_glob],color='k',linestyle=':',linewidth=0.5)
ax1.plot(mean_BSL_gradient2 - (2. * std_BSL_gradient2),t_vals[w_d13C_glob],color='k',linestyle=':',linewidth=0.5)


ax1.set_yticks(numpy.arange(-30,30,2.), minor=True)

ax1.set_ylim(-30,30)

ax1.set_title("NJ shelf  $\delta$$^{13}$C")
ax1.set_xlabel("$\delta$$^{13}$C/Correlation unit$^2$")
ax1.set_ylabel("Correlation units (rel. to CIE)")


ax1.grid(alpha=0.0,which='both')
ax1.legend()


ax1 = plt.subplot(3,5,12)


out_matrix_BSL = numpy.zeros((len(output_matrix_5[:,0]),len(w_d13C_bulk)))
out_matrix2_BSL = numpy.zeros((len(output_matrix_5[:,0]),len(w_d13C_bulk)))

for n in range(0,len(output_matrix_5[:,0])):
	out_matrix_BSL[n,:] = numpy.gradient(output_matrix_5[n,w_d13C_bulk],t_vals[w_d13C_bulk])
	out_matrix2_BSL[n,:] = numpy.gradient(out_matrix_BSL[n,:],t_vals[w_d13C_bulk])
	
mean_BSL_gradient_bulk = numpy.nanmean(out_matrix_BSL,axis=0)
std_BSL_gradient_bulk = numpy.nanstd(out_matrix_BSL,axis=0)

mean_BSL_gradient_bulk2 = numpy.nanmean(out_matrix2_BSL,axis=0)
std_BSL_gradient_bulk2 = numpy.nanstd(out_matrix2_BSL,axis=0)

out_mat_fold = 1

ax1.plot(mean_BSL_gradient_bulk2,t_vals[w_d13C_bulk],color='tomato',linewidth=1.0)
ax1.plot([0,0],[0,0],color='tomato',linewidth=1.0,label="NJ $\Delta$$^2$$\delta$$^{13}$C$_{bulk}$")

ax1.plot(mean_BSL_gradient_bulk2 + (std_BSL_gradient_bulk2),t_vals[w_d13C_bulk],color='tomato',linestyle='--',linewidth=0.75)
ax1.plot(mean_BSL_gradient_bulk2 - (std_BSL_gradient_bulk2),t_vals[w_d13C_bulk],color='tomato',linestyle='--',linewidth=0.75)
ax1.plot(mean_BSL_gradient_bulk2 + (2. * std_BSL_gradient_bulk2),t_vals[w_d13C_bulk],color='tomato',linestyle=':',linewidth=0.5)
ax1.plot(mean_BSL_gradient_bulk2 - (2. * std_BSL_gradient_bulk2),t_vals[w_d13C_bulk],color='tomato',linestyle=':',linewidth=0.5)


ax1.set_yticks(numpy.arange(-30,30,2.), minor=True)

ax1.set_ylim(-30,30)

ax1.set_title("NJ shelf $\delta$$^{13}$C$_{bulk}$")
ax1.set_xlabel("$\delta$$^{13}$C/Correlation unit$^2$")

ax1.grid(alpha=0.0,which='both')
ax1.legend()

ax1 = plt.subplot(3,5,13)


out_matrix_BSL = numpy.zeros((len(output_matrix_5[:,0]),len(w_d13C_org)))
out_matrix2_BSL = numpy.zeros((len(output_matrix_5[:,0]),len(w_d13C_org)))

for n in range(0,len(output_matrix_5[:,0])):
	out_matrix_BSL[n,:] = numpy.gradient(output_matrix_5[n,w_d13C_org],t_vals[w_d13C_org])
	out_matrix2_BSL[n,:] = numpy.gradient(out_matrix_BSL[n,:],t_vals[w_d13C_org])
	
mean_BSL_gradient_org = numpy.nanmean(out_matrix_BSL,axis=0)
std_BSL_gradient_org = numpy.nanstd(out_matrix_BSL,axis=0)

mean_BSL_gradient_org2 = numpy.nanmean(out_matrix2_BSL,axis=0)
std_BSL_gradient_org2 = numpy.nanstd(out_matrix2_BSL,axis=0)

out_mat_fold = 1

ax1.plot(mean_BSL_gradient_org2,t_vals[w_d13C_org],color='forestgreen',linewidth=1.0)
ax1.plot([0,0],[0,0],color='forestgreen',linewidth=1.0,label="NJ $\Delta$$^2$$\delta$$^{13}$C$_{org}$")

ax1.plot(mean_BSL_gradient_org2 + (std_BSL_gradient_org2),t_vals[w_d13C_org],color='forestgreen',linestyle='--',linewidth=0.75)
ax1.plot(mean_BSL_gradient_org2 - (std_BSL_gradient_org2),t_vals[w_d13C_org],color='forestgreen',linestyle='--',linewidth=0.75)
ax1.plot(mean_BSL_gradient_org2 + (2. * std_BSL_gradient_org2),t_vals[w_d13C_org],color='forestgreen',linestyle=':',linewidth=0.5)
ax1.plot(mean_BSL_gradient_org2 - (2. * std_BSL_gradient_org2),t_vals[w_d13C_org],color='forestgreen',linestyle=':',linewidth=0.5)


ax1.set_yticks(numpy.arange(-30,30,2.), minor=True)

ax1.set_ylim(-30,30)

ax1.set_title("NJ shelf $\delta$$^{13}$C$_{org}$")
ax1.set_xlabel("$\delta$$^{13}$C/Correlation unit$^2$")

ax1.grid(alpha=0.0,which='both')
ax1.legend()

ax1 = plt.subplot(3,5,14)


out_matrix_BSL = numpy.zeros((len(output_matrix_5[:,0]),len(w_d13C_PF)))
out_matrix2_BSL = numpy.zeros((len(output_matrix_5[:,0]),len(w_d13C_PF)))

for n in range(0,len(output_matrix_5[:,0])):
	out_matrix_BSL[n,:] = numpy.gradient(output_matrix_5[n,w_d13C_PF],t_vals[w_d13C_PF])
	out_matrix2_BSL[n,:] = numpy.gradient(out_matrix_BSL[n,:],t_vals[w_d13C_PF])
	
mean_BSL_gradient_PF = numpy.nanmean(out_matrix_BSL,axis=0)
std_BSL_gradient_PF = numpy.nanstd(out_matrix_BSL,axis=0)

mean_BSL_gradient_PF2 = numpy.nanmean(out_matrix2_BSL,axis=0)
std_BSL_gradient_PF2 = numpy.nanstd(out_matrix2_BSL,axis=0)

out_mat_fold = 1

ax1.plot(mean_BSL_gradient_PF2,t_vals[w_d13C_PF],color='dodgerblue',linewidth=1.0)
ax1.plot([0,0],[0,0],color='dodgerblue',linewidth=1.0,label="NJ $\Delta$$^2$$\delta$$^{13}$C$_{PF}$")

ax1.plot(mean_BSL_gradient_PF2 + (std_BSL_gradient_PF2),t_vals[w_d13C_PF],color='dodgerblue',linestyle='--',linewidth=0.75)
ax1.plot(mean_BSL_gradient_PF2 - (std_BSL_gradient_PF2),t_vals[w_d13C_PF],color='dodgerblue',linestyle='--',linewidth=0.75)
ax1.plot(mean_BSL_gradient_PF2 + (2. * std_BSL_gradient_PF2),t_vals[w_d13C_PF],color='dodgerblue',linestyle=':',linewidth=0.5)
ax1.plot(mean_BSL_gradient_PF2 - (2. * std_BSL_gradient_PF2),t_vals[w_d13C_PF],color='dodgerblue',linestyle=':',linewidth=0.5)

out_matrix_BSL = numpy.zeros((len(output_matrix_5[:,0]),len(w_d13C_BF)))
out_matrix2_BSL = numpy.zeros((len(output_matrix_5[:,0]),len(w_d13C_BF)))

for n in range(0,len(output_matrix_5[:,0])):
	out_matrix_BSL[n,:] = numpy.gradient(output_matrix_5[n,w_d13C_BF],t_vals[w_d13C_BF])
	out_matrix2_BSL[n,:] = numpy.gradient(out_matrix_BSL[n,:],t_vals[w_d13C_BF])
	
mean_BSL_gradient_BF = numpy.nanmean(out_matrix_BSL,axis=0)
std_BSL_gradient_BF = numpy.nanstd(out_matrix_BSL,axis=0)

mean_BSL_gradient_BF2 = numpy.nanmean(out_matrix2_BSL,axis=0)
std_BSL_gradient_BF2 = numpy.nanstd(out_matrix2_BSL,axis=0)

out_mat_fold = 1

ax1.plot(mean_BSL_gradient_BF2,t_vals[w_d13C_BF],color='indigo',linewidth=1.0)
ax1.plot([0,0],[0,0],color='indigo',linewidth=1.0,label="NJ $\Delta$$^2$$\delta$$^{13}$C$_{BF}$")

ax1.plot(mean_BSL_gradient_BF2 + (std_BSL_gradient_BF2),t_vals[w_d13C_BF],color='indigo',linestyle='--',linewidth=0.75)
ax1.plot(mean_BSL_gradient_BF2 - (std_BSL_gradient_BF2),t_vals[w_d13C_BF],color='indigo',linestyle='--',linewidth=0.75)
ax1.plot(mean_BSL_gradient_BF2 + (2. * std_BSL_gradient_BF2),t_vals[w_d13C_BF],color='indigo',linestyle=':',linewidth=0.5)
ax1.plot(mean_BSL_gradient_BF2 - (2. * std_BSL_gradient_BF2),t_vals[w_d13C_BF],color='indigo',linestyle=':',linewidth=0.5)


ax1.set_yticks(numpy.arange(-30,30,2.), minor=True)

ax1.set_ylim(-30,30)


ax1.set_title("NJ shelf forams")
ax1.set_xlabel("$\delta$$^{13}$C/Correlation unit$^2$")

ax1.grid(alpha=0.0,which='both')
ax1.legend()

ax1 = plt.subplot(3,5,15)


out_matrix_BSL = numpy.zeros((len(output_matrix_5[:,0]),len(w_TEX86)))
out_matrix2_BSL = numpy.zeros((len(output_matrix_5[:,0]),len(w_TEX86)))

for n in range(0,len(output_matrix_5[:,0])):
	out_matrix_BSL[n,:] = numpy.gradient(output_matrix_5[n,w_TEX86],t_vals[w_TEX86])
	out_matrix2_BSL[n,:] = numpy.gradient(out_matrix_BSL[n,:],t_vals[w_TEX86])
	
mean_BSL_gradient_tex = numpy.nanmean(out_matrix_BSL,axis=0)
std_BSL_gradient_tex = numpy.nanstd(out_matrix_BSL,axis=0)

mean_BSL_gradient_tex2 = numpy.nanmean(out_matrix2_BSL,axis=0)
std_BSL_gradient_tex2 = numpy.nanstd(out_matrix2_BSL,axis=0)

out_mat_fold = 1

ax1.plot(mean_BSL_gradient_tex2,t_vals[w_TEX86],color='mediumvioletred',linewidth=1.0)
ax1.plot([0,0],[0,0],color='mediumvioletred',linewidth=1.0,label="NJ $\Delta$$^2$TEX86")

ax1.plot(mean_BSL_gradient_tex2 + (std_BSL_gradient_tex2),t_vals[w_TEX86],color='mediumvioletred',linestyle='--',linewidth=0.75)
ax1.plot(mean_BSL_gradient_tex2 - (std_BSL_gradient_tex2),t_vals[w_TEX86],color='mediumvioletred',linestyle='--',linewidth=0.75)
ax1.plot(mean_BSL_gradient_tex2 + (2. * std_BSL_gradient_tex2),t_vals[w_TEX86],color='mediumvioletred',linestyle=':',linewidth=0.5)
ax1.plot(mean_BSL_gradient_tex2 - (2. * std_BSL_gradient_tex2),t_vals[w_TEX86],color='mediumvioletred',linestyle=':',linewidth=0.5)


ax1.set_yticks(numpy.arange(-30,30,2.), minor=True)

ax1.set_ylim(-30,30)

ax1.set_title("NJ shelf temperature")
ax1.set_xlabel("$\degree$C/Correlation unit$^2$")

ax1.grid(alpha=0.0,which='both')
ax1.legend(loc=2)

plt.tight_layout()

pltname = wkspc +  'NJ_d13C_derivative_20210604.png'
plt.savefig(pltname, dpi = 300)
pltname = wkspc +  'NJ_d13C_derivative_20230314.pdf'
plt.savefig(pltname, dpi = 300)
plt.close()
######

mean_global = numpy.nanmean(output_matrix[:,w_d13C_glob],axis=0)
std_global = numpy.nanstd(output_matrix[:,w_d13C_glob],axis=0)

mean_NJ_d13C_bulk = numpy.nanmean(output_matrix_3[:,w_d13C_bulk],axis=0)
std_NJ_d13C_bulk = numpy.nanstd(output_matrix_3[:,w_d13C_bulk],axis=0)

mean_NJ_d13C_org = numpy.nanmean(output_matrix_3[:,w_d13C_org],axis=0)
std_NJ_d13C_org = numpy.nanstd(output_matrix_3[:,w_d13C_org],axis=0)

mean_NJ_d13C_PF = numpy.nanmean(output_matrix_3[:,w_d13C_PF],axis=0)
std_NJ_d13C_PF = numpy.nanstd(output_matrix_3[:,w_d13C_PF],axis=0)

mean_NJ_d13C_BF = numpy.nanmean(output_matrix_3[:,w_d13C_BF],axis=0)
std_NJ_d13C_BF = numpy.nanstd(output_matrix_3[:,w_d13C_BF],axis=0)

mean_NJ_TEX86 = numpy.nanmean(output_matrix_3[:,w_TEX86],axis=0)
std_NJ_TEX86 = numpy.nanstd(output_matrix_3[:,w_TEX86],axis=0)


fig= plt.figure(4,figsize=(8.5,6.5))

def cross_spectrum(x1,x2,dt):
		
	x1 = numpy.flipud(x1) * 1.0
	x2 = numpy.flipud(x2) * 1.0
				
	N_orig = len(x1)
	
	x1_nomean = scipy.signal.detrend(x1,type='constant')
	x2_nomean = scipy.signal.detrend(x2,type='constant')
	
	a = (x1_nomean - numpy.mean(x1_nomean)) / (numpy.std(x1_nomean) * len(x1_nomean))
	v = (x2_nomean - numpy.mean(x2_nomean)) /  numpy.std(x2_nomean)

	C_12 = numpy.correlate (a,v,'full');

	zeros = numpy.zeros(len(x1_nomean))
		
	x1_nomean = numpy.append(x1_nomean,zeros)
	x2_nomean = numpy.append(x2_nomean,zeros)
		
	N = len(x1_nomean)
	freq =  numpy.fft.fftfreq(N) * (1./dt)
	x_1_f = fft(x1_nomean)
	x_2_f = fft(x2_nomean)
		
	S_12 = (1./(N*dt))*(numpy.conj(x_1_f)*x_2_f)
		
	return S_12, C_12, x_1_f, x_2_f, freq, N
	
iterations = len(output_matrix_3[:,0])

dt = per_year * 1.0

output_1 = numpy.zeros((iterations,int(len(t_vals[w_d13C_org])*2)-1))
output_2 = numpy.zeros((iterations,2))
C_12_new = numpy.zeros(int(len(t_vals[w_d13C_org])*2)-1)

int_list1 = numpy.arange(0,iterations,1)
int_list2 = numpy.arange(0,iterations,1)
numpy.random.shuffle(int_list1)
numpy.random.shuffle(int_list2)

ax1 = plt.subplot(221)

##

for datasets in range(0,iterations):

	x_1 =  output_matrix_5[int_list1[datasets],w_d13C_glob]
	x_2 =  output_matrix_5[int_list2[datasets],w_TEX86]

	S_12, C_12, x_1_f, x_2_f, freq, N = cross_spectrum(x_2*-1,x_1,dt)
	C_12_new = C_12*1.0
	index = numpy.argmax(C_12_new)
	
	t_vals2 = numpy.arange(-1*(int(N/2)-1),(int(N/2)),1)*dt
	
	output_1[datasets,:] = C_12_new
	output_2[datasets,0] = t_vals2[index]
	output_2[datasets,1] = C_12_new[index]
	

mean_1 = numpy.mean(output_1,axis=0)
std_1 = numpy.std(output_1,axis=0)

ax1.plot(t_vals2,mean_1,color='black',linestyle='-',label="$\delta$$^{13}$C$_{NJ}$")
ax1.plot(t_vals2,mean_1+std_1,color='black',linestyle='--')
ax1.plot(t_vals2,mean_1-std_1,color='black',linestyle='--')
output_2[:,1] = output_2[:,1]
mean_2 = numpy.mean(output_2,axis=0)
std_2 = numpy.std(output_2,axis=0)

w_max = numpy.argmax(mean_1)
max_val = mean_1[w_max]

ax1.plot(mean_2[0],mean_2[1],color='black',marker='o',markersize=5.)
ax1.plot([mean_2[0]-std_2[0],mean_2[0]+std_2[0]],[mean_2[1],mean_2[1]],color='black')
ax1.plot([mean_2[0]-(2.*std_2[0]),mean_2[0]+(2.*std_2[0])],[mean_2[1],mean_2[1]],linewidth=0.0,marker="|",color='black')																														

text_1 = str(round(t_vals2[w_max],3)) + " Myr."
print ("global, ", text_1, " lag")
ax1.set_xlabel("Lag relative to TEX86 ('Correlation units')")
ax1.set_ylabel("r")	
ax1.set_xticks(numpy.arange(-10,10,5.))
ax1.set_xticks(numpy.arange(-10,10,1.), minor=True)
ax1.set_xlim(-10.0,10.0)
ax1.set_ylim(0.8,1.0)

ax1.grid(alpha=0.0,which='both')
ax1.legend()
##
##
ax1 = plt.subplot(222)
for datasets in range(0,iterations):

	x_1 =  output_matrix_5[int_list1[datasets],w_d13C_bulk]
	x_2 =  output_matrix_5[int_list2[datasets],w_TEX86]

	S_12, C_12, x_1_f, x_2_f, freq, N = cross_spectrum(x_2*-1,x_1,dt)
	C_12_new = C_12*1.0
	index = numpy.argmax(C_12_new)
	
	t_vals2 = numpy.arange(-1*(int(N/2)-1),(int(N/2)),1)*dt
	
	output_1[datasets,:] = C_12_new
	output_2[datasets,0] = t_vals2[index]
	output_2[datasets,1] = C_12_new[index]


mean_1 = numpy.mean(output_1,axis=0)
std_1 = numpy.std(output_1,axis=0)

ax1.plot(t_vals2,mean_1,color='tomato',linestyle='-',label="$\delta$$^{13}$C$_{bulk}$")
ax1.plot(t_vals2,mean_1+std_1,color='tomato',linestyle='--')
ax1.plot(t_vals2,mean_1-std_1,color='tomato',linestyle='--')
output_2[:,1] = output_2[:,1]
mean_2 = numpy.mean(output_2,axis=0)
std_2 = numpy.std(output_2,axis=0)
ax1.plot(mean_2[0],mean_2[1],color='tomato',marker='o',markersize=5.)
ax1.plot([mean_2[0]-std_2[0],mean_2[0]+std_2[0]],[mean_2[1],mean_2[1]],color='tomato')
ax1.plot([mean_2[0]-(2.*std_2[0]),mean_2[0]+(2.*std_2[0])],[mean_2[1],mean_2[1]],linewidth=0.0,marker="|",color='tomato')


w_max = numpy.argmax(mean_1)
max_val = mean_1[w_max]


text_1 = str(round(t_vals2[w_max],3)) + " Myr."

print ("bulk, ", text_1, " lag")
ax1.set_xlabel("Lag relative to TEX86 ('Correlation units')")
ax1.set_ylabel("r")	
ax1.set_xticks(numpy.arange(-10,10,5.))
ax1.set_xticks(numpy.arange(-10,10,1.), minor=True)
ax1.set_xlim(-10.0,10.0)
ax1.set_ylim(0.8,1.0)

#ax1.grid(which='major')
#ax1.grid(alpha=0.1,linestyle=":",which='minor')
ax1.grid(alpha=0.0,which='both')
ax1.legend()
#ax1.text((t_vals2[w_max]),mean_1[w_max] + 0.025,text_1,color="tomato")	

ax1 = plt.subplot(223)

##
##

for datasets in range(0,iterations):

	x_1 =  output_matrix_5[int_list1[datasets],w_d13C_org]
	x_2 =  output_matrix_5[int_list2[datasets],w_TEX86]

	S_12, C_12, x_1_f, x_2_f, freq, N = cross_spectrum(x_2*-1,x_1,dt)
	C_12_new = C_12*1.0
	index = numpy.argmax(C_12_new)
	
	t_vals2 = numpy.arange(-1*(int(N/2)-1),(int(N/2)),1)*dt
	
	output_1[datasets,:] = C_12_new
	output_2[datasets,0] = t_vals2[index]
	output_2[datasets,1] = C_12_new[index]
	

mean_1 = numpy.mean(output_1,axis=0)
std_1 = numpy.std(output_1,axis=0)

ax1.plot(t_vals2,mean_1,color='forestgreen',linestyle='-',label="$\delta$$^{13}$C$_{org}$")
ax1.plot(t_vals2,mean_1+std_1,color='forestgreen',linestyle='--')
ax1.plot(t_vals2,mean_1-std_1,color='forestgreen',linestyle='--')
output_2[:,1] = output_2[:,1]
mean_2 = numpy.mean(output_2,axis=0)
std_2 = numpy.std(output_2,axis=0)

ax1.plot(mean_2[0],mean_2[1],color='forestgreen',marker='o',markersize=5.)
ax1.plot([mean_2[0]-std_2[0],mean_2[0]+std_2[0]],[mean_2[1],mean_2[1]],color='forestgreen')
ax1.plot([mean_2[0]-(2.*std_2[0]),mean_2[0]+(2.*std_2[0])],[mean_2[1],mean_2[1]],linewidth=0.0,marker="|",color='forestgreen')

w_max = numpy.argmax(mean_1)
max_val = mean_1[w_max]


text_1 = str(round(t_vals2[w_max],3)) + " Myr."
print ("org, ", text_1, " lag")

ax1.set_xlabel("Lag relative to TEX86 ('Correlation units')")
ax1.set_ylabel("r")	
ax1.set_xticks(numpy.arange(-10,10,5.))
ax1.set_xticks(numpy.arange(-10,10,1.), minor=True)

ax1.set_xlim(-10.0,10.0)
ax1.set_ylim(0.8,1.0)

ax1.grid(alpha=0.0,which='both')
ax1.legend()

##
##
ax1 = plt.subplot(224)
for datasets in range(0,iterations):

	x_1 =  output_matrix_5[int_list1[datasets],w_d13C_PF]
	x_2 =  output_matrix_5[int_list2[datasets],w_TEX86]

	S_12, C_12, x_1_f, x_2_f, freq, N = cross_spectrum(x_2*-1,x_1,dt)
	C_12_new = C_12*1.0
	index = numpy.argmax(C_12_new)
	
	t_vals2 = numpy.arange(-1*(int(N/2)-1),(int(N/2)),1)*dt
	
	output_1[datasets,:] = C_12_new
	output_2[datasets,0] = t_vals2[index]
	output_2[datasets,1] = C_12_new[index]

mean_1 = numpy.mean(output_1,axis=0)
std_1 = numpy.std(output_1,axis=0)

ax1.plot(t_vals2,mean_1,color='dodgerblue',linestyle='-',label="$\delta$$^{13}$C$_{PF}$")
ax1.plot(t_vals2,mean_1+std_1,color='dodgerblue',linestyle='--')
ax1.plot(t_vals2,mean_1-std_1,color='dodgerblue',linestyle='--')
output_2[:,1] = output_2[:,1]
mean_2 = numpy.mean(output_2,axis=0)
std_2 = numpy.std(output_2,axis=0)

ax1.plot(mean_2[0],mean_2[1],color='dodgerblue',marker='o',markersize=5.)
ax1.plot([mean_2[0]-std_2[0],mean_2[0]+std_2[0]],[mean_2[1],mean_2[1]],color='dodgerblue')
ax1.plot([mean_2[0]-(2.*std_2[0]),mean_2[0]+(2.*std_2[0])],[mean_2[1],mean_2[1]],linewidth=0.0,marker="|",color='dodgerblue')

w_max = numpy.argmax(mean_1)
max_val = mean_1[w_max]

text_1 = str(round(t_vals2[w_max],3)) + " Myr."
print ("PF, ", text_1, " lag")

for datasets in range(0,iterations):

	x_1 =  output_matrix_5[int_list1[datasets],w_d13C_BF]
	x_2 =  output_matrix_5[int_list2[datasets],w_TEX86]

	S_12, C_12, x_1_f, x_2_f, freq, N = cross_spectrum(x_2*-1,x_1,dt)
	C_12_new = C_12*1.0
	index = numpy.argmax(C_12_new)
	
	t_vals2 = numpy.arange(-1*(int(N/2)-1),(int(N/2)),1)*dt
	
	output_1[datasets,:] = C_12_new
	output_2[datasets,0] = t_vals2[index]
	output_2[datasets,1] = C_12_new[index]

mean_1 = numpy.mean(output_1,axis=0)
std_1 = numpy.std(output_1,axis=0)

ax1.plot(t_vals2,mean_1,color='indigo',linestyle='-',label="$\delta$$^{13}$C$_{BF}$")
ax1.plot(t_vals2,mean_1+std_1,color='indigo',linestyle='--')
ax1.plot(t_vals2,mean_1-std_1,color='indigo',linestyle='--')
output_2[:,1] = output_2[:,1]
mean_2 = numpy.mean(output_2,axis=0)
std_2 = numpy.std(output_2,axis=0)

ax1.plot(mean_2[0],mean_2[1],color='indigo',marker='o',markersize=5.)
ax1.plot([mean_2[0]-std_2[0],mean_2[0]+std_2[0]],[mean_2[1],mean_2[1]],color='indigo')
ax1.plot([mean_2[0]-(2.*std_2[0]),mean_2[0]+(2.*std_2[0])],[mean_2[1],mean_2[1]],linewidth=0.0,marker="|",color='indigo')

w_max = numpy.argmax(mean_1)


text_1 = str(round(t_vals2[w_max],3)) + " Myr."
print ("BF, ", text_1, " lag")

ax1.set_xlabel("Lag relative to TEX86 ('Correlation units')")
ax1.set_ylabel("r")	
ax1.set_xticks(numpy.arange(-10,10,5.))
ax1.set_xticks(numpy.arange(-10,10,1.), minor=True)
ax1.set_xlim(-10.0,10.0)
ax1.set_ylim(0.8,1.0)

ax1.grid(alpha=0.0,which='both')
ax1.legend()

plt.tight_layout()
	
pltname = wkspc +  'NJ_shelf_cross_corr_20210615.png'
plt.savefig(pltname, dpi = 300)
pltname = wkspc +  'NJ_shelf_cross_corr_20230314.pdf'
plt.savefig(pltname, dpi = 300)
plt.close()
######

fig= plt.figure(5,figsize=(20.,27.5))

ax1 = plt.subplot(551)


MC_iterations = 0
for n in range(0,MC_iterations):
	
	val2 = numpy.random.normal(loc = mean_global, scale = std_global)
	
	ax1.plot((val2),t_vals[w_d13C_glob],color='k',marker = 'x',markersize=2.0,linewidth=0.0,alpha=0.002)

out_mat_fold = 1

ax1.plot(mean_global,t_vals[w_d13C_glob],'k',linewidth=1.0)
line11, = ax1.plot([0,0],[0,0],'k',linewidth=1.0,label="$\delta$$^{13}$C$_{NJ}$")
ax1.plot(mean_global + (std_global),t_vals[w_d13C_glob],'k--',linewidth=0.75)
ax1.plot(mean_global - (std_global),t_vals[w_d13C_glob],'k--',linewidth=0.75)
ax1.plot(mean_global + (2. * std_global),t_vals[w_d13C_glob],'k:',linewidth=0.5)
ax1.plot(mean_global - (2. * std_global),t_vals[w_d13C_glob],'k:',linewidth=0.5)
ax1.set_ylim((-150),(150))
ax1.set_xlim(-5.,3.5)

line1, = ax1.plot([-10000,10000],[recovery_time,recovery_time],color='blue',linewidth=1.0,alpha=1.0,label="'recovery'")
ax1.plot([-10000,10000],[recovery_time+recovery_time_err,recovery_time+recovery_time_err],color='blue',linestyle="--",linewidth=1.0,alpha=1.0)
ax1.plot([-10000,10000],[recovery_time-recovery_time_err,recovery_time-recovery_time_err],color='blue',linestyle="--",linewidth=1.0,alpha=1.0)

line2, = ax1.plot([-10000,10000],[body_time,body_time],color='orange',linewidth=1.0,alpha=1.0,label="'core'")
ax1.plot([-10000,10000],[body_time+body_time_err,body_time+body_time_err],color='orange',linestyle="--",linewidth=1.0,alpha=1.0)
ax1.plot([-10000,10000],[body_time-body_time_err,body_time-body_time_err],color='orange',linestyle="--",linewidth=1.0,alpha=1.0)

line3, = ax1.plot([-10000,10000],[CIE_time_2,CIE_time_2],color='r',linewidth=1.5,alpha=1.0,label="'onset'")
ax1.plot([-10000,10000],[CIE_time_2+CIE_time_2_err,CIE_time_2+CIE_time_2_err],color='r',linestyle="--",linewidth=1.0,alpha=1.0)
ax1.plot([-10000,10000],[CIE_time_2-CIE_time_2_err,CIE_time_2-CIE_time_2_err],color='r',linestyle="--",linewidth=1.0,alpha=1.0)

line4, = ax1.plot([-10000,10000],[onset_time,onset_time],color='gray',linewidth=1.0,alpha=1.0,label="'precursor'")
ax1.plot([-10000,10000],[onset_time+onset_time_err,onset_time+onset_time_err],color='gray',linestyle="--",linewidth=1.0,alpha=1.0)
ax1.plot([-10000,10000],[onset_time-onset_time_err,onset_time-onset_time_err],color='gray',linestyle="--",linewidth=1.0,alpha=1.0)

ax1.grid(alpha=0.0,)
first_legend = ax1.legend(handles=[line1,line2,line3],loc=1)
ax1.add_artist(first_legend)
ax1.legend(handles=[line11],loc=3)

ax1.set_title("NJ shelf  $\delta$$^{13}$C")
ax1.set_xlabel("$\delta$$^{13}$C")
ax1.set_ylabel("Correlation units (rel. to CIE)")

ax1 = plt.subplot(552)


MC_iterations = 0
for n in range(0,MC_iterations):
	
	val2 = numpy.random.normal(loc = mean_NJ_d13C_bulk, scale = std_NJ_d13C_bulk)
	
	ax1.plot((val2),t_vals[w_d13C_glob],color='tomato',marker = 'x',markersize=2.0,linewidth=0.0,alpha=0.002)

out_mat_fold = 1

ax1.plot(SL_data_combined[w_d13C_bulk_c,1],mean_timing[w_d13C_bulk_c],color='tomato',marker='o',linewidth=0.0,markersize=2.0,alpha=0.15,label="NJ $\delta$$^{13}$C$_{bulk}$ data; $f(t)$ + $\Delta$$_{1}(t)$")

ax1.plot(mean_NJ_d13C_bulk,t_vals[w_d13C_glob],color='tomato',linewidth=1.0)
ax1.plot([0,0],[0,0],color='tomato',linewidth=1.0,label="$\delta$$^{13}$C$_{bulk}$")
ax1.plot(mean_NJ_d13C_bulk + (std_NJ_d13C_bulk),t_vals[w_d13C_glob],color='tomato',linestyle='--',linewidth=0.75)
ax1.plot(mean_NJ_d13C_bulk - (std_NJ_d13C_bulk),t_vals[w_d13C_glob],color='tomato',linestyle='--',linewidth=0.75)
ax1.plot(mean_NJ_d13C_bulk + (2. * std_NJ_d13C_bulk),t_vals[w_d13C_glob],color='tomato',linestyle=':',linewidth=0.5)
ax1.plot(mean_NJ_d13C_bulk - (2. * std_NJ_d13C_bulk),t_vals[w_d13C_glob],color='tomato',linestyle=':',linewidth=0.5)
ax1.set_ylim((-150),(150))
ax1.set_xlim(-5.,3.5)

ax1.plot([-10000,10000],[recovery_time,recovery_time],color='blue',linewidth=1.0,alpha=1.0)
ax1.plot([-10000,10000],[recovery_time+recovery_time_err,recovery_time+recovery_time_err],color='blue',linestyle="--",linewidth=1.0,alpha=1.0)
ax1.plot([-10000,10000],[recovery_time-recovery_time_err,recovery_time-recovery_time_err],color='blue',linestyle="--",linewidth=1.0,alpha=1.0)

ax1.plot([-10000,10000],[body_time,body_time],color='orange',linewidth=1.0,alpha=1.0)
ax1.plot([-10000,10000],[body_time+body_time_err,body_time+body_time_err],color='orange',linestyle="--",linewidth=1.0,alpha=1.0)
ax1.plot([-10000,10000],[body_time-body_time_err,body_time-body_time_err],color='orange',linestyle="--",linewidth=1.0,alpha=1.0)

ax1.plot([-10000,10000],[CIE_time_2,CIE_time_2],color='r',linewidth=1.5,alpha=1.0)
ax1.plot([-10000,10000],[CIE_time_2+CIE_time_2_err,CIE_time_2+CIE_time_2_err],color='r',linestyle="--",linewidth=1.0,alpha=1.0)
ax1.plot([-10000,10000],[CIE_time_2-CIE_time_2_err,CIE_time_2-CIE_time_2_err],color='r',linestyle="--",linewidth=1.0,alpha=1.0)

ax1.plot([-10000,10000],[onset_time,onset_time],color='gray',linewidth=1.0,alpha=1.0)
ax1.plot([-10000,10000],[onset_time+onset_time_err,onset_time+onset_time_err],color='gray',linestyle="--",linewidth=1.0,alpha=1.0)
ax1.plot([-10000,10000],[onset_time-onset_time_err,onset_time-onset_time_err],color='gray',linestyle="--",linewidth=1.0,alpha=1.0)

ax1.grid(alpha=0.0,)
ax1.legend()

ax1.set_title("NJ shelf $\delta$$^{13}$C$_{bulk}$")
ax1.set_xlabel("$\delta$$^{13}$C")

ax1 = plt.subplot(553)


MC_iterations = 0
for n in range(0,MC_iterations):
	
	val2 = numpy.random.normal(loc = mean_NJ_d13C_org, scale = std_NJ_d13C_org)
	
	ax1.plot((val2),t_vals[w_d13C_glob],color='forestgreen',marker = 'x',markersize=2.0,linewidth=0.0,alpha=0.002)

out_mat_fold = 1

ax1.plot(SL_data_combined[w_d13C_org_c,1],mean_timing[w_d13C_org_c],color='forestgreen',marker='o',linewidth=0.0,markersize=2.0,alpha=0.15,label="NJ $\delta$$^{13}$C$_{org}$ data")

ax1.plot(mean_NJ_d13C_org,t_vals[w_d13C_glob],color='forestgreen',linewidth=1.0)
ax1.plot([0,0],[0,0],color='forestgreen',linewidth=1.0,label="$\delta$$^{13}$C$_{org}$; $f(t)$ + $\Delta$$_{2}(t)$")
ax1.plot(mean_NJ_d13C_org + (std_NJ_d13C_org),t_vals[w_d13C_glob],color='forestgreen',linestyle='--',linewidth=0.75)
ax1.plot(mean_NJ_d13C_org - (std_NJ_d13C_org),t_vals[w_d13C_glob],color='forestgreen',linestyle='--',linewidth=0.75)
ax1.plot(mean_NJ_d13C_org + (2. * std_NJ_d13C_org),t_vals[w_d13C_glob],color='forestgreen',linestyle=':',linewidth=0.5)
ax1.plot(mean_NJ_d13C_org - (2. * std_NJ_d13C_org),t_vals[w_d13C_glob],color='forestgreen',linestyle=':',linewidth=0.5)
ax1.set_ylim((-150),(150))
ax1.set_xlim(-31.,-22.)

ax1.plot([-10000,10000],[recovery_time,recovery_time],color='blue',linewidth=1.0,alpha=1.0)
ax1.plot([-10000,10000],[recovery_time+recovery_time_err,recovery_time+recovery_time_err],color='blue',linestyle="--",linewidth=1.0,alpha=1.0)
ax1.plot([-10000,10000],[recovery_time-recovery_time_err,recovery_time-recovery_time_err],color='blue',linestyle="--",linewidth=1.0,alpha=1.0)

ax1.plot([-10000,10000],[body_time,body_time],color='orange',linewidth=1.0,alpha=1.0)
ax1.plot([-10000,10000],[body_time+body_time_err,body_time+body_time_err],color='orange',linestyle="--",linewidth=1.0,alpha=1.0)
ax1.plot([-10000,10000],[body_time-body_time_err,body_time-body_time_err],color='orange',linestyle="--",linewidth=1.0,alpha=1.0)

ax1.plot([-10000,10000],[CIE_time_2,CIE_time_2],color='r',linewidth=1.5,alpha=1.0)
ax1.plot([-10000,10000],[CIE_time_2+CIE_time_2_err,CIE_time_2+CIE_time_2_err],color='r',linestyle="--",linewidth=1.0,alpha=1.0)
ax1.plot([-10000,10000],[CIE_time_2-CIE_time_2_err,CIE_time_2-CIE_time_2_err],color='r',linestyle="--",linewidth=1.0,alpha=1.0)

ax1.plot([-10000,10000],[onset_time,onset_time],color='gray',linewidth=1.0,alpha=1.0)
ax1.plot([-10000,10000],[onset_time+onset_time_err,onset_time+onset_time_err],color='gray',linestyle="--",linewidth=1.0,alpha=1.0)
ax1.plot([-10000,10000],[onset_time-onset_time_err,onset_time-onset_time_err],color='gray',linestyle="--",linewidth=1.0,alpha=1.0)


ax1.set_title("NJ shelf $\delta$$^{13}$C$_{org}$")
ax1.set_xlabel("$\delta$$^{13}$C")

ax1.grid(alpha=0.0,)
ax1.legend()

ax1 = plt.subplot(554)


MC_iterations = 0
for n in range(0,MC_iterations):
	
	val2 = numpy.random.normal(loc = mean_NJ_d13C_PF, scale = std_NJ_d13C_PF)
	
	ax1.plot((val2),t_vals[w_d13C_glob],color='dodgerblue',marker = 'x',markersize=2.0,linewidth=0.0,alpha=0.002)

out_mat_fold = 1

ax1.plot(SL_data_combined[w_d13C_PF_c,1],mean_timing[w_d13C_PF_c],color='dodgerblue',marker='o',linewidth=0.0,markersize=2.0,alpha=0.15,label="NJ $\delta$$^{13}$C$_{PF}$ data")

ax1.plot(mean_NJ_d13C_PF,t_vals[w_d13C_glob],color='dodgerblue',linewidth=1.0)
ax1.plot([0,0],[0,0],color='dodgerblue',linewidth=1.0,label="$\delta$$^{13}$C$_{PF}$; $f(t)$ + $\Delta$$_{3}(t)$")
ax1.plot(mean_NJ_d13C_PF + (std_NJ_d13C_PF),t_vals[w_d13C_glob],color='dodgerblue',linestyle='--',linewidth=0.75)
ax1.plot(mean_NJ_d13C_PF - (std_NJ_d13C_PF),t_vals[w_d13C_glob],color='dodgerblue',linestyle='--',linewidth=0.75)
ax1.plot(mean_NJ_d13C_PF + (2. * std_NJ_d13C_PF),t_vals[w_d13C_glob],color='dodgerblue',linestyle=':',linewidth=0.5)
ax1.plot(mean_NJ_d13C_PF - (2. * std_NJ_d13C_PF),t_vals[w_d13C_glob],color='dodgerblue',linestyle=':',linewidth=0.5)

MC_iterations = 0
for n in range(0,MC_iterations):
	
	val2 = numpy.random.normal(loc = mean_NJ_d13C_BF, scale = std_NJ_d13C_BF)
	
	ax1.plot((val2),t_vals[w_d13C_glob],color='indigo',marker = 'x',markersize=2.0,linewidth=0.0,alpha=0.002)

out_mat_fold = 1

ax1.plot(SL_data_combined[w_d13C_BF_c,1],mean_timing[w_d13C_BF_c],color='indigo',marker='o',linewidth=0.0,markersize=2.0,alpha=0.15,label="NJ $\delta$$^{13}$C$_{BF}$ data")

ax1.plot(mean_NJ_d13C_BF,t_vals[w_d13C_glob],color='indigo',linewidth=1.0)
ax1.plot([0,0],[0,0],color='indigo',linewidth=1.0,label="$\delta$$^{13}$C$_{BF}$; $f(t)$ + $\Delta$$_{4}(t)$")
ax1.plot(mean_NJ_d13C_BF + (std_NJ_d13C_BF),t_vals[w_d13C_glob],color='indigo',linestyle='--',linewidth=0.75)
ax1.plot(mean_NJ_d13C_BF - (std_NJ_d13C_BF),t_vals[w_d13C_glob],color='indigo',linestyle='--',linewidth=0.75)
ax1.plot(mean_NJ_d13C_BF + (2. * std_NJ_d13C_BF),t_vals[w_d13C_glob],color='indigo',linestyle=':',linewidth=0.5)
ax1.plot(mean_NJ_d13C_BF - (2. * std_NJ_d13C_BF),t_vals[w_d13C_glob],color='indigo',linestyle=':',linewidth=0.5)
ax1.set_ylim((-150),(150))
ax1.set_xlim(-5.5,4.)

ax1.plot([-10000,10000],[recovery_time,recovery_time],color='blue',linewidth=1.0,alpha=1.0)
ax1.plot([-10000,10000],[recovery_time+recovery_time_err,recovery_time+recovery_time_err],color='blue',linestyle="--",linewidth=1.0,alpha=1.0)
ax1.plot([-10000,10000],[recovery_time-recovery_time_err,recovery_time-recovery_time_err],color='blue',linestyle="--",linewidth=1.0,alpha=1.0)

ax1.plot([-10000,10000],[body_time,body_time],color='orange',linewidth=1.0,alpha=1.0)
ax1.plot([-10000,10000],[body_time+body_time_err,body_time+body_time_err],color='orange',linestyle="--",linewidth=1.0,alpha=1.0)
ax1.plot([-10000,10000],[body_time-body_time_err,body_time-body_time_err],color='orange',linestyle="--",linewidth=1.0,alpha=1.0)

ax1.plot([-10000,10000],[CIE_time_2,CIE_time_2],color='r',linewidth=1.5,alpha=1.0)
ax1.plot([-10000,10000],[CIE_time_2+CIE_time_2_err,CIE_time_2+CIE_time_2_err],color='r',linestyle="--",linewidth=1.0,alpha=1.0)
ax1.plot([-10000,10000],[CIE_time_2-CIE_time_2_err,CIE_time_2-CIE_time_2_err],color='r',linestyle="--",linewidth=1.0,alpha=1.0)

ax1.plot([-10000,10000],[onset_time,onset_time],color='gray',linewidth=1.0,alpha=1.0)
ax1.plot([-10000,10000],[onset_time+onset_time_err,onset_time+onset_time_err],color='gray',linestyle="--",linewidth=1.0,alpha=1.0)
ax1.plot([-10000,10000],[onset_time-onset_time_err,onset_time-onset_time_err],color='gray',linestyle="--",linewidth=1.0,alpha=1.0)


ax1.set_title("NJ shelf forams")
ax1.set_xlabel("$\delta$$^{13}$C")

ax1.grid(alpha=0.0,)
ax1.legend()

ax1 = plt.subplot(555)


MC_iterations = 0
for n in range(0,MC_iterations):
	
	val2 = numpy.random.normal(loc = mean_NJ_TEX86, scale = std_NJ_TEX86)
	
	ax1.plot((val2),t_vals[w_d13C_glob],color='mediumvioletred',marker = 'x',markersize=2.0,linewidth=0.0,alpha=0.002)

out_mat_fold = 1

ax1.plot(SL_data_combined[w_TEX86_c,1],mean_timing[w_TEX86_c],color='mediumvioletred',marker='o',linewidth=0.0,markersize=2.0,alpha=0.15,label="TEX86 data")

ax1.plot(mean_NJ_TEX86,t_vals[w_d13C_glob],color='mediumvioletred',linewidth=1.0)
ax1.plot([0,0],[0,0],color='mediumvioletred',linewidth=1.0,label="TEX86; $n(t)$")
ax1.plot(mean_NJ_TEX86 + (std_NJ_TEX86),t_vals[w_d13C_glob],color='mediumvioletred',linestyle='--',linewidth=0.75)
ax1.plot(mean_NJ_TEX86 - (std_NJ_TEX86),t_vals[w_d13C_glob],color='mediumvioletred',linestyle='--',linewidth=0.75)
ax1.plot(mean_NJ_TEX86 + (2. * std_NJ_TEX86),t_vals[w_d13C_glob],color='mediumvioletred',linestyle=':',linewidth=0.5)
ax1.plot(mean_NJ_TEX86 - (2. * std_NJ_TEX86),t_vals[w_d13C_glob],color='mediumvioletred',linestyle=':',linewidth=0.5)
ax1.set_ylim((-150),(150))
ax1.set_xlim(26.75,37.51)

ax1.plot([-10000,10000],[recovery_time,recovery_time],color='blue',linewidth=1.0,alpha=1.0)
ax1.plot([-10000,10000],[recovery_time+recovery_time_err,recovery_time+recovery_time_err],color='blue',linestyle="--",linewidth=1.0,alpha=1.0)
ax1.plot([-10000,10000],[recovery_time-recovery_time_err,recovery_time-recovery_time_err],color='blue',linestyle="--",linewidth=1.0,alpha=1.0)

ax1.plot([-10000,10000],[body_time,body_time],color='orange',linewidth=1.0,alpha=1.0)
ax1.plot([-10000,10000],[body_time+body_time_err,body_time+body_time_err],color='orange',linestyle="--",linewidth=1.0,alpha=1.0)
ax1.plot([-10000,10000],[body_time-body_time_err,body_time-body_time_err],color='orange',linestyle="--",linewidth=1.0,alpha=1.0)

ax1.plot([-10000,10000],[CIE_time_2,CIE_time_2],color='r',linewidth=1.5,alpha=1.0)
ax1.plot([-10000,10000],[CIE_time_2+CIE_time_2_err,CIE_time_2+CIE_time_2_err],color='r',linestyle="--",linewidth=1.0,alpha=1.0)
ax1.plot([-10000,10000],[CIE_time_2-CIE_time_2_err,CIE_time_2-CIE_time_2_err],color='r',linestyle="--",linewidth=1.0,alpha=1.0)

ax1.plot([-10000,10000],[onset_time,onset_time],color='gray',linewidth=1.0,alpha=1.0)
ax1.plot([-10000,10000],[onset_time+onset_time_err,onset_time+onset_time_err],color='gray',linestyle="--",linewidth=1.0,alpha=1.0)
ax1.plot([-10000,10000],[onset_time-onset_time_err,onset_time-onset_time_err],color='gray',linestyle="--",linewidth=1.0,alpha=1.0)

ax1.set_title("NJ shelf temperature")
ax1.set_xlabel("Temperature ($\degree$C)")

ax1.grid(alpha=0.0,)
ax1.legend()

ax1 = plt.subplot(5,5,10)

min_time_BR = numpy.min(new_y_BR)
max_time_BR = numpy.max(new_y_BR)

w_time_BR = numpy.where((t_vals[w_d13C_bulk_BR]>=min_time_BR)&(t_vals[w_d13C_bulk_BR]<=max_time_BR))[0]
time_convert = t_vals[w_d13C_bulk_BR][w_time_BR]
interpolate_BR = scipy.interpolate.interp1d(new_y_BR,new_x_BR)
depth_BR = interpolate_BR(time_convert)

depth_min_BR = numpy.min(depth_BR)
depth_max_BR = numpy.max(depth_BR)


MC_iterations = 0
for n in range(0,MC_iterations):
	
	val2 = numpy.random.normal(loc = mean_d13C_bulk_BR[w_time_BR], scale = std_d13C_bulk_BR[w_time_BR])
	
	ax1.plot((val2),depth_BR,color='tomato',marker = 'x',markersize=2.0,linewidth=0.0,alpha=0.002)

out_mat_fold = 1

ax1.plot(SL_data_combined[w_d13C_bulk_BR_c,1],SL_data_combined[w_d13C_bulk_BR_c,0],color='tomato',marker='o',linewidth=0.0,markersize=4.0,alpha=0.5,label="BR $\delta$$^{13}$C$_{bulk}$ data")

ax1.plot(mean_d13C_bulk_BR[w_time_BR],depth_BR,color='tomato',linewidth=1.0)
ax1.plot([0,0],[0,0],color='tomato',linewidth=1.0,label="BR $\delta$$^{13}$C$_{bulk}$; $g_{1,5}(t)$")
ax1.plot(mean_d13C_bulk_BR[w_time_BR] + (std_d13C_bulk_BR[w_time_BR]),depth_BR,color='tomato',linestyle='--',linewidth=0.75)
ax1.plot(mean_d13C_bulk_BR[w_time_BR] - (std_d13C_bulk_BR[w_time_BR]),depth_BR,color='tomato',linestyle='--',linewidth=0.75)
ax1.plot(mean_d13C_bulk_BR[w_time_BR] + (2. * std_d13C_bulk_BR[w_time_BR]),depth_BR,color='tomato',linestyle=':',linewidth=0.5)
ax1.plot(mean_d13C_bulk_BR[w_time_BR] - (2. * std_d13C_bulk_BR[w_time_BR]),depth_BR,color='tomato',linestyle=':',linewidth=0.5)
ax1.set_ylim(low_BR,high_BR)
ax1.set_xlim(-5.,3.5)

ax1.plot([-10000,10000],[recovery_BR,recovery_BR],color='blue',linewidth=1.0,alpha=1.0)
ax1.plot([-10000,10000],[core_BR,core_BR],color='orange',linewidth=1.0,alpha=1.0)
ax1.plot([-10000,10000],[CIE_BR,CIE_BR],color='r',linewidth=1.5,alpha=1.0)
ax1.plot([-10000,10000],[base_BR,base_BR],color='blue',linewidth=1.0,alpha=1.0)

ax1.set_title("Bass River bulk")
ax1.set_xlabel("$\delta$$^{13}$C")


ax1.grid(alpha=0.0,)
ax1.legend()

ax1 = plt.subplot(5,5,9)


min_time_MI = numpy.min(new_y_MI)
max_time_MI = numpy.max(new_y_MI)

w_time_MI = numpy.where((t_vals[w_d13C_bulk_MI]>=min_time_MI)&(t_vals[w_d13C_bulk_MI]<=max_time_MI))[0]
time_convert = t_vals[w_d13C_bulk_MI][w_time_MI]
interpolate_MI = scipy.interpolate.interp1d(new_y_MI,new_x_MI)
depth_MI = interpolate_MI(time_convert)

depth_min_MI = numpy.min(depth_MI)
depth_max_MI = numpy.max(depth_MI)


MC_iterations = 0
for n in range(0,MC_iterations):
	
	val2 = numpy.random.normal(loc = mean_d13C_bulk_MI[w_time_MI], scale = std_d13C_bulk_MI[w_time_MI])
	
	ax1.plot((val2),depth_MI,color='tomato',marker = 'x',markersize=2.0,linewidth=0.0,alpha=0.002)

out_mat_fold = 1

ax1.plot(SL_data_combined[w_d13C_bulk_MI_c,1],SL_data_combined[w_d13C_bulk_MI_c,0],color='tomato',marker='o',linewidth=0.0,markersize=4.0,alpha=0.5,label="MI $\delta$$^{13}$C$_{bulk}$ data")

ax1.plot(mean_d13C_bulk_MI[w_time_MI],depth_MI,color='tomato',linewidth=1.0)
ax1.plot([0,0],[0,0],color='tomato',linewidth=1.0,label="MI $\delta$$^{13}$C$_{bulk}$; $g_{1,4}(t)$")
ax1.plot(mean_d13C_bulk_MI[w_time_MI] + (std_d13C_bulk_MI[w_time_MI]),depth_MI,color='tomato',linestyle='--',linewidth=0.75)
ax1.plot(mean_d13C_bulk_MI[w_time_MI] - (std_d13C_bulk_MI[w_time_MI]),depth_MI,color='tomato',linestyle='--',linewidth=0.75)
ax1.plot(mean_d13C_bulk_MI[w_time_MI] + (2. * std_d13C_bulk_MI[w_time_MI]),depth_MI,color='tomato',linestyle=':',linewidth=0.5)
ax1.plot(mean_d13C_bulk_MI[w_time_MI] - (2. * std_d13C_bulk_MI[w_time_MI]),depth_MI,color='tomato',linestyle=':',linewidth=0.5)
ax1.set_ylim(low_MI,high_MI)
ax1.set_xlim(-5.,3.5)

ax1.plot([-10000,10000],[recovery_MI,recovery_MI],color='blue',linewidth=1.0,alpha=1.0)
ax1.plot([-10000,10000],[core_MI,core_MI],color='orange',linewidth=1.0,alpha=1.0)
ax1.plot([-10000,10000],[CIE_MI,CIE_MI],color='r',linewidth=1.5,alpha=1.0)
ax1.plot([-10000,10000],[base_MI,base_MI],color='blue',linewidth=1.0,alpha=1.0)

ax1.set_title("Millville bulk")
ax1.set_xlabel("$\delta$$^{13}$C")

ax1.grid(alpha=0.0,)
ax1.legend()

ax1 = plt.subplot(5,5,8)


min_time_AN = numpy.min(new_y_AN)
max_time_AN = numpy.max(new_y_AN)

w_time_AN = numpy.where((t_vals[w_d13C_bulk_AN]>=min_time_AN)&(t_vals[w_d13C_bulk_AN]<=max_time_AN))[0]
time_convert = t_vals[w_d13C_bulk_AN][w_time_AN]
interpolate_AN = scipy.interpolate.interp1d(new_y_AN,new_x_AN)
depth_AN = interpolate_AN(time_convert)

depth_min_AN = numpy.min(depth_AN)
depth_max_AN = numpy.max(depth_AN)

MC_iterations = 0
for n in range(0,MC_iterations):
	
	val2 = numpy.random.normal(loc = mean_d13C_bulk_AN[w_time_AN], scale = std_d13C_bulk_AN[w_time_AN])
	
	ax1.plot((val2),depth_AN,color='tomato',marker = 'x',markersize=2.0,linewidth=0.0,alpha=0.002)

out_mat_fold = 1

ax1.plot(SL_data_combined[w_d13C_bulk_AN_c,1],SL_data_combined[w_d13C_bulk_AN_c,0],color='tomato',marker='o',linewidth=0.0,markersize=4.0,alpha=0.5,label="AN $\delta$$^{13}$C$_{bulk}$ data")

ax1.plot(mean_d13C_bulk_AN[w_time_AN],depth_AN,color='tomato',linewidth=1.0)
ax1.plot([0,0],[0,0],color='tomato',linewidth=1.0,label="AN $\delta$$^{13}$C$_{bulk}$; $g_{1,3}(t)$")
ax1.plot(mean_d13C_bulk_AN[w_time_AN] + (std_d13C_bulk_AN[w_time_AN]),depth_AN,color='tomato',linestyle='--',linewidth=0.75)
ax1.plot(mean_d13C_bulk_AN[w_time_AN] - (std_d13C_bulk_AN[w_time_AN]),depth_AN,color='tomato',linestyle='--',linewidth=0.75)
ax1.plot(mean_d13C_bulk_AN[w_time_AN] + (2. * std_d13C_bulk_AN[w_time_AN]),depth_AN,color='tomato',linestyle=':',linewidth=0.5)
ax1.plot(mean_d13C_bulk_AN[w_time_AN] - (2. * std_d13C_bulk_AN[w_time_AN]),depth_AN,color='tomato',linestyle=':',linewidth=0.5)
ax1.set_ylim(low_AN,high_AN)
ax1.set_xlim(-5.,3.5)

ax1.plot([-10000,10000],[recovery_AN,recovery_AN],color='blue',linewidth=1.0,alpha=1.0)
ax1.plot([-10000,10000],[core_AN,core_AN],color='orange',linewidth=1.0,alpha=1.0)
ax1.plot([-10000,10000],[CIE_AN,CIE_AN],color='r',linewidth=1.5,alpha=1.0)
ax1.plot([-10000,10000],[base_AN,base_AN],color='blue',linewidth=1.0,alpha=1.0)

ax1.set_title("Ancora bulk")
ax1.set_xlabel("$\delta$$^{13}$C")

ax1.grid(alpha=0.0,)
ax1.legend()

ax1 = plt.subplot(5,5,7)


min_time_WL = numpy.min(new_y_WL)
max_time_WL = numpy.max(new_y_WL)

w_time_WL = numpy.where((t_vals[w_d13C_bulk_WL]>=min_time_WL)&(t_vals[w_d13C_bulk_WL]<=max_time_WL))[0]
time_convert = t_vals[w_d13C_bulk_WL][w_time_WL]
interpolate_WL = scipy.interpolate.interp1d(new_y_WL,new_x_WL)
depth_WL = interpolate_WL(time_convert)

depth_min_WL = numpy.min(depth_WL)
depth_max_WL = numpy.max(depth_WL)

MC_iterations = 0
for n in range(0,MC_iterations):
	
	val2 = numpy.random.normal(loc = mean_d13C_bulk_WL[w_time_WL], scale = std_d13C_bulk_WL[w_time_WL])
	
	ax1.plot((val2),depth_WL,color='tomato',marker = 'x',markersize=2.0,linewidth=0.0,alpha=0.002)

out_mat_fold = 1

ax1.plot(SL_data_combined[w_d13C_bulk_WL_c,1],SL_data_combined[w_d13C_bulk_WL_c,0],color='tomato',marker='o',linewidth=0.0,markersize=4.0,alpha=0.5,label="WL $\delta$$^{13}$C$_{bulk}$ data")

ax1.plot(mean_d13C_bulk_WL[w_time_WL],depth_WL,color='tomato',linewidth=1.0)
ax1.plot([0,0],[0,0],color='tomato',linewidth=1.0,label="WL $\delta$$^{13}$C$_{bulk}$")
ax1.plot(mean_d13C_bulk_WL[w_time_WL] + (std_d13C_bulk_WL[w_time_WL]),depth_WL,color='tomato',linestyle='--',linewidth=0.75)
ax1.plot(mean_d13C_bulk_WL[w_time_WL] - (std_d13C_bulk_WL[w_time_WL]),depth_WL,color='tomato',linestyle='--',linewidth=0.75)
ax1.plot(mean_d13C_bulk_WL[w_time_WL] + (2. * std_d13C_bulk_WL[w_time_WL]),depth_WL,color='tomato',linestyle=':',linewidth=0.5)
ax1.plot(mean_d13C_bulk_WL[w_time_WL] - (2. * std_d13C_bulk_WL[w_time_WL]),depth_WL,color='tomato',linestyle=':',linewidth=0.5)
ax1.set_ylim(low_WL,high_WL)
ax1.set_xlim(-5.,3.5)

ax1.plot([-10000,10000],[recovery_WL,recovery_WL],color='blue',linewidth=1.0,alpha=1.0)
ax1.plot([-10000,10000],[core_WL,core_WL],color='orange',linewidth=1.0,alpha=1.0)
ax1.plot([-10000,10000],[CIE_WL,CIE_WL],color='r',linewidth=1.5,alpha=1.0)
ax1.plot([-10000,10000],[base_WL,base_WL],color='blue',linewidth=1.0,alpha=1.0)

ax1.set_title("Wilson Lake bulk")
ax1.set_xlabel("$\delta$$^{13}$C")

ax1.grid(alpha=0.0,)
ax1.legend()

ax1 = plt.subplot(5,5,6)


min_time_MAP3A = numpy.min(new_y_MAP3A)
max_time_MAP3A = numpy.max(new_y_MAP3A)

w_time_MAP3A = numpy.where((t_vals[w_d13C_bulk_MAP3A]>=min_time_MAP3A)&(t_vals[w_d13C_bulk_MAP3A]<=max_time_MAP3A))[0]
time_convert = t_vals[w_d13C_bulk_MAP3A][w_time_MAP3A]
interpolate_MAP3A = scipy.interpolate.interp1d(new_y_MAP3A,new_x_MAP3A)
depth_MAP3A = interpolate_MAP3A(time_convert)

depth_min_MAP3A = numpy.min(depth_MAP3A)
depth_max_MAP3A = numpy.max(depth_MAP3A)

MC_iterations = 0
for n in range(0,MC_iterations):
	
	val2 = numpy.random.normal(loc = mean_d13C_bulk_MAP3A[w_time_MAP3A], scale = std_d13C_bulk_MAP3A[w_time_MAP3A])
	
	ax1.plot((val2),depth_MAP3A,color='tomato',marker = 'x',markersize=2.0,linewidth=0.0,alpha=0.002)

out_mat_fold = 1

ax1.plot(SL_data_combined[w_d13C_bulk_MAP3A_c,1],SL_data_combined[w_d13C_bulk_MAP3A_c,0],color='tomato',marker='o',linewidth=0.0,markersize=4.0,alpha=0.5,label="MAP3A/B $\delta$$^{13}$C$_{bulk}$ data")

ax1.plot(mean_d13C_bulk_MAP3A[w_time_MAP3A],depth_MAP3A,color='tomato',linewidth=1.0)
ax1.plot([0,0],[0,0],color='tomato',linewidth=1.0,label="MAP3A/B $\delta$$^{13}$C$_{bulk}$; $g_{1,1}(t)$")
ax1.plot(mean_d13C_bulk_MAP3A[w_time_MAP3A] + (std_d13C_bulk_MAP3A[w_time_MAP3A]),depth_MAP3A,color='tomato',linestyle='--',linewidth=0.75)
ax1.plot(mean_d13C_bulk_MAP3A[w_time_MAP3A] - (std_d13C_bulk_MAP3A[w_time_MAP3A]),depth_MAP3A,color='tomato',linestyle='--',linewidth=0.75)
ax1.plot(mean_d13C_bulk_MAP3A[w_time_MAP3A] + (2. * std_d13C_bulk_MAP3A[w_time_MAP3A]),depth_MAP3A,color='tomato',linestyle=':',linewidth=0.5)
ax1.plot(mean_d13C_bulk_MAP3A[w_time_MAP3A] - (2. * std_d13C_bulk_MAP3A[w_time_MAP3A]),depth_MAP3A,color='tomato',linestyle=':',linewidth=0.5)
ax1.set_ylim(low_MAP3A,high_MAP3A)
ax1.set_xlim(-7.5,1.)

ax1.plot([-10000,10000],[recovery_MAP3A,recovery_MAP3A],color='blue',linewidth=1.0,alpha=1.0)
ax1.plot([-10000,10000],[core_MAP3A,core_MAP3A],color='orange',linewidth=1.0,alpha=1.0)
ax1.plot([-10000,10000],[CIE_MAP3A,CIE_MAP3A],color='r',linewidth=1.5,alpha=1.0)
ax1.plot([-10000,10000],[base_MAP3A,base_MAP3A],color='blue',linewidth=1.0,alpha=1.0)

ax1.set_title("MAP3A/B bulk")
ax1.set_xlabel("$\delta$$^{13}$C")
ax1.set_ylabel("Depth (ft.)")
ax1.grid(alpha=0.0,)
ax1.legend()

########################
########################
########################

ax1 = plt.subplot(5,5,15)


MC_iterations = 0
for n in range(0,MC_iterations):
	
	val2 = numpy.random.normal(loc = mean_d13C_org_BR[w_time_BR], scale = std_d13C_org_BR[w_time_BR])
	
	ax1.plot((val2),depth_BR,color='forestgreen',marker = 'x',markersize=2.0,linewidth=0.0,alpha=0.002)

out_mat_fold = 1

ax1.plot(SL_data_combined[w_d13C_org_BR_c,1],SL_data_combined[w_d13C_org_BR_c,0],color='forestgreen',marker='o',linewidth=0.0,markersize=4.0,alpha=0.5,label="BR $\delta$$^{13}$C$_{org}$ data")

ax1.plot(mean_d13C_org_BR[w_time_BR],depth_BR,color='forestgreen',linewidth=1.0)
ax1.plot([0,0],[0,0],color='forestgreen',linewidth=1.0,label="BR $\delta$$^{13}$C$_{org}$; $g_{2,5}(t)$")
ax1.plot(mean_d13C_org_BR[w_time_BR] + (std_d13C_org_BR[w_time_BR]),depth_BR,color='forestgreen',linestyle='--',linewidth=0.75)
ax1.plot(mean_d13C_org_BR[w_time_BR] - (std_d13C_org_BR[w_time_BR]),depth_BR,color='forestgreen',linestyle='--',linewidth=0.75)
ax1.plot(mean_d13C_org_BR[w_time_BR] + (2. * std_d13C_org_BR[w_time_BR]),depth_BR,color='forestgreen',linestyle=':',linewidth=0.5)
ax1.plot(mean_d13C_org_BR[w_time_BR] - (2. * std_d13C_org_BR[w_time_BR]),depth_BR,color='forestgreen',linestyle=':',linewidth=0.5)
ax1.set_ylim(low_BR,high_BR)
ax1.set_xlim(-31.,-22.)

ax1.plot([-10000,10000],[recovery_BR,recovery_BR],color='blue',linewidth=1.0,alpha=1.0)
ax1.plot([-10000,10000],[core_BR,core_BR],color='orange',linewidth=1.0,alpha=1.0)
ax1.plot([-10000,10000],[CIE_BR,CIE_BR],color='r',linewidth=1.5,alpha=1.0)
ax1.plot([-10000,10000],[base_BR,base_BR],color='blue',linewidth=1.0,alpha=1.0)

ax1.set_title("Bass River organic")
ax1.set_xlabel("$\delta$$^{13}$C")


ax1.grid(alpha=0.0,)
ax1.legend()

ax1 = plt.subplot(5,5,14)


MC_iterations = 0
for n in range(0,MC_iterations):
	
	val2 = numpy.random.normal(loc = mean_d13C_org_MI[w_time_MI], scale = std_d13C_org_MI[w_time_MI])
	
	ax1.plot((val2),depth_MI,color='forestgreen',marker = 'x',markersize=2.0,linewidth=0.0,alpha=0.002)

out_mat_fold = 1

ax1.plot(SL_data_combined[w_d13C_org_MI_c,1],SL_data_combined[w_d13C_org_MI_c,0],color='forestgreen',marker='o',linewidth=0.0,markersize=4.0,alpha=0.5,label="MI $\delta$$^{13}$C$_{org}$ data")

ax1.plot(mean_d13C_org_MI[w_time_MI],depth_MI,color='forestgreen',linewidth=1.0)
ax1.plot([0,0],[0,0],color='forestgreen',linewidth=1.0,label="MI $\delta$$^{13}$C$_{org}$; $g_{2,4}(t)$")
ax1.plot(mean_d13C_org_MI[w_time_MI] + (std_d13C_org_MI[w_time_MI]),depth_MI,color='forestgreen',linestyle='--',linewidth=0.75)
ax1.plot(mean_d13C_org_MI[w_time_MI] - (std_d13C_org_MI[w_time_MI]),depth_MI,color='forestgreen',linestyle='--',linewidth=0.75)
ax1.plot(mean_d13C_org_MI[w_time_MI] + (2. * std_d13C_org_MI[w_time_MI]),depth_MI,color='forestgreen',linestyle=':',linewidth=0.5)
ax1.plot(mean_d13C_org_MI[w_time_MI] - (2. * std_d13C_org_MI[w_time_MI]),depth_MI,color='forestgreen',linestyle=':',linewidth=0.5)
ax1.set_ylim(low_MI,high_MI)
ax1.set_xlim(-31.,-22.)

ax1.plot([-10000,10000],[recovery_MI,recovery_MI],color='blue',linewidth=1.0,alpha=1.0)
ax1.plot([-10000,10000],[core_MI,core_MI],color='orange',linewidth=1.0,alpha=1.0)
ax1.plot([-10000,10000],[CIE_MI,CIE_MI],color='r',linewidth=1.5,alpha=1.0)
ax1.plot([-10000,10000],[base_MI,base_MI],color='blue',linewidth=1.0,alpha=1.0)

ax1.set_title("Millville organic")
ax1.set_xlabel("$\delta$$^{13}$C")

ax1.grid(alpha=0.0,)
ax1.legend()

ax1 = plt.subplot(5,5,13)


MC_iterations = 0
for n in range(0,MC_iterations):
	
	val2 = numpy.random.normal(loc = mean_d13C_org_AN[w_time_AN], scale = std_d13C_org_AN[w_time_AN])
	
	ax1.plot((val2),depth_AN,color='forestgreen',marker = 'x',markersize=2.0,linewidth=0.0,alpha=0.002)

out_mat_fold = 1

ax1.plot(SL_data_combined[w_d13C_org_AN_c,1],SL_data_combined[w_d13C_org_AN_c,0],color='forestgreen',marker='o',linewidth=0.0,markersize=4.0,alpha=0.5,label="AN $\delta$$^{13}$C$_{org}$ data")

ax1.plot(mean_d13C_org_AN[w_time_AN],depth_AN,color='forestgreen',linewidth=1.0)
ax1.plot([0,0],[0,0],color='forestgreen',linewidth=1.0,label="AN $\delta$$^{13}$C$_{org}$; $g_{2,3}(t)$")
ax1.plot(mean_d13C_org_AN[w_time_AN] + (std_d13C_org_AN[w_time_AN]),depth_AN,color='forestgreen',linestyle='--',linewidth=0.75)
ax1.plot(mean_d13C_org_AN[w_time_AN] - (std_d13C_org_AN[w_time_AN]),depth_AN,color='forestgreen',linestyle='--',linewidth=0.75)
ax1.plot(mean_d13C_org_AN[w_time_AN] + (2. * std_d13C_org_AN[w_time_AN]),depth_AN,color='forestgreen',linestyle=':',linewidth=0.5)
ax1.plot(mean_d13C_org_AN[w_time_AN] - (2. * std_d13C_org_AN[w_time_AN]),depth_AN,color='forestgreen',linestyle=':',linewidth=0.5)
ax1.set_ylim(low_AN,high_AN)
ax1.set_xlim(-31.,-22.)

ax1.plot([-10000,10000],[recovery_AN,recovery_AN],color='blue',linewidth=1.0,alpha=1.0)
ax1.plot([-10000,10000],[core_AN,core_AN],color='orange',linewidth=1.0,alpha=1.0)
ax1.plot([-10000,10000],[CIE_AN,CIE_AN],color='r',linewidth=1.5,alpha=1.0)
ax1.plot([-10000,10000],[base_AN,base_AN],color='blue',linewidth=1.0,alpha=1.0)

ax1.set_title("Ancora organic")
ax1.set_xlabel("$\delta$$^{13}$C")

ax1.grid(alpha=0.0,)
ax1.legend()

ax1 = plt.subplot(5,5,12)


MC_iterations = 0
for n in range(0,MC_iterations):
	
	val2 = numpy.random.normal(loc = mean_d13C_org_WL[w_time_WL], scale = std_d13C_org_WL[w_time_WL])
	
	ax1.plot((val2),depth_WL,color='forestgreen',marker = 'x',markersize=2.0,linewidth=0.0,alpha=0.002)

out_mat_fold = 1

ax1.plot(SL_data_combined[w_d13C_org_WL_c,1],SL_data_combined[w_d13C_org_WL_c,0],color='forestgreen',marker='o',linewidth=0.0,markersize=4.0,alpha=0.5,label="WL $\delta$$^{13}$C$_{org}$ data")

ax1.plot(mean_d13C_org_WL[w_time_WL],depth_WL,color='forestgreen',linewidth=1.0)
ax1.plot([0,0],[0,0],color='forestgreen',linewidth=1.0,label="WL $\delta$$^{13}$C$_{org}$; $g_{2,2}(t)$")
ax1.plot(mean_d13C_org_WL[w_time_WL] + (std_d13C_org_WL[w_time_WL]),depth_WL,color='forestgreen',linestyle='--',linewidth=0.75)
ax1.plot(mean_d13C_org_WL[w_time_WL] - (std_d13C_org_WL[w_time_WL]),depth_WL,color='forestgreen',linestyle='--',linewidth=0.75)
ax1.plot(mean_d13C_org_WL[w_time_WL] + (2. * std_d13C_org_WL[w_time_WL]),depth_WL,color='forestgreen',linestyle=':',linewidth=0.5)
ax1.plot(mean_d13C_org_WL[w_time_WL] - (2. * std_d13C_org_WL[w_time_WL]),depth_WL,color='forestgreen',linestyle=':',linewidth=0.5)
ax1.set_ylim(low_WL,high_WL)
ax1.set_xlim(-31.,-22.)

ax1.plot([-10000,10000],[recovery_WL,recovery_WL],color='blue',linewidth=1.0,alpha=1.0)
ax1.plot([-10000,10000],[core_WL,core_WL],color='orange',linewidth=1.0,alpha=1.0)
ax1.plot([-10000,10000],[CIE_WL,CIE_WL],color='r',linewidth=1.5,alpha=1.0)
ax1.plot([-10000,10000],[base_WL,base_WL],color='blue',linewidth=1.0,alpha=1.0)

ax1.set_title("Wilson Lake organic")
ax1.set_xlabel("$\delta$$^{13}$C")

ax1.grid(alpha=0.0,)
ax1.legend()

ax1 = plt.subplot(5,5,11)


MC_iterations = 0
for n in range(0,MC_iterations):
	
	val2 = numpy.random.normal(loc = mean_d13C_org_MAP3A[w_time_MAP3A], scale = std_d13C_org_MAP3A[w_time_MAP3A])
	
	ax1.plot((val2),depth_MAP3A,color='forestgreen',marker = 'x',markersize=2.0,linewidth=0.0,alpha=0.002)

out_mat_fold = 1

ax1.plot(SL_data_combined[w_d13C_org_MAP3A_c,1],SL_data_combined[w_d13C_org_MAP3A_c,0],color='forestgreen',marker='o',linewidth=0.0,markersize=4.0,alpha=0.5,label="MAP3A/B $\delta$$^{13}$C$_{org}$ data")

ax1.plot(mean_d13C_org_MAP3A[w_time_MAP3A],depth_MAP3A,color='forestgreen',linewidth=1.0)
ax1.plot([0,0],[0,0],color='forestgreen',linewidth=1.0,label="MAP3A/B $\delta$$^{13}$C$_{org}$; $g_{2,1}(t)$")
ax1.plot(mean_d13C_org_MAP3A[w_time_MAP3A] + (std_d13C_org_MAP3A[w_time_MAP3A]),depth_MAP3A,color='forestgreen',linestyle='--',linewidth=0.75)
ax1.plot(mean_d13C_org_MAP3A[w_time_MAP3A] - (std_d13C_org_MAP3A[w_time_MAP3A]),depth_MAP3A,color='forestgreen',linestyle='--',linewidth=0.75)
ax1.plot(mean_d13C_org_MAP3A[w_time_MAP3A] + (2. * std_d13C_org_MAP3A[w_time_MAP3A]),depth_MAP3A,color='forestgreen',linestyle=':',linewidth=0.5)
ax1.plot(mean_d13C_org_MAP3A[w_time_MAP3A] - (2. * std_d13C_org_MAP3A[w_time_MAP3A]),depth_MAP3A,color='forestgreen',linestyle=':',linewidth=0.5)
ax1.set_ylim(low_MAP3A,high_MAP3A)
ax1.set_xlim(-31.,-22.)

ax1.plot([-10000,10000],[recovery_MAP3A,recovery_MAP3A],color='blue',linewidth=1.0,alpha=1.0)
ax1.plot([-10000,10000],[core_MAP3A,core_MAP3A],color='orange',linewidth=1.0,alpha=1.0)
ax1.plot([-10000,10000],[CIE_MAP3A,CIE_MAP3A],color='r',linewidth=1.5,alpha=1.0)
ax1.plot([-10000,10000],[base_MAP3A,base_MAP3A],color='blue',linewidth=1.0,alpha=1.0)

ax1.set_title("MAP3A/B organic")
ax1.set_xlabel("$\delta$$^{13}$C")
ax1.set_ylabel("Depth (ft.)")
ax1.grid(alpha=0.0,)
ax1.legend()

########################
########################
########################

ax1 = plt.subplot(5,5,20)


MC_iterations = 0
for n in range(0,MC_iterations):
	
	val2 = numpy.random.normal(loc = mean_d13C_PF_BR[w_time_BR], scale = std_d13C_PF_BR[w_time_BR])
	
	ax1.plot((val2),depth_BR,color='dodgerblue',marker = 'x',markersize=2.0,linewidth=0.0,alpha=0.002)

out_mat_fold = 1

ax1.plot(SL_data_combined[w_d13C_PF_BR_c,1],SL_data_combined[w_d13C_PF_BR_c,0],color='dodgerblue',marker='o',linewidth=0.0,markersize=4.0,alpha=0.5,label="BR $\delta$$^{13}$C$_{PF}$ data")

ax1.plot(mean_d13C_PF_BR[w_time_BR],depth_BR,color='dodgerblue',linewidth=1.0)
ax1.plot([0,0],[0,0],color='dodgerblue',linewidth=1.0,label="BR $\delta$$^{13}$C$_{PF}$; $g_{3,5}(t)$")
ax1.plot(mean_d13C_PF_BR[w_time_BR] + (std_d13C_PF_BR[w_time_BR]),depth_BR,color='dodgerblue',linestyle='--',linewidth=0.75)
ax1.plot(mean_d13C_PF_BR[w_time_BR] - (std_d13C_PF_BR[w_time_BR]),depth_BR,color='dodgerblue',linestyle='--',linewidth=0.75)
ax1.plot(mean_d13C_PF_BR[w_time_BR] + (2. * std_d13C_PF_BR[w_time_BR]),depth_BR,color='dodgerblue',linestyle=':',linewidth=0.5)
ax1.plot(mean_d13C_PF_BR[w_time_BR] - (2. * std_d13C_PF_BR[w_time_BR]),depth_BR,color='dodgerblue',linestyle=':',linewidth=0.5)

MC_iterations = 0
for n in range(0,MC_iterations):
	
	val2 = numpy.random.normal(loc = mean_d13C_BF_BR[w_time_BR], scale = std_d13C_BF_BR[w_time_BR])
	
	ax1.plot((val2),depth_BR,color='indigo',marker = 'x',markersize=2.0,linewidth=0.0,alpha=0.002)

out_mat_fold = 1

ax1.plot(SL_data_combined[w_d13C_BF_BR_c,1],SL_data_combined[w_d13C_BF_BR_c,0],color='indigo',marker='o',linewidth=0.0,markersize=4.0,alpha=0.5,label="BR $\delta$$^{13}$C$_{BF}$ data")

ax1.plot(mean_d13C_BF_BR[w_time_BR],depth_BR,color='indigo',linewidth=1.0)
ax1.plot([0,0],[0,0],color='indigo',linewidth=1.0,label="BR $\delta$$^{13}$C$_{BF}$; $g_{4,5}(t)$")
ax1.plot(mean_d13C_BF_BR[w_time_BR] + (std_d13C_BF_BR[w_time_BR]),depth_BR,color='indigo',linestyle='--',linewidth=0.75)
ax1.plot(mean_d13C_BF_BR[w_time_BR] - (std_d13C_BF_BR[w_time_BR]),depth_BR,color='indigo',linestyle='--',linewidth=0.75)
ax1.plot(mean_d13C_BF_BR[w_time_BR] + (2. * std_d13C_BF_BR[w_time_BR]),depth_BR,color='indigo',linestyle=':',linewidth=0.5)
ax1.plot(mean_d13C_BF_BR[w_time_BR] - (2. * std_d13C_BF_BR[w_time_BR]),depth_BR,color='indigo',linestyle=':',linewidth=0.5)

ax1.set_ylim(low_BR,high_BR)
ax1.set_xlim(-5.5,4.)

ax1.plot([-10000,10000],[recovery_BR,recovery_BR],color='blue',linewidth=1.0,alpha=1.0)
ax1.plot([-10000,10000],[core_BR,core_BR],color='orange',linewidth=1.0,alpha=1.0)
ax1.plot([-10000,10000],[CIE_BR,CIE_BR],color='r',linewidth=1.5,alpha=1.0)
ax1.plot([-10000,10000],[base_BR,base_BR],color='blue',linewidth=1.0,alpha=1.0)

ax1.set_title("Bass River forams")
ax1.set_xlabel("$\delta$$^{13}$C")

ax1.grid(alpha=0.0,)
ax1.legend()

ax1 = plt.subplot(5,5,19)


MC_iterations = 0
for n in range(0,MC_iterations):
	
	val2 = numpy.random.normal(loc = mean_d13C_PF_MI[w_time_MI], scale = std_d13C_PF_MI[w_time_MI])
	
	ax1.plot((val2),depth_MI,color='dodgerblue',marker = 'x',markersize=2.0,linewidth=0.0,alpha=0.002)

out_mat_fold = 1

ax1.plot(SL_data_combined[w_d13C_PF_MI_c,1],SL_data_combined[w_d13C_PF_MI_c,0],color='dodgerblue',marker='o',linewidth=0.0,markersize=4.0,alpha=0.5,label="MI $\delta$$^{13}$C$_{PF}$ data")

ax1.plot(mean_d13C_PF_MI[w_time_MI],depth_MI,color='dodgerblue',linewidth=1.0)
ax1.plot([0,0],[0,0],color='dodgerblue',linewidth=1.0,label="MI $\delta$$^{13}$C$_{PF}$; $g_{3,4}(t)$")
ax1.plot(mean_d13C_PF_MI[w_time_MI] + (std_d13C_PF_MI[w_time_MI]),depth_MI,color='dodgerblue',linestyle='--',linewidth=0.75)
ax1.plot(mean_d13C_PF_MI[w_time_MI] - (std_d13C_PF_MI[w_time_MI]),depth_MI,color='dodgerblue',linestyle='--',linewidth=0.75)
ax1.plot(mean_d13C_PF_MI[w_time_MI] + (2. * std_d13C_PF_MI[w_time_MI]),depth_MI,color='dodgerblue',linestyle=':',linewidth=0.5)
ax1.plot(mean_d13C_PF_MI[w_time_MI] - (2. * std_d13C_PF_MI[w_time_MI]),depth_MI,color='dodgerblue',linestyle=':',linewidth=0.5)

MC_iterations = 0
for n in range(0,MC_iterations):
	
	val2 = numpy.random.normal(loc = mean_d13C_BF_MI[w_time_MI], scale = std_d13C_BF_MI[w_time_MI])
	
	ax1.plot((val2),depth_MI,color='indigo',marker = 'x',markersize=2.0,linewidth=0.0,alpha=0.002)

out_mat_fold = 1

ax1.plot(SL_data_combined[w_d13C_BF_MI_c,1],SL_data_combined[w_d13C_BF_MI_c,0],color='indigo',marker='o',linewidth=0.0,markersize=4.0,alpha=0.5,label="MI $\delta$$^{13}$C$_{BF}$ data")

ax1.plot(mean_d13C_BF_MI[w_time_MI],depth_MI,color='indigo',linewidth=1.0)
ax1.plot([0,0],[0,0],color='indigo',linewidth=1.0,label="MI $\delta$$^{13}$C$_{BF}$; $g_{4,4}(t)$")
ax1.plot(mean_d13C_BF_MI[w_time_MI] + (std_d13C_BF_MI[w_time_MI]),depth_MI,color='indigo',linestyle='--',linewidth=0.75)
ax1.plot(mean_d13C_BF_MI[w_time_MI] - (std_d13C_BF_MI[w_time_MI]),depth_MI,color='indigo',linestyle='--',linewidth=0.75)
ax1.plot(mean_d13C_BF_MI[w_time_MI] + (2. * std_d13C_BF_MI[w_time_MI]),depth_MI,color='indigo',linestyle=':',linewidth=0.5)
ax1.plot(mean_d13C_BF_MI[w_time_MI] - (2. * std_d13C_BF_MI[w_time_MI]),depth_MI,color='indigo',linestyle=':',linewidth=0.5)

ax1.plot([-10000,10000],[recovery_MI,recovery_MI],color='blue',linewidth=1.0,alpha=1.0)
ax1.plot([-10000,10000],[core_MI,core_MI],color='orange',linewidth=1.0,alpha=1.0)
ax1.plot([-10000,10000],[CIE_MI,CIE_MI],color='r',linewidth=1.5,alpha=1.0)
ax1.plot([-10000,10000],[base_MI,base_MI],color='blue',linewidth=1.0,alpha=1.0)

ax1.set_ylim(low_MI,high_MI)
ax1.set_xlim(-5.5,4.)

ax1.set_title("Millville forams")
ax1.set_xlabel("$\delta$$^{13}$C")

ax1.grid(alpha=0.0,)
ax1.legend()

ax1 = plt.subplot(5,5,18)


MC_iterations = 0
for n in range(0,MC_iterations):
	
	val2 = numpy.random.normal(loc = mean_d13C_PF_AN[w_time_AN], scale = std_d13C_PF_AN[w_time_AN])
	
	ax1.plot((val2),depth_AN,color='dodgerblue',marker = 'x',markersize=2.0,linewidth=0.0,alpha=0.002)

out_mat_fold = 1

ax1.plot(SL_data_combined[w_d13C_PF_AN_c,1],SL_data_combined[w_d13C_PF_AN_c,0],color='dodgerblue',marker='o',linewidth=0.0,markersize=4.0,alpha=0.5,label="AN $\delta$$^{13}$C$_{PF}$ data")

ax1.plot(mean_d13C_PF_AN[w_time_AN],depth_AN,color='dodgerblue',linewidth=1.0)
ax1.plot([0,0],[0,0],color='dodgerblue',linewidth=1.0,label="AN $\delta$$^{13}$C$_{PF}$; $g_{3,3}(t)$")
ax1.plot(mean_d13C_PF_AN[w_time_AN] + (std_d13C_PF_AN[w_time_AN]),depth_AN,color='dodgerblue',linestyle='--',linewidth=0.75)
ax1.plot(mean_d13C_PF_AN[w_time_AN] - (std_d13C_PF_AN[w_time_AN]),depth_AN,color='dodgerblue',linestyle='--',linewidth=0.75)
ax1.plot(mean_d13C_PF_AN[w_time_AN] + (2. * std_d13C_PF_AN[w_time_AN]),depth_AN,color='dodgerblue',linestyle=':',linewidth=0.5)
ax1.plot(mean_d13C_PF_AN[w_time_AN] - (2. * std_d13C_PF_AN[w_time_AN]),depth_AN,color='dodgerblue',linestyle=':',linewidth=0.5)

MC_iterations = 0
for n in range(0,MC_iterations):
	
	val2 = numpy.random.normal(loc = mean_d13C_BF_AN[w_time_AN], scale = std_d13C_BF_AN[w_time_AN])
	
	ax1.plot((val2),depth_AN,color='indigo',marker = 'x',markersize=2.0,linewidth=0.0,alpha=0.002)

out_mat_fold = 1

ax1.plot(SL_data_combined[w_d13C_BF_AN_c,1],SL_data_combined[w_d13C_BF_AN_c,0],color='indigo',marker='o',linewidth=0.0,markersize=4.0,alpha=0.5,label="AN $\delta$$^{13}$C$_{BF}$ data")

ax1.plot(mean_d13C_BF_AN[w_time_AN],depth_AN,color='indigo',linewidth=1.0)
ax1.plot([0,0],[0,0],color='indigo',linewidth=1.0,label="AN $\delta$$^{13}$C$_{BF}$; $g_{4,3}(t)$")
ax1.plot(mean_d13C_BF_AN[w_time_AN] + (std_d13C_BF_AN[w_time_AN]),depth_AN,color='indigo',linestyle='--',linewidth=0.75)
ax1.plot(mean_d13C_BF_AN[w_time_AN] - (std_d13C_BF_AN[w_time_AN]),depth_AN,color='indigo',linestyle='--',linewidth=0.75)
ax1.plot(mean_d13C_BF_AN[w_time_AN] + (2. * std_d13C_BF_AN[w_time_AN]),depth_AN,color='indigo',linestyle=':',linewidth=0.5)
ax1.plot(mean_d13C_BF_AN[w_time_AN] - (2. * std_d13C_BF_AN[w_time_AN]),depth_AN,color='indigo',linestyle=':',linewidth=0.5)

ax1.set_title("Ancora forams")
ax1.set_xlabel("$\delta$$^{13}$C")

ax1.plot([-10000,10000],[recovery_AN,recovery_AN],color='blue',linewidth=1.0,alpha=1.0)
ax1.plot([-10000,10000],[core_AN,core_AN],color='orange',linewidth=1.0,alpha=1.0)
ax1.plot([-10000,10000],[CIE_AN,CIE_AN],color='r',linewidth=1.5,alpha=1.0)
ax1.plot([-10000,10000],[base_AN,base_AN],color='blue',linewidth=1.0,alpha=1.0)

ax1.set_ylim(low_AN,high_AN)
ax1.set_xlim(-5.5,4.)

ax1.grid(alpha=0.0,)
ax1.legend()

ax1 = plt.subplot(5,5,17)


MC_iterations = 0
for n in range(0,MC_iterations):
	
	val2 = numpy.random.normal(loc = mean_d13C_PF_WL[w_time_WL], scale = std_d13C_PF_WL[w_time_WL])
	
	ax1.plot((val2),depth_WL,color='dodgerblue',marker = 'x',markersize=2.0,linewidth=0.0,alpha=0.002)

out_mat_fold = 1

ax1.plot(SL_data_combined[w_d13C_PF_WL_c,1],SL_data_combined[w_d13C_PF_WL_c,0],color='dodgerblue',marker='o',linewidth=0.0,markersize=4.0,alpha=0.5,label="WL $\delta$$^{13}$C$_{PF}$ data")

ax1.plot(mean_d13C_PF_WL[w_time_WL],depth_WL,color='dodgerblue',linewidth=1.0)
ax1.plot([0,0],[0,0],color='dodgerblue',linewidth=1.0,label="WL $\delta$$^{13}$C$_{PF}$; $g_{3,2}(t)$")
ax1.plot(mean_d13C_PF_WL[w_time_WL] + (std_d13C_PF_WL[w_time_WL]),depth_WL,color='dodgerblue',linestyle='--',linewidth=0.75)
ax1.plot(mean_d13C_PF_WL[w_time_WL] - (std_d13C_PF_WL[w_time_WL]),depth_WL,color='dodgerblue',linestyle='--',linewidth=0.75)
ax1.plot(mean_d13C_PF_WL[w_time_WL] + (2. * std_d13C_PF_WL[w_time_WL]),depth_WL,color='dodgerblue',linestyle=':',linewidth=0.5)
ax1.plot(mean_d13C_PF_WL[w_time_WL] - (2. * std_d13C_PF_WL[w_time_WL]),depth_WL,color='dodgerblue',linestyle=':',linewidth=0.5)

MC_iterations = 0
for n in range(0,MC_iterations):
	
	val2 = numpy.random.normal(loc = mean_d13C_BF_WL[w_time_WL], scale = std_d13C_BF_WL[w_time_WL])
	
	ax1.plot((val2),depth_WL,color='indigo',marker = 'x',markersize=2.0,linewidth=0.0,alpha=0.002)

out_mat_fold = 1

ax1.plot(SL_data_combined[w_d13C_BF_WL_c,1],SL_data_combined[w_d13C_BF_WL_c,0],color='indigo',marker='o',linewidth=0.0,markersize=4.0,alpha=0.5,label="WL $\delta$$^{13}$C$_{BF}$ data")

ax1.plot(mean_d13C_BF_WL[w_time_WL],depth_WL,color='indigo',linewidth=1.0)
ax1.plot([0,0],[0,0],color='indigo',linewidth=1.0,label="WL $\delta$$^{13}$C$_{BF}$; $g_{4,2}(t)$")
ax1.plot(mean_d13C_BF_WL[w_time_WL] + (std_d13C_BF_WL[w_time_WL]),depth_WL,color='indigo',linestyle='--',linewidth=0.75)
ax1.plot(mean_d13C_BF_WL[w_time_WL] - (std_d13C_BF_WL[w_time_WL]),depth_WL,color='indigo',linestyle='--',linewidth=0.75)
ax1.plot(mean_d13C_BF_WL[w_time_WL] + (2. * std_d13C_BF_WL[w_time_WL]),depth_WL,color='indigo',linestyle=':',linewidth=0.5)
ax1.plot(mean_d13C_BF_WL[w_time_WL] - (2. * std_d13C_BF_WL[w_time_WL]),depth_WL,color='indigo',linestyle=':',linewidth=0.5)

ax1.set_ylim(low_WL,high_WL)
ax1.set_xlim(-5.5,4.)

ax1.plot([-10000,10000],[recovery_WL,recovery_WL],color='blue',linewidth=1.0,alpha=1.0)
ax1.plot([-10000,10000],[core_WL,core_WL],color='orange',linewidth=1.0,alpha=1.0)
ax1.plot([-10000,10000],[CIE_WL,CIE_WL],color='r',linewidth=1.5,alpha=1.0)
ax1.plot([-10000,10000],[base_WL,base_WL],color='blue',linewidth=1.0,alpha=1.0)

ax1.set_title("Wilson Lake forams")
ax1.set_xlabel("$\delta$$^{13}$C")

ax1.grid(alpha=0.0,)
ax1.legend()

ax1 = plt.subplot(5,5,16)


MC_iterations = 0
for n in range(0,MC_iterations):
	
	val2 = numpy.random.normal(loc = mean_d13C_PF_MAP3A[w_time_MAP3A], scale = std_d13C_PF_MAP3A[w_time_MAP3A])
	
	ax1.plot((val2),depth_MAP3A,color='dodgerblue',marker = 'x',markersize=2.0,linewidth=0.0,alpha=0.002)

out_mat_fold = 1

ax1.plot(SL_data_combined[w_d13C_PF_MAP3A_c,1],SL_data_combined[w_d13C_PF_MAP3A_c,0],color='dodgerblue',marker='o',linewidth=0.0,markersize=4.0,alpha=0.5,label="MAP3A/B $\delta$$^{13}$C$_{PF}$ data")

ax1.plot(mean_d13C_PF_MAP3A[w_time_MAP3A],depth_MAP3A,color='dodgerblue',linewidth=1.0)
ax1.plot([0,0],[0,0],color='dodgerblue',linewidth=1.0,label="MAP3A/B $\delta$$^{13}$C$_{PF}$; $g_{3,1}(t)$")
ax1.plot(mean_d13C_PF_MAP3A[w_time_MAP3A] + (std_d13C_PF_MAP3A[w_time_MAP3A]),depth_MAP3A,color='dodgerblue',linestyle='--',linewidth=0.75)
ax1.plot(mean_d13C_PF_MAP3A[w_time_MAP3A] - (std_d13C_PF_MAP3A[w_time_MAP3A]),depth_MAP3A,color='dodgerblue',linestyle='--',linewidth=0.75)
ax1.plot(mean_d13C_PF_MAP3A[w_time_MAP3A] + (2. * std_d13C_PF_MAP3A[w_time_MAP3A]),depth_MAP3A,color='dodgerblue',linestyle=':',linewidth=0.5)
ax1.plot(mean_d13C_PF_MAP3A[w_time_MAP3A] - (2. * std_d13C_PF_MAP3A[w_time_MAP3A]),depth_MAP3A,color='dodgerblue',linestyle=':',linewidth=0.5)

MC_iterations = 0
for n in range(0,MC_iterations):
	
	val2 = numpy.random.normal(loc = mean_d13C_BF_MAP3A[w_time_MAP3A], scale = std_d13C_BF_MAP3A[w_time_MAP3A])
	
	ax1.plot((val2),depth_MAP3A,color='indigo',marker = 'x',markersize=2.0,linewidth=0.0,alpha=0.002)

out_mat_fold = 1

ax1.plot(SL_data_combined[w_d13C_BF_MAP3A_c,1],SL_data_combined[w_d13C_BF_MAP3A_c,0],color='indigo',marker='o',linewidth=0.0,markersize=4.0,alpha=0.5,label="MAP3A/B $\delta$$^{13}$C$_{BF}$ data")

ax1.plot(mean_d13C_BF_MAP3A[w_time_MAP3A],depth_MAP3A,color='indigo',linewidth=1.0)
ax1.plot([0,0],[0,0],color='indigo',linewidth=1.0,label="MAP3A/B $\delta$$^{13}$C$_{BF}$; $g_{4,1}(t)$")
ax1.plot(mean_d13C_BF_MAP3A[w_time_MAP3A] + (std_d13C_BF_MAP3A[w_time_MAP3A]),depth_MAP3A,color='indigo',linestyle='--',linewidth=0.75)
ax1.plot(mean_d13C_BF_MAP3A[w_time_MAP3A] - (std_d13C_BF_MAP3A[w_time_MAP3A]),depth_MAP3A,color='indigo',linestyle='--',linewidth=0.75)
ax1.plot(mean_d13C_BF_MAP3A[w_time_MAP3A] + (2. * std_d13C_BF_MAP3A[w_time_MAP3A]),depth_MAP3A,color='indigo',linestyle=':',linewidth=0.5)
ax1.plot(mean_d13C_BF_MAP3A[w_time_MAP3A] - (2. * std_d13C_BF_MAP3A[w_time_MAP3A]),depth_MAP3A,color='indigo',linestyle=':',linewidth=0.5)

ax1.set_ylim(low_MAP3A,high_MAP3A)
ax1.set_xlim(-5.5,4.)

ax1.plot([-10000,10000],[recovery_MAP3A,recovery_MAP3A],color='blue',linewidth=1.0,alpha=1.0)
ax1.plot([-10000,10000],[core_MAP3A,core_MAP3A],color='orange',linewidth=1.0,alpha=1.0)
ax1.plot([-10000,10000],[CIE_MAP3A,CIE_MAP3A],color='r',linewidth=1.5,alpha=1.0)
ax1.plot([-10000,10000],[base_MAP3A,base_MAP3A],color='blue',linewidth=1.0,alpha=1.0)


ax1.set_title("MAP3A/B forams")
ax1.set_xlabel("$\delta$$^{13}$C")
ax1.set_ylabel("Depth (ft.)")

ax1.grid(alpha=0.0,)
ax1.legend()

########################
########################
########################

ax1 = plt.subplot(5,5,25)


MC_iterations = 0
for n in range(0,MC_iterations):
	
	val2 = numpy.random.normal(loc = mean_TEX86_BR[w_time_BR], scale = std_TEX86_BR[w_time_BR])
	
	ax1.plot((val2),depth_BR,color='mediumvioletred',marker = 'x',markersize=2.0,linewidth=0.0,alpha=0.002)

out_mat_fold = 1

ax1.plot(SL_data_combined[w_TEX86_BR_c,1],SL_data_combined[w_TEX86_BR_c,0],color='mediumvioletred',marker='o',linewidth=0.0,markersize=4.0,alpha=0.5,label="BR TEX86 data")

ax1.plot(mean_TEX86_BR[w_time_BR],depth_BR,color='mediumvioletred',linewidth=1.0)
ax1.plot([0,0],[0,0],color='mediumvioletred',linewidth=1.0,label="BR TEX86; $h_{5}(t)$")
ax1.plot(mean_TEX86_BR[w_time_BR] + (std_TEX86_BR[w_time_BR]),depth_BR,color='mediumvioletred',linestyle='--',linewidth=0.75)
ax1.plot(mean_TEX86_BR[w_time_BR] - (std_TEX86_BR[w_time_BR]),depth_BR,color='mediumvioletred',linestyle='--',linewidth=0.75)
ax1.plot(mean_TEX86_BR[w_time_BR] + (2. * std_TEX86_BR[w_time_BR]),depth_BR,color='mediumvioletred',linestyle=':',linewidth=0.5)
ax1.plot(mean_TEX86_BR[w_time_BR] - (2. * std_TEX86_BR[w_time_BR]),depth_BR,color='mediumvioletred',linestyle=':',linewidth=0.5)
ax1.set_ylim(low_BR,high_BR)
ax1.set_xlim(26.75,37.51)

ax1.plot([-10000,10000],[recovery_BR,recovery_BR],color='blue',linewidth=1.0,alpha=1.0)
ax1.plot([-10000,10000],[core_BR,core_BR],color='orange',linewidth=1.0,alpha=1.0)
ax1.plot([-10000,10000],[CIE_BR,CIE_BR],color='r',linewidth=1.5,alpha=1.0)
ax1.plot([-10000,10000],[base_BR,base_BR],color='blue',linewidth=1.0,alpha=1.0)

ax1.grid(alpha=0.0,)
ax1.legend()

ax1.set_title("Bass River temperature")
ax1.set_xlabel("Temperature ($\degree$C)")

ax1 = plt.subplot(5,5,24)


MC_iterations = 0
for n in range(0,MC_iterations):
	
	val2 = numpy.random.normal(loc = mean_TEX86_MI[w_time_MI], scale = std_TEX86_MI[w_time_MI])
	
	ax1.plot((val2),t_vals[w_TEX86_MI[w_time_MI]],color='mediumvioletred',marker = 'x',markersize=2.0,linewidth=0.0,alpha=0.002)

out_mat_fold = 1

ax1.plot(SL_data_combined[w_TEX86_MI_c,1],SL_data_combined[w_TEX86_MI_c,0],color='mediumvioletred',marker='o',linewidth=0.0,markersize=4.0,alpha=0.5,label="MI TEX86 data")

ax1.plot(mean_TEX86_MI[w_time_MI],depth_MI,color='mediumvioletred',linewidth=1.0)
ax1.plot([0,0],[0,0],color='mediumvioletred',linewidth=1.0,label="MI TEX86; $h_{4}(t)$")
ax1.plot(mean_TEX86_MI[w_time_MI] + (std_TEX86_MI[w_time_MI]),depth_MI,color='mediumvioletred',linestyle='--',linewidth=0.75)
ax1.plot(mean_TEX86_MI[w_time_MI] - (std_TEX86_MI[w_time_MI]),depth_MI,color='mediumvioletred',linestyle='--',linewidth=0.75)
ax1.plot(mean_TEX86_MI[w_time_MI] + (2. * std_TEX86_MI[w_time_MI]),depth_MI,color='mediumvioletred',linestyle=':',linewidth=0.5)
ax1.plot(mean_TEX86_MI[w_time_MI] - (2. * std_TEX86_MI[w_time_MI]),depth_MI,color='mediumvioletred',linestyle=':',linewidth=0.5)
ax1.set_ylim(low_MI,high_MI)
ax1.set_xlim(26.75,37.51)

ax1.plot([-10000,10000],[recovery_MI,recovery_MI],color='blue',linewidth=1.0,alpha=1.0)
ax1.plot([-10000,10000],[core_MI,core_MI],color='orange',linewidth=1.0,alpha=1.0)
ax1.plot([-10000,10000],[CIE_MI,CIE_MI],color='r',linewidth=1.5,alpha=1.0)
ax1.plot([-10000,10000],[base_MI,base_MI],color='blue',linewidth=1.0,alpha=1.0)

ax1.set_title("Millville temperature")
ax1.set_xlabel("Temperature ($\degree$C)")

ax1.grid(alpha=0.0,)
ax1.legend()

ax1 = plt.subplot(5,5,23)


MC_iterations = 0
for n in range(0,MC_iterations):
	
	val2 = numpy.random.normal(loc = mean_TEX86_AN[w_time_AN], scale = std_TEX86_AN[w_time_AN])
	
	ax1.plot((val2),depth_AN,color='mediumvioletred',marker = 'x',markersize=2.0,linewidth=0.0,alpha=0.002)

out_mat_fold = 1

ax1.plot(SL_data_combined[w_TEX86_AN_c,1],SL_data_combined[w_TEX86_AN_c,0],color='mediumvioletred',marker='o',linewidth=0.0,markersize=4.0,alpha=0.5,label="AN TEX86 data")

ax1.plot(mean_TEX86_AN[w_time_AN],depth_AN,color='mediumvioletred',linewidth=1.0)
ax1.plot([0,0],[0,0],color='mediumvioletred',linewidth=1.0,label="AN TEX86; $h_{3}(t)$")
ax1.plot(mean_TEX86_AN[w_time_AN] + (std_TEX86_AN[w_time_AN]),depth_AN,color='mediumvioletred',linestyle='--',linewidth=0.75)
ax1.plot(mean_TEX86_AN[w_time_AN] - (std_TEX86_AN[w_time_AN]),depth_AN,color='mediumvioletred',linestyle='--',linewidth=0.75)
ax1.plot(mean_TEX86_AN[w_time_AN] + (2. * std_TEX86_AN[w_time_AN]),depth_AN,color='mediumvioletred',linestyle=':',linewidth=0.5)
ax1.plot(mean_TEX86_AN[w_time_AN] - (2. * std_TEX86_AN[w_time_AN]),depth_AN,color='mediumvioletred',linestyle=':',linewidth=0.5)
ax1.set_ylim(low_AN,high_AN)
ax1.set_xlim(26.75,37.51)

ax1.plot([-10000,10000],[recovery_AN,recovery_AN],color='blue',linewidth=1.0,alpha=1.0)
ax1.plot([-10000,10000],[core_AN,core_AN],color='orange',linewidth=1.0,alpha=1.0)
ax1.plot([-10000,10000],[CIE_AN,CIE_AN],color='r',linewidth=1.5,alpha=1.0)
ax1.plot([-10000,10000],[base_AN,base_AN],color='blue',linewidth=1.0,alpha=1.0)


ax1.set_title("Ancora temperature")
ax1.set_xlabel("Temperature ($\degree$C)")

ax1.grid(alpha=0.0,)
ax1.legend()

ax1 = plt.subplot(5,5,22)


MC_iterations = 0
for n in range(0,MC_iterations):
	
	val2 = numpy.random.normal(loc = mean_TEX86_WL[w_time_WL], scale = std_TEX86_WL[w_time_WL])
	
	ax1.plot((val2),depth_WL,color='mediumvioletred',marker = 'x',markersize=2.0,linewidth=0.0,alpha=0.002)

out_mat_fold = 1

ax1.plot(SL_data_combined[w_TEX86_WL_c,1],SL_data_combined[w_TEX86_WL_c,0],color='mediumvioletred',marker='o',linewidth=0.0,markersize=4.0,alpha=0.5,label="WL TEX86 data")

ax1.plot(mean_TEX86_WL[w_time_WL],depth_WL,color='mediumvioletred',linewidth=1.0)
ax1.plot([0,0],[0,0],color='mediumvioletred',linewidth=1.0,label="WL TEX86; $h_{2}(t)$")
ax1.plot(mean_TEX86_WL[w_time_WL] + (std_TEX86_WL[w_time_WL]),depth_WL,color='mediumvioletred',linestyle='--',linewidth=0.75)
ax1.plot(mean_TEX86_WL[w_time_WL] - (std_TEX86_WL[w_time_WL]),depth_WL,color='mediumvioletred',linestyle='--',linewidth=0.75)
ax1.plot(mean_TEX86_WL[w_time_WL] + (2. * std_TEX86_WL[w_time_WL]),depth_WL,color='mediumvioletred',linestyle=':',linewidth=0.5)
ax1.plot(mean_TEX86_WL[w_time_WL] - (2. * std_TEX86_WL[w_time_WL]),depth_WL,color='mediumvioletred',linestyle=':',linewidth=0.5)
ax1.set_ylim(low_WL,high_WL)
ax1.set_xlim(26.75,37.51)

ax1.plot([-10000,10000],[recovery_WL,recovery_WL],color='blue',linewidth=1.0,alpha=1.0)
ax1.plot([-10000,10000],[core_WL,core_WL],color='orange',linewidth=1.0,alpha=1.0)
ax1.plot([-10000,10000],[CIE_WL,CIE_WL],color='r',linewidth=1.5,alpha=1.0)
ax1.plot([-10000,10000],[base_WL,base_WL],color='blue',linewidth=1.0,alpha=1.0)

ax1.set_title("Wilson Lake temperature")
ax1.set_xlabel("Temperature ($\degree$C)")

ax1.grid(alpha=0.0,)
ax1.legend()

ax1 = plt.subplot(5,5,21)


MC_iterations = 0
for n in range(0,MC_iterations):
	
	val2 = numpy.random.normal(loc = mean_TEX86_MAP3A[w_time_MAP3A], scale = std_TEX86_MAP3A[w_time_MAP3A])
	
	ax1.plot((val2),depth_MAP3A,color='mediumvioletred',marker = 'x',markersize=2.0,linewidth=0.0,alpha=0.002)

out_mat_fold = 1

ax1.plot(SL_data_combined[w_TEX86_MAP3A_c,1],SL_data_combined[w_TEX86_MAP3A_c,0],color='mediumvioletred',marker='o',linewidth=0.0,markersize=4.0,alpha=0.5,label="MAP3A/B TEX86 data")

ax1.plot(mean_TEX86_MAP3A[w_time_MAP3A],depth_MAP3A,color='mediumvioletred',linewidth=1.0)
ax1.plot([0,0],[0,0],color='mediumvioletred',linewidth=1.0,label="MAP3A/B TEX86; $h_{1}(t)$")
ax1.plot(mean_TEX86_MAP3A[w_time_MAP3A] + (std_TEX86_MAP3A[w_time_MAP3A]),depth_MAP3A,color='mediumvioletred',linestyle='--',linewidth=0.75)
ax1.plot(mean_TEX86_MAP3A[w_time_MAP3A] - (std_TEX86_MAP3A[w_time_MAP3A]),depth_MAP3A,color='mediumvioletred',linestyle='--',linewidth=0.75)
ax1.plot(mean_TEX86_MAP3A[w_time_MAP3A] + (2. * std_TEX86_MAP3A[w_time_MAP3A]),depth_MAP3A,color='mediumvioletred',linestyle=':',linewidth=0.5)
ax1.plot(mean_TEX86_MAP3A[w_time_MAP3A] - (2. * std_TEX86_MAP3A[w_time_MAP3A]),depth_MAP3A,color='mediumvioletred',linestyle=':',linewidth=0.5)
ax1.set_ylim(low_MAP3A,high_MAP3A)
ax1.set_xlim(26.75,37.51)

ax1.plot([-10000,10000],[recovery_MAP3A,recovery_MAP3A],color='blue',linewidth=1.0,alpha=1.0)
ax1.plot([-10000,10000],[core_MAP3A,core_MAP3A],color='orange',linewidth=1.0,alpha=1.0)
ax1.plot([-10000,10000],[CIE_MAP3A,CIE_MAP3A],color='r',linewidth=1.5,alpha=1.0)
ax1.plot([-10000,10000],[base_MAP3A,base_MAP3A],color='blue',linewidth=1.0,alpha=1.0)

ax1.set_title("MAP3A/B temperature")
ax1.set_xlabel("Temperature ($\degree$C)")
ax1.set_ylabel("Depth (ft.)")

ax1.grid(alpha=0.0,)
ax1.legend()

plt.tight_layout()

pltname = wkspc +  'NJ_d13C_depth_20210604.png'

plt.savefig(pltname, dpi = 300)

pltname = wkspc +  'NJ_d13C_depth_20230314.pdf'

plt.savefig(pltname, dpi = 300)

plt.close()


fig= plt.figure(6,figsize=(10.,5.))

def lead_lag_det(x1,y1,x2,y2):
	y1_last = numpy.nan
	y2_last = numpy.nan
	x1_interpolated = numpy.nan
	x2_interpolated = numpy.nan
	new_depth = numpy.arange(max(y1.min(), y2.min()), min(y1.max(), y2.max()), 1.)

	interp_x1 = scipy.interpolate.interp1d(y1, x1, kind='linear')
	interp_x2 = scipy.interpolate.interp1d(y2, x2, kind='linear')

	x1 = interp_x1(new_depth)
	x2 = interp_x2(new_depth)

	y1 = new_depth * 1.0
	y2 = new_depth * 1.0

	grad1 = numpy.gradient(x1,y1)
	grad2 = numpy.gradient(x2,y2)

	w1_grad = numpy.where(y1 < 2.)[0]
	w2_grad = numpy.where(y2 < 2.)[0]

	w1_grad2 = numpy.where(grad1[w1_grad] < 0.0)[0]
	w2_grad2 = numpy.where(grad2[w2_grad] < 0.0)[0]

	if len(w1_grad2)>0:
		y1_last1 = numpy.max(y1[w1_grad[w1_grad2]])
	if len(w2_grad2)>0:	
		y2_last1 = numpy.max(y2[w2_grad[w2_grad2]])


	if (len(w1_grad2)>0) &(len(w1_grad2)>0):
		
		w1_neg = numpy.where(y1[w1_grad] == y1_last1)[0]
		w2_neg = numpy.where(y2[w2_grad] == y2_last1)[0]
				
		w1_grad1 = numpy.where(y1 > y1_last1)[0]
		w2_grad1 = numpy.where(y2 > y2_last1)[0]		

		w1_grad2 = numpy.where(grad1[w1_grad1] > 0.0)[0]
		w2_grad2 = numpy.where(grad2[w2_grad1] > 0.0)[0]
		
		
		if len(w1_grad2)>0:
			y1_last2 = numpy.min(y1[w1_grad1[w1_grad2]])
		if len(w2_grad2)>0:	
			y2_last2 = numpy.min(y2[w2_grad1[w2_grad2]])
		if (len(w1_grad2)>0) &(len(w1_grad2)>0):
			w1_pos = numpy.where(y1[w1_grad1] == y1_last2)[0]
			w2_pos = numpy.where(y2[w2_grad1] == y2_last2)[0]
			
			x_arr = numpy.ndarray.flatten(numpy.array([grad1[w1_grad][w1_neg],grad1[w1_grad1][w1_pos]]))
			y_arr = numpy.ndarray.flatten(numpy.array([y1[w1_grad][w1_neg],y1[w1_grad1][w1_pos]]))

			x1_interpolated = numpy.interp(0.0, x_arr, y_arr)

			x_arr = numpy.ndarray.flatten(numpy.array([grad2[w2_grad][w2_neg],grad2[w2_grad1][w2_pos]]))
			y_arr = numpy.ndarray.flatten(numpy.array([y2[w2_grad][w2_neg],y2[w2_grad1][w2_pos]])			)

			x2_interpolated = numpy.interp(0.0, x_arr, y_arr)

	dif = x1_interpolated-x2_interpolated

	if (y1_last < -10) or (y2_last<-10):
		y1_last = numpy.nan
		y2_last = numpy.nan
		dif = numpy.nan

	return dif, x1_interpolated, x2_interpolated

iterations = len(output_matrix_5[:,0])

dt = per_year * 1.0 

output_1 = numpy.zeros((iterations,3))

int_list1 = numpy.arange(0,iterations,1)
int_list2 = numpy.arange(0,iterations,1)
numpy.random.shuffle(int_list1)
numpy.random.shuffle(int_list2)
##
##
ax1 = plt.subplot(241)

for datasets in range(0,iterations):

	x_1 =  output_matrix_5[int_list1[datasets],w_d13C_glob]*-1.
	x_2 =  output_matrix_5[int_list2[datasets],w_TEX86]

	output_1[datasets,:] = lead_lag_det(x_1,t_vals[w_d13C_glob],x_2,t_vals[w_TEX86])
new_x = numpy.arange(-10.,10.5,0.25)
num_bins = 21
w1 = numpy.where(numpy.isnan(output_1[:,0])==False)[0]

n, bins, patches = ax1.hist(output_1[w1,0], range=(-10.5,10.5),bins=num_bins, density=True,color='k',alpha=0.5,histtype='stepfilled')
mu = numpy.nanmean(output_1[w1,0])
sigma = numpy.nanstd(output_1[w1,0])

y_bulk = ((1 / (numpy.sqrt(2 * numpy.pi) * sigma)) *
     numpy.exp(-0.5 * (1 / sigma * (new_x - mu))**2))
hist, bin_edges = numpy.histogram(output_1[w1,0],bins=num_bins, range=(-10.5,10.5), density=True)
bins_glob =  new_x * 1.0
cdf_glob = numpy.cumsum(y_bulk*(new_x[1]-new_x[0]))
ax1.plot(new_x, y_bulk,color='k',linestyle='-',linewidth=2.,label="$\delta$$^{13}$C")
ax1.set_ylabel('Probability density')

ax1.set_xticks(numpy.arange(-10,10,5.))
ax1.set_xticks(numpy.arange(-10,10,1.), minor=True)
ax1.set_xlim(-10.,10.)
ax1.set_ylim(0.0,0.25)

ax1.set_xticklabels([])

ax1.grid(alpha=0.0,which='both')
ax1.legend(fontsize=7.)

output_1 = numpy.zeros((iterations,3))

numpy.random.shuffle(int_list1)
numpy.random.shuffle(int_list2)
##
##
ax1 = plt.subplot(242)

for datasets in range(0,iterations):

	x_1 =  output_matrix_5[int_list1[datasets],w_d13C_bulk]*-1.
	x_2 =  output_matrix_5[int_list2[datasets],w_TEX86]

	output_1[datasets,:] = lead_lag_det(x_1,t_vals[w_d13C_bulk],x_2,t_vals[w_TEX86])
new_x = numpy.arange(-10.,10.5,0.25)

w1 = numpy.where(numpy.isnan(output_1[:,0])==False)[0]

n, bins, patches = ax1.hist(output_1[w1,0], range=(-10.5,10.5),bins=num_bins, density=True,color='tomato',alpha=0.5,histtype='stepfilled')
mu = numpy.nanmean(output_1[w1,0])
sigma = numpy.nanstd(output_1[w1,0])

y_bulk = ((1 / (numpy.sqrt(2 * numpy.pi) * sigma)) *
     numpy.exp(-0.5 * (1 / sigma * (new_x - mu))**2))
hist, bin_edges = numpy.histogram(output_1[w1,0],bins=num_bins, range=(-10.5,10.5), density=True)
bins_bulk =  new_x * 1.0
cdf_bulk = numpy.cumsum(y_bulk*(new_x[1]-new_x[0]))
ax1.plot(new_x, y_bulk,color='tomato',linestyle='-',linewidth=2.,label="$\delta$$^{13}$C$_{bulk}$")

ax1.set_xticks(numpy.arange(-10,10,5.))
ax1.set_xticks(numpy.arange(-10,10,1.), minor=True)
ax1.set_yticklabels([])
ax1.set_xlim(-10.,10.)
ax1.set_ylim(0.0,0.25)

ax1.set_xticklabels([])

ax1.grid(alpha=0.0,which='both')
ax1.legend(fontsize=7.)

ax1 = plt.subplot(243)
numpy.random.shuffle(int_list1)
numpy.random.shuffle(int_list2)
for datasets in range(0,iterations):

	x_1 =  output_matrix_5[int_list1[datasets],w_d13C_org]*-1.
	x_2 =  output_matrix_5[int_list2[datasets],w_TEX86]

	output_1[datasets,:] = lead_lag_det(x_1,t_vals[w_d13C_org],x_2,t_vals[w_TEX86])
w1 = numpy.where(numpy.isnan(output_1[:,0])==False)[0]
n, bins, patches = ax1.hist(output_1[w1,0], range=(-10.5,10.5), bins=num_bins, density=True,color="forestgreen",alpha=0.5,histtype='stepfilled')
mu = numpy.nanmean(output_1[w1,0])
sigma = numpy.nanstd(output_1[w1,0])
y_org = ((1 / (numpy.sqrt(2 * numpy.pi) * sigma)) *
     numpy.exp(-0.5 * (1 / sigma * (new_x - mu))**2))
	 
bins_org = new_x * 1.0
cdf_org = numpy.cumsum(y_org*(new_x[1]-new_x[0]))

ax1.plot(new_x, y_org,color="forestgreen",linestyle='-',linewidth=2.,label="$\delta$$^{13}$C$_{org}$")
ax1.set_xticks(numpy.arange(-10,10,5.))
ax1.set_xticks(numpy.arange(-10,10,1.), minor=True)
ax1.set_xlim(-10.,10.)
ax1.set_ylim(0.0,0.25)
ax1.set_yticklabels([])
ax1.set_xticklabels([])
ax1.grid(alpha=0.0,which='both')
ax1.legend(fontsize=7.)

##
##
ax1 = plt.subplot(244)
numpy.random.shuffle(int_list1)
numpy.random.shuffle(int_list2)
for datasets in range(0,iterations):

	x_1 =  output_matrix_5[int_list1[datasets],w_d13C_PF]*-1.
	x_2 =  output_matrix_5[int_list2[datasets],w_TEX86]

	output_1[datasets,:] = lead_lag_det(x_1,t_vals[w_d13C_PF],x_2,t_vals[w_TEX86])
w1 = numpy.where(numpy.isnan(output_1[:,0])==False)[0]
n, bins, patches = ax1.hist(output_1[w1,0], range=(-10.5,10.5), bins=num_bins, density=True, color="dodgerblue",alpha=0.5,histtype='stepfilled')
mu = numpy.nanmean(output_1[w1,0])
sigma = numpy.nanstd(output_1[w1,0])
y = ((1 / (numpy.sqrt(2 * numpy.pi) * sigma)) *
     numpy.exp(-0.5 * (1 / sigma * (new_x - mu))**2))

bins_BF = new_x * 1.0
cdf_BF = numpy.cumsum(y*(new_x[1]-new_x[0]))

ax1.plot(new_x, y, color="dodgerblue",linestyle='-',linewidth=2.,label="$\delta$$^{13}$C$_{BF}$")
numpy.random.shuffle(int_list1)
numpy.random.shuffle(int_list2)
for datasets in range(0,iterations):

	x_1 =  output_matrix_5[int_list1[datasets],w_d13C_BF]*-1.
	x_2 =  output_matrix_5[int_list2[datasets],w_TEX86]

	output_1[datasets,:] = lead_lag_det(x_1,t_vals[w_d13C_BF],x_2,t_vals[w_TEX86])
w1 = numpy.where(numpy.isnan(output_1[:,0])==False)[0]
n, bins, patches = ax1.hist(output_1[w1,0], range=(-10.5,10.5), bins=num_bins, density=True, color="indigo",alpha=0.5,histtype='stepfilled')
mu = numpy.nanmean(output_1[w1,0])
sigma = numpy.nanstd(output_1[w1,0])
y = ((1 / (numpy.sqrt(2 * numpy.pi) * sigma)) *
     numpy.exp(-0.5 * (1 / sigma * (new_x - mu))**2))

bins_PF = new_x * 1.0
cdf_PF = numpy.cumsum(y*(new_x[1]-new_x[0]))
	 	 
ax1.plot(new_x, y, color="indigo",linestyle='-',linewidth=2.,label="$\delta$$^{13}$C$_{PF}$")

ax1.set_xticks(numpy.arange(-10,10,5.))
ax1.set_xticks(numpy.arange(-10,10,1.), minor=True)
ax1.set_xlim(-10.,10.)
ax1.set_ylim(0.0,0.25)
ax1.set_yticklabels([])

ax1.set_xticklabels([])
ax1.grid(alpha=0.0,which='both')
ax1.legend(fontsize=7.)

##

numpy.random.shuffle(int_list1)
numpy.random.shuffle(int_list2)

ax1 = plt.subplot(245)
numpy.random.shuffle(int_list1)
numpy.random.shuffle(int_list2)
for datasets in range(0,iterations):

	x_1 =  output_matrix_5[int_list1[datasets],w_d13C_glob]*-1.
	x_2 =  output_matrix_5[int_list2[datasets],w_TEX86]

	output_1[datasets,:] = lead_lag_det(x_1,t_vals[w_d13C_glob],x_2,t_vals[w_TEX86])
w1 = numpy.where(numpy.isnan(output_1[:,0])==False)[0]
n, bins, patches = ax1.hist(output_1[w1,0], range=(-10.5,10.5), bins=num_bins, cumulative=True, density=True,color='k',alpha=0.5,histtype='stepfilled')
hist, bin_edges = numpy.histogram(output_1[w1,0], range=(-10.5,10.5),bins=num_bins, density=True)
cdf = numpy.cumsum(hist*numpy.diff(bin_edges))
x_bins = (bin_edges[1:] + bin_edges[0:-1])/2.

ax1.plot(bins_glob, cdf_glob,color='k', linestyle='-',linewidth=2.0,label="$\delta$$^{13}$C")

ax1.set_ylabel('Cumulative density')

ax1.set_xticks(numpy.arange(-10,10,5.))
ax1.set_xticks(numpy.arange(-10,10,1.), minor=True)
ax1.set_yticks(numpy.arange(0.,1.,0.1), minor=True)
ax1.set_xlim(-10.,10.)
ax1.set_ylim(0.0,1.0)

ax1.grid(alpha=0.0,which='both')

ax1 = plt.subplot(246)
numpy.random.shuffle(int_list1)
numpy.random.shuffle(int_list2)
for datasets in range(0,iterations):

	x_1 =  output_matrix_5[int_list1[datasets],w_d13C_bulk]*-1.
	x_2 =  output_matrix_5[int_list2[datasets],w_TEX86]

	output_1[datasets,:] = lead_lag_det(x_1,t_vals[w_d13C_bulk],x_2,t_vals[w_TEX86])
w1 = numpy.where(numpy.isnan(output_1[:,0])==False)[0]
n, bins, patches = ax1.hist(output_1[w1,0], range=(-10.5,10.5), bins=num_bins, cumulative=True, density=True,color='tomato',alpha=0.5,histtype='stepfilled')
hist, bin_edges = numpy.histogram(output_1[w1,0], range=(-10.5,10.5),bins=num_bins, density=True)
cdf = numpy.cumsum(hist*numpy.diff(bin_edges))
x_bins = (bin_edges[1:] + bin_edges[0:-1])/2.

ax1.plot(bins_bulk, cdf_bulk,color='tomato', linestyle='-',linewidth=2.0,label="$\delta$$^{13}$C$_{bulk}$")

ax1.set_xticks(numpy.arange(-10,10,5.))
ax1.set_xticks(numpy.arange(-10,10,1.), minor=True)
ax1.set_yticklabels([])
ax1.set_xlim(-10.,10.)
ax1.set_ylim(0.0,1.0)


ax1.grid(alpha=0.0,which='both')

ax1 = plt.subplot(247)
numpy.random.shuffle(int_list1)
numpy.random.shuffle(int_list2)
for datasets in range(0,iterations):

	x_1 =  output_matrix_5[int_list1[datasets],w_d13C_org]*-1.
	x_2 =  output_matrix_5[int_list2[datasets],w_TEX86]

	output_1[datasets,:] = lead_lag_det(x_1,t_vals[w_d13C_org],x_2,t_vals[w_TEX86])
w1 = numpy.where(numpy.isnan(output_1[:,0])==False)[0]

n, bins, patches = ax1.hist(output_1[w1,0], range=(-10.5,10.5), bins=num_bins, cumulative=True, density=True,color='forestgreen',alpha=0.5,histtype='stepfilled')
hist, bin_edges = numpy.histogram(output_1[w1,0], range=(-10.5,10.5),bins=num_bins, density=True)
cdf = numpy.cumsum(hist*numpy.diff(bin_edges))
x_bins = (bin_edges[1:] + bin_edges[0:-1])/2.

ax1.plot(bins_org, cdf_org,color='forestgreen', linestyle='-',linewidth=2.0,label="$\delta$$^{13}$C$_{org}$")

ax1.set_xticks(numpy.arange(-10,10,5.))
ax1.set_xticks(numpy.arange(-10,10,1.), minor=True)
ax1.set_yticks(numpy.arange(0.,1.,0.1), minor=True)
ax1.set_xlim(-10.,10.)
ax1.set_ylim(0.0,1.0)

ax1.set_yticklabels([])

ax1.grid(alpha=0.0,which='both')

##
##
ax1 = plt.subplot(248)
numpy.random.shuffle(int_list1)
numpy.random.shuffle(int_list2)
for datasets in range(0,iterations):

	x_1 =  output_matrix_5[int_list1[datasets],w_d13C_PF]*-1.
	x_2 =  output_matrix_5[int_list2[datasets],w_TEX86]

	output_1[datasets,:] = lead_lag_det(x_1,t_vals[w_d13C_PF],x_2,t_vals[w_TEX86])
w1 = numpy.where(numpy.isnan(output_1[:,0])==False)[0]
n, bins, patches = ax1.hist(output_1[w1,0], range=(-10.5,10.5), bins=num_bins, cumulative=True, density=True,color='dodgerblue',alpha=0.5,histtype='stepfilled')
hist, bin_edges = numpy.histogram(output_1[w1,0], range=(-10.5,10.5),bins=num_bins, density=True)
cdf = numpy.cumsum(hist*numpy.diff(bin_edges))
x_bins = (bin_edges[1:] + bin_edges[0:-1])/2.

ax1.plot(bins_BF, cdf_BF,color='dodgerblue', linestyle='-',linewidth=2.0,label="$\delta$$^{13}$C$_{BF}$")
numpy.random.shuffle(int_list1)
numpy.random.shuffle(int_list2)
for datasets in range(0,iterations):

	x_1 =  output_matrix_5[int_list1[datasets],w_d13C_BF]*-1.
	x_2 =  output_matrix_5[int_list2[datasets],w_TEX86]

	output_1[datasets,:] = lead_lag_det(x_1,t_vals[w_d13C_BF],x_2,t_vals[w_TEX86])
w1 = numpy.where(numpy.isnan(output_1[:,0])==False)[0]
n, bins, patches = ax1.hist(output_1[w1,0], range=(-10.5,10.5), bins=num_bins, cumulative=True, density=True,color='indigo',alpha=0.5,histtype='stepfilled')
hist, bin_edges = numpy.histogram(output_1[w1,0], range=(-10.5,10.5),bins=num_bins, density=True)
cdf = numpy.cumsum(hist*numpy.diff(bin_edges))
x_bins = (bin_edges[1:] + bin_edges[0:-1])/2.

ax1.plot(bins_PF, cdf_PF,color='indigo', linestyle='-',linewidth=2.0,label="$\delta$$^{13}$C$_{PF}$")

ax1.set_xticks(numpy.arange(-10,10,5.))
ax1.set_xticks(numpy.arange(-10,10,1.), minor=True)
ax1.set_yticks(numpy.arange(0.,1.,0.1), minor=True)
ax1.set_xlim(-10.,10.)
ax1.set_ylim(0.0,1.0)

ax1.set_yticklabels([])

ax1.grid(alpha=0.0,which='both')

plt.suptitle("Lag relative to TEX86 ('Correlation units')", y=0.05,fontsize=10.)

	
pltname = wkspc +  'NJ_shelf_lead_lag_prob_20210615.png'
plt.savefig(pltname, dpi = 300)
pltname = wkspc +  'NJ_shelf_lead_lag_prob_20230314.pdf'
plt.savefig(pltname, dpi = 300)
plt.close()
######