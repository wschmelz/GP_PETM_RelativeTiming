import os
import sys
import datetime
import re
import shutil
import time
from time import gmtime, strftime
import csv
import numpy
import scipy
import loess2D

from numpy import matrix
from numpy import genfromtxt
from numpy import linalg

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
	SL_data_2[:,2] = (SL_data[:,2] * 0.0)
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

		SL_data_2[:,0] = SL_data_2[:,0] *1.0
		w1 = numpy.where((SL_data_2[:,0]>317.441) & (SL_data_2[:,0]<378.23))[0]	
	if float(SL_table.split("\\")[1][4:6]) == 4.:
		
		SL_data_2[:,0] = SL_data_2[:,0] *1.0 # - 56.2
		w1 = numpy.where((SL_data_2[:,0]>48.07) & (SL_data_2[:,0]<68.9))[0]	
			
	if float(SL_table.split("\\")[1][4:6]) == 5.:
		
		SL_data_2[:,0] = SL_data_2[:,0] #- 56.2
		w1 = numpy.where((SL_data_2[:,0]>=51.02) & (SL_data_2[:,0]<=60.98))[0]			
	
	
	if float(SL_table.split("\\")[1][4:6]) >= 6.:
		w1 = numpy.where((SL_data_2[:,0]>=-100000000.0) & (SL_data_2[:,0]<=100000000.))[0]			
		
	if n == 0:
		SL_data_combined = SL_data_2[w1,:]	
	
	if n > 0:
		SL_data_combined = numpy.append(SL_data_combined,SL_data_2[w1,:],axis=0)


sites = numpy.unique(SL_data_combined[:,3])

site_matrix_depth = numpy.zeros((5,4))

site_locations = []

for n in range(4,6):
	if n == 0.:

		w1 = numpy.where((SL_data_combined[:,3] == n))[0]
		site_locations.append(w1,)
		
		site_matrix_depth[n,:] = 1138.6,1158.2,1172.2,1184.8
		
	if n == 1.:

		w1 = numpy.where((SL_data_combined[:,3] == n))[0]
		site_locations.append(w1,)
		
		site_matrix_depth[n,:] = 846.7,865.1,898.2,910.75
		
	if n == 2.:

		w1 = numpy.where((SL_data_combined[:,3] == n))[0]
		site_locations.append(w1,)
		
		site_matrix_depth[n,:] = 521.86,542.64,562.1,574.74
		
	if n == 3.:

		w1 = numpy.where((SL_data_combined[:,3] == n))[0]
		site_locations.append(w1,)
		
		site_matrix_depth[n,:] = 317.441,331.42,365.7,378.23
		
	if n == 4.:

		w1 = numpy.where((SL_data_combined[:,3] == n))[0]
		site_locations.append(w1,)

		sed_rate1 = 0.7
		depth_mod = 73.2916667 * sed_rate1
		depth_mod2 = (153.28125-73.2916667) * sed_rate1
		site_matrix_depth[n,:] = (56.180879409999996-depth_mod) - depth_mod2,56.180879409999996-depth_mod,56.180879409999996,69.302
		
site_matrix_age = numpy.zeros((5,4))

for n in range(0,5):

	site_matrix_age[n,:] = 153.28125,73.2916667,0.0,-192.357481

##Covariance functions

def matern(sigma,numer,denom,data):
	
	if numer >= 5.0:
		ret_val = (sigma**2.) * (1.+ ((numpy.sqrt(5.) * (data))/denom) + ((5. * (data**2))/(3.*(denom**2.)))) *  numpy.exp(-1.*((numpy.sqrt(5.) * (data))/denom)) 
	if numer > 1.0 and numer <5.0:
		ret_val = (sigma**2.) * (1.+((numpy.sqrt(3.) * (data))/denom)) *  numpy.exp(-1.*((numpy.sqrt(3.) * (data))/denom)) 
	if numer <= 1.0:
		ret_val = (sigma**2.) * numpy.exp(-1.*((1. * (data))/denom))
	return ret_val 
	
def linear1(sigma1,sigma2,data,s_mat1):
	
	ret_val = ((sigma1**2.) * s_mat1) + ((((sigma2**2.) * s_mat1)) * data)
	
	return ret_val 	
	
def linear2(sigma3,sigma4,sigma5,sigma6,data,s_mat2,s_mat3):
	
	ret_val = ((sigma3**2.)* s_mat2) + ((sigma4**2.)* s_mat3) + ((((sigma5**2.) * s_mat2) + ((sigma6**2.) * s_mat3)) * data)
	
	return ret_val 	

mat_deg_SL = 3.
mat_deg_temp = 3.
mat_deg_CO2 = 1.

def MATERN_X_2(hyp1_in,hyp2_in,t_matrix_i,mat_deg_in):

	ls_time_t = 1.0
		
	new_dists_t = numpy.sqrt((t_matrix_i/(ls_time_t))**2.)

	return matern(hyp1_in,mat_deg_in,numpy.absolute(hyp2_in),new_dists_t) 
	
def linear_X_2(sigma1,sigma2,data):

	ls_time_t = 1.0
		
	new_dists_t = numpy.sqrt((t_matrix_i/(ls_time_t))**2.)

	return linear(hyp1_in,mat_deg_in,numpy.absolute(hyp2_in),new_dists_t) 	

def SQ_EXP_PER(hyp1_in,hyp2_in,hyp3_in,periodicity_in,t_matrix_i,output_mat):

	ls_time_t = 1.0
		
	new_dists_t = (t_matrix_i/(ls_time_t))

	return (hyp1_in**2.) * numpy.exp((-1. * ((new_dists_t**2.)/(2.*(hyp2_in**2.)))) - (((2.*(numpy.sin((numpy.pi*new_dists_t)/periodicity_in)**2.)))/(hyp3_in**2.)) )
	
def RQ(hyp1_in,hyp2_in,hyp3_in,t_matrix_i,output_mat):

	ls_time_t = 1.0
		
	new_dists_t = (t_matrix_i/(ls_time_t))

	return (hyp1_in**2.) * (( 1. + ((new_dists_t**2.)/(2.*hyp2_in*(hyp3_in**2.))) )** ( -1.*hyp2_in))	


def WHE_NSE(hyp1_in,hyp2_in,hyp3_in,hyp4_in,t_matrix_i,noise_mat_in,s_mat1,s_mat2,s_mat3,s_mat4):
	
	return (noise_mat_in) + ((s_mat1*(numpy.identity(len(t_matrix_i[:,0])))) * (hyp1_in **2.)) +  ((s_mat2*(numpy.identity(len(t_matrix_i[:,0])))) * (hyp2_in **2.)) + ((s_mat3*(numpy.identity(len(t_matrix_i[:,0])))) * (hyp3_in **2.)) + ((s_mat4*(numpy.identity(len(t_matrix_i[:,0])))) * (hyp4_in **2.))
	
	
MCMC_iterations = 2000

per_year = 0.1

ice_end = 60.

mean_x = numpy.mean(SL_data_combined[:,0])

t_vals = numpy.arange(51.,ice_end+ per_year,per_year) - mean_x

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

m1 =len(t_vals)

newy_mat = numpy.zeros((1,m1))
newy_p_mat = numpy.zeros((1,m1))

for index_Kdat in range(2,3):
	print (index_Kdat)
	if index_Kdat != 2011 and index_Kdat != 2012:
		hyperparams_tmp = wkspc + "posterior_hyperparams" + str(index_Kdat) + ".csv"
		hyperparams_tmp = numpy.genfromtxt(hyperparams_tmp,delimiter=',')
	if index_Kdat == 2:
		K_data = hyperparams_tmp[100000:,:]*1.0

w1 = numpy.where(numpy.isnan(K_data[:,0]))[0]

K_data = numpy.delete(K_data,w1,axis=0)

####SL1 Params#####

mean_x = numpy.mean(SL_data_combined[:,0])
xes1_t_1 = SL_data_combined[:,0] - mean_x

mean_val =  numpy.mean(SL_data_combined[:,1])
xes2_sl_1 = SL_data_combined[:,1]
xes4_error_1 = SL_data_combined[:,2]
xes3_type_1 = SL_data_combined[:,3]
xes5_type_2 = SL_data_combined[:,4]

unique_data_types = numpy.unique(xes5_type_2)
type_mean = []

for n in range(0,len(unique_data_types)):

	typ_idx = numpy.where(SL_data_combined[:, 4] == unique_data_types[n])[0]
	mean_val1 = numpy.mean(SL_data_combined[typ_idx, 1])
	type_mean.append(mean_val1)
	
	SL_data_combined[typ_idx,1] = SL_data_combined[typ_idx,1] - mean_val1


n_SL_1 = len(xes2_sl_1)

output_matrix_7 = numpy.zeros((MCMC_iterations,n_SL_1))

y_1 = numpy.reshape(xes2_sl_1,(-1,1))	
y_transpose_1 = numpy.transpose(y_1)

noise_mat_1 = (1.*(numpy.identity(n_SL_1)))

for index1 in range(0,n_SL_1):
	noise_mat_1[index1,index1] = noise_mat_1[index1,index1] * (xes4_error_1[index1]**2.)

###

###Global d13C matern

s_matrix_1_tmp = numpy.repeat(numpy.reshape(xes5_type_2,(1,-1)),n_SL_1,axis=0)
s_matrix_2_tmp = numpy.repeat(numpy.reshape(xes5_type_2,(-1,1)),n_SL_1,axis=1)

w1 = numpy.where((s_matrix_1_tmp!=2.)&(s_matrix_2_tmp!=2.))

s_matrix_1 = numpy.zeros(numpy.shape(s_matrix_1_tmp))
s_matrix_1[w1[0],w1[1]] = s_matrix_1[w1[0],w1[1]] + 1.0

###NJ d13C bulk and d13C org and TEX86 matern

s_matrix_1_tmp = numpy.repeat(numpy.reshape(xes5_type_2,(1,-1)),n_SL_1,axis=0)
s_matrix_2_tmp = numpy.repeat(numpy.reshape(xes5_type_2,(-1,1)),n_SL_1,axis=1)

w1 = numpy.where((s_matrix_1_tmp==s_matrix_2_tmp)&((s_matrix_2_tmp<6.)&(s_matrix_2_tmp!=2.)))

s_matrix_2 = numpy.zeros(numpy.shape(s_matrix_1_tmp))
s_matrix_2[w1[0],w1[1]] = s_matrix_2[w1[0],w1[1]] + 1.0

s_matrix_1_tmp = numpy.repeat(numpy.reshape(xes5_type_2,(1,-1)),n_SL_1,axis=0)
s_matrix_2_tmp = numpy.repeat(numpy.reshape(xes5_type_2,(-1,1)),n_SL_1,axis=1)

w1 = numpy.where((s_matrix_1_tmp==s_matrix_2_tmp)&(s_matrix_2_tmp==2.))

s_matrix_3 = numpy.zeros(numpy.shape(s_matrix_1_tmp))
s_matrix_3[w1[0],w1[1]] = s_matrix_3[w1[0],w1[1]] + 1.0


###Local matern

s_matrix_1_tmp = numpy.repeat(numpy.reshape(xes5_type_2,(1,-1)),n_SL_1,axis=0)
s_matrix_2_tmp = numpy.repeat(numpy.reshape(xes5_type_2,(-1,1)),n_SL_1,axis=1)

w1_tmp = (s_matrix_1_tmp==s_matrix_2_tmp)&(s_matrix_2_tmp!=2.)

s_matrix_1_tmp = numpy.repeat(numpy.reshape(xes3_type_1,(1,-1)),n_SL_1,axis=0)
s_matrix_2_tmp = numpy.repeat(numpy.reshape(xes3_type_1,(-1,1)),n_SL_1,axis=1)

w2_tmp = (s_matrix_1_tmp==s_matrix_2_tmp)&((s_matrix_2_tmp<6.))

w1 = numpy.where((w2_tmp) & (w1_tmp))

s_matrix_4 = numpy.zeros(numpy.shape(s_matrix_1_tmp))
s_matrix_4[w1[0],w1[1]] = s_matrix_4[w1[0],w1[1]] + 1.0

s_matrix_1_tmp = numpy.repeat(numpy.reshape(xes5_type_2,(1,-1)),n_SL_1,axis=0)
s_matrix_2_tmp = numpy.repeat(numpy.reshape(xes5_type_2,(-1,1)),n_SL_1,axis=1)

w1_tmp = ((s_matrix_1_tmp==s_matrix_2_tmp)&(s_matrix_2_tmp==2.))

s_matrix_1_tmp = numpy.repeat(numpy.reshape(xes3_type_1,(1,-1)),n_SL_1,axis=0)
s_matrix_2_tmp = numpy.repeat(numpy.reshape(xes3_type_1,(-1,1)),n_SL_1,axis=1)

w2_tmp = (s_matrix_1_tmp==s_matrix_2_tmp)

w1 = numpy.where((w2_tmp) & (w1_tmp))

s_matrix_5 = numpy.zeros(numpy.shape(s_matrix_1_tmp))
s_matrix_5[w1[0],w1[1]] = s_matrix_5[w1[0],w1[1]] + 1.0

###Error indicators

s_matrix_1_tmp = numpy.repeat(numpy.reshape(xes5_type_2,(1,-1)),n_SL_1,axis=0)
s_matrix_2_tmp = numpy.repeat(numpy.reshape(xes5_type_2,(-1,1)),n_SL_1,axis=1)

w1 = numpy.where((s_matrix_1_tmp == s_matrix_2_tmp) & (s_matrix_1_tmp==0.))

s_matrix_9 = numpy.zeros(numpy.shape(s_matrix_1_tmp))
s_matrix_9[w1[0],w1[1]] = s_matrix_9[w1[0],w1[1]] + 1.0

w1 = numpy.where((s_matrix_1_tmp == s_matrix_2_tmp) & (s_matrix_1_tmp==1.))

s_matrix_10 = numpy.zeros(numpy.shape(s_matrix_1_tmp))
s_matrix_10[w1[0],w1[1]] = s_matrix_10[w1[0],w1[1]] + 1.0

w1 = numpy.where((s_matrix_1_tmp == s_matrix_2_tmp) & (s_matrix_1_tmp==2.))

s_matrix_11 = numpy.zeros(numpy.shape(s_matrix_1_tmp))
s_matrix_11[w1[0],w1[1]] = s_matrix_11[w1[0],w1[1]] + 1.0

w1 = numpy.where((s_matrix_1_tmp == s_matrix_2_tmp) & ((s_matrix_1_tmp==3.)|(s_matrix_1_tmp==4.)))

s_matrix_12 = numpy.zeros(numpy.shape(s_matrix_1_tmp))
s_matrix_12[w1[0],w1[1]] = s_matrix_12[w1[0],w1[1]] + 1.0

w1 = numpy.where((s_matrix_1_tmp == s_matrix_2_tmp) & (s_matrix_1_tmp==5.))

s_matrix_13 = numpy.zeros(numpy.shape(s_matrix_1_tmp))
s_matrix_13[w1[0],w1[1]] = s_matrix_13[w1[0],w1[1]] + 1.0


###

###Global d13C matern

s_matrix_1_tmp = numpy.repeat(numpy.reshape(s_vals2,(1,-1)),len(xes5_type_2),axis=0)
s_matrix_2_tmp = numpy.repeat(numpy.reshape(xes5_type_2,(-1,1)),len(s_vals2),axis=1)

w1 = numpy.where((s_matrix_1_tmp!=2.) & (s_matrix_2_tmp!=2.))

s_matrix_1_b = numpy.zeros(numpy.shape(s_matrix_1_tmp))
s_matrix_1_b[w1[0],w1[1]] = s_matrix_1_b[w1[0],w1[1]] + 1.0

###NJ d13C bulk and d13C org and TEX86 matern

s_matrix_1_tmp = numpy.repeat(numpy.reshape(s_vals2,(1,-1)),len(xes5_type_2),axis=0)
s_matrix_2_tmp = numpy.repeat(numpy.reshape(xes5_type_2,(-1,1)),len(s_vals2),axis=1)

w1 = numpy.where((s_matrix_1_tmp==s_matrix_2_tmp)&((s_matrix_2_tmp<6.)&(s_matrix_2_tmp!=2.)))

s_matrix_2_b = numpy.zeros(numpy.shape(s_matrix_1_tmp))
s_matrix_2_b[w1[0],w1[1]] = s_matrix_2_b[w1[0],w1[1]] + 1.0

s_matrix_1_tmp = numpy.repeat(numpy.reshape(s_vals2,(1,-1)),len(xes5_type_2),axis=0)
s_matrix_2_tmp = numpy.repeat(numpy.reshape(xes5_type_2,(-1,1)),len(s_vals2),axis=1)

w1 = numpy.where((s_matrix_1_tmp==s_matrix_2_tmp)&(s_matrix_2_tmp==2.))

s_matrix_3_b = numpy.zeros(numpy.shape(s_matrix_1_tmp))
s_matrix_3_b[w1[0],w1[1]] = s_matrix_3_b[w1[0],w1[1]] + 1.0

###Local matern

s_matrix_1_tmp = numpy.repeat(numpy.reshape(s_vals2,(1,-1)),len(xes5_type_2),axis=0)
s_matrix_2_tmp = numpy.repeat(numpy.reshape(xes5_type_2,(-1,1)),len(s_vals2),axis=1)

w1_tmp = (s_matrix_1_tmp==s_matrix_2_tmp)&(s_matrix_2_tmp!=2.)

s_matrix_1_tmp = numpy.repeat(numpy.reshape(s_vals1,(1,-1)),len(xes3_type_1),axis=0)
s_matrix_2_tmp = numpy.repeat(numpy.reshape(xes3_type_1,(-1,1)),len(s_vals1),axis=1)

w2_tmp = (s_matrix_1_tmp==s_matrix_2_tmp)&((s_matrix_2_tmp<6.))

w1 = numpy.where((w2_tmp) & (w1_tmp))

s_matrix_4_b = numpy.zeros(numpy.shape(s_matrix_1_tmp))
s_matrix_4_b[w1[0],w1[1]] = s_matrix_4_b[w1[0],w1[1]] + 1.0

s_matrix_1_tmp = numpy.repeat(numpy.reshape(s_vals2,(1,-1)),len(xes5_type_2),axis=0)
s_matrix_2_tmp = numpy.repeat(numpy.reshape(xes5_type_2,(-1,1)),len(s_vals2),axis=1)

w1_tmp = ((s_matrix_1_tmp==s_matrix_2_tmp)&(s_matrix_2_tmp==2.))

s_matrix_1_tmp = numpy.repeat(numpy.reshape(s_vals1,(1,-1)),len(xes3_type_1),axis=0)
s_matrix_2_tmp = numpy.repeat(numpy.reshape(xes3_type_1,(-1,1)),len(s_vals1),axis=1)

w2_tmp = (s_matrix_1_tmp==s_matrix_2_tmp)

w1 = numpy.where((w2_tmp) & (w1_tmp))

s_matrix_5_b = numpy.zeros(numpy.shape(s_matrix_1_tmp))
s_matrix_5_b[w1[0],w1[1]] = s_matrix_5_b[w1[0],w1[1]] + 1.0


###

###Global d13C matern

s_matrix_1_tmp = numpy.repeat(numpy.reshape(s_vals2,(1,-1)),m1,axis=0)
s_matrix_2_tmp = numpy.repeat(numpy.reshape(s_vals2,(-1,1)),m1,axis=1)

w1 = numpy.where((s_matrix_1_tmp!=2.) & (s_matrix_2_tmp!=2.))

s_matrix_1_c = numpy.zeros(numpy.shape(s_matrix_1_tmp))
s_matrix_1_c[w1[0],w1[1]] = s_matrix_1_c[w1[0],w1[1]] + 1.0

###NJ d13C bulk and d13C org and TEX86 matern

s_matrix_1_tmp = numpy.repeat(numpy.reshape(s_vals2,(1,-1)),m1,axis=0)
s_matrix_2_tmp = numpy.repeat(numpy.reshape(s_vals2,(-1,1)),m1,axis=1)

w1 = numpy.where((s_matrix_1_tmp==s_matrix_2_tmp)&((s_matrix_2_tmp<6.)&(s_matrix_2_tmp!=2.)))

s_matrix_2_c = numpy.zeros(numpy.shape(s_matrix_1_tmp))
s_matrix_2_c[w1[0],w1[1]] = s_matrix_2_c[w1[0],w1[1]] + 1.0

s_matrix_1_tmp = numpy.repeat(numpy.reshape(s_vals2,(1,-1)),m1,axis=0)
s_matrix_2_tmp = numpy.repeat(numpy.reshape(s_vals2,(-1,1)),m1,axis=1)

w1 = numpy.where((s_matrix_1_tmp==s_matrix_2_tmp)&(s_matrix_2_tmp==2.))

s_matrix_3_c = numpy.zeros(numpy.shape(s_matrix_1_tmp))
s_matrix_3_c[w1[0],w1[1]] = s_matrix_3_c[w1[0],w1[1]] + 1.0

###Local matern

s_matrix_1_tmp = numpy.repeat(numpy.reshape(s_vals2,(1,-1)),m1,axis=0)
s_matrix_2_tmp = numpy.repeat(numpy.reshape(s_vals2,(-1,1)),m1,axis=1)

w1_tmp = (s_matrix_1_tmp==s_matrix_2_tmp)&(s_matrix_2_tmp!=2.)

s_matrix_1_tmp = numpy.repeat(numpy.reshape(s_vals1,(1,-1)),m1,axis=0)
s_matrix_2_tmp = numpy.repeat(numpy.reshape(s_vals1,(-1,1)),m1,axis=1)

w2_tmp = (s_matrix_1_tmp==s_matrix_2_tmp)&((s_matrix_2_tmp<6.))

w1 = numpy.where((w2_tmp) & (w1_tmp))

s_matrix_4_c = numpy.zeros(numpy.shape(s_matrix_1_tmp))
s_matrix_4_c[w1[0],w1[1]] = s_matrix_4_c[w1[0],w1[1]] + 1.0

s_matrix_1_tmp = numpy.repeat(numpy.reshape(s_vals2,(1,-1)),m1,axis=0)
s_matrix_2_tmp = numpy.repeat(numpy.reshape(s_vals2,(-1,1)),m1,axis=1)

w1_tmp = ((s_matrix_1_tmp==s_matrix_2_tmp)&(s_matrix_2_tmp==2.))

s_matrix_1_tmp = numpy.repeat(numpy.reshape(s_vals1,(1,-1)),m1,axis=0)
s_matrix_2_tmp = numpy.repeat(numpy.reshape(s_vals1,(-1,1)),m1,axis=1)

w2_tmp = (s_matrix_1_tmp==s_matrix_2_tmp)

w1 = numpy.where((w2_tmp) & (w1_tmp))

s_matrix_5_c = numpy.zeros(numpy.shape(s_matrix_1_tmp))
s_matrix_5_c[w1[0],w1[1]] = s_matrix_5_c[w1[0],w1[1]] + 1.0

###

file_out_1 = wkspc + 'MC_probability_output_1.csv'	
file_out_2 = wkspc + 'MC_probability_output_2.csv'	
file_out_3 = wkspc + 'MC_probability_output_3.csv'	
file_out_4 = wkspc + 'MC_probability_output_4.csv'	
file_out_5 = wkspc + 'MC_probability_output_5.csv'	
file_out_6 = wkspc + 'MC_probability_output_6.csv'	
file_out_7 = wkspc + 'MC_probability_output_7.csv'	
file_out_8 = wkspc + 'MC_probability_output_8.csv'	

output_matrix_1 = numpy.zeros((MCMC_iterations,len(t_vals)))
output_matrix_2 = numpy.zeros((MCMC_iterations,len(t_vals)))
output_matrix_3 = numpy.zeros((MCMC_iterations,len(t_vals)))
output_matrix_4 = numpy.zeros((MCMC_iterations,len(t_vals)))
output_matrix_5 = numpy.zeros((MCMC_iterations,len(t_vals)))
output_matrix_6 = numpy.zeros((MCMC_iterations,len(t_vals)))
output_matrix_8 = numpy.zeros((MCMC_iterations,len(t_vals)))

int_list = numpy.arange(0,len(K_data[:,0]),1)
numpy.random.shuffle(int_list)

t_matrix_1 = numpy.sqrt((numpy.repeat(numpy.reshape(xes1_t_1,(1,-1)),n_SL_1,axis=0) - numpy.repeat(numpy.reshape(xes1_t_1,(-1,1)),n_SL_1,axis=1))**2.)
new_mults_t_1 = numpy.repeat(numpy.reshape(xes1_t_1,(1,-1)),n_SL_1,axis=0) * numpy.repeat(numpy.reshape(xes1_t_1,(-1,1)),n_SL_1,axis=1)

t_new_1 = numpy.sqrt((numpy.repeat(numpy.reshape(t_vals,(1,-1)),len(xes1_t_1),axis=0) - numpy.repeat(numpy.reshape(xes1_t_1,(-1,1)),len(t_vals),axis=1))**2.)
t_mults_new_1 = numpy.repeat(numpy.reshape(t_vals,(1,-1)),len(xes1_t_1),axis=0) * numpy.repeat(numpy.reshape(xes1_t_1,(-1,1)),len(t_vals),axis=1)

t_matrix2_1 = numpy.sqrt((numpy.repeat(numpy.reshape(t_vals,(1,-1)),m1,axis=0) - numpy.repeat(numpy.reshape(t_vals,(-1,1)),m1,axis=1))**2.)
t_matrix2_mult_1 = numpy.repeat(numpy.reshape(t_vals,(1,-1)),m1,axis=0) * numpy.repeat(numpy.reshape(t_vals,(-1,1)),m1,axis=1)


for MCMC_index in range(0,MCMC_iterations):

	guess1 =  K_data[int_list[MCMC_index],:]	
	
	################################

	K1_1 = matern(guess1[0],3.,numpy.absolute(guess1[1]),t_matrix_1) * s_matrix_1

	K2_1 = matern(guess1[2],3.,numpy.absolute(guess1[3]),t_matrix_1) * s_matrix_2

	K3_1 = matern(guess1[4],3.,numpy.absolute(guess1[5]),t_matrix_1) * s_matrix_3
	
	#K4_1 = matern(guess1[6],3.,numpy.absolute(guess1[7]),t_matrix_1) * s_matrix_3

	K5_1 = linear1(guess1[6],guess1[7],new_mults_t_1,s_matrix_1)
	K6_1 = linear1(guess1[8],guess1[9],new_mults_t_1,s_matrix_2)
	K7_1 = linear1(guess1[10],guess1[11],new_mults_t_1,s_matrix_3)
	
	K8_1 = matern(guess1[12],1.,numpy.absolute(guess1[13]),t_matrix_1) * s_matrix_4

	K9_1 = matern(guess1[14],1.,numpy.absolute(guess1[15]),t_matrix_1) * s_matrix_5
	
	WN = WHE_NSE(guess1[16],guess1[17],guess1[18],guess1[19],t_matrix_1,noise_mat_1,s_matrix_9,s_matrix_10,s_matrix_11,s_matrix_12)

	K = K1_1 + K2_1 + K3_1 + K5_1 +  K6_1 + K7_1 + K8_1 + K9_1 + WN #+ K4_1 
		
	K_inv = numpy.linalg.inv(K)
	
	#####
	
	K1_2 =  matern(guess1[0],3.,numpy.absolute(guess1[1]),t_new_1) * s_matrix_1_b
	
	K2_2 =  matern(guess1[2],3.,numpy.absolute(guess1[3]),t_new_1) * s_matrix_2_b
	
	K3_2 =  matern(guess1[4],3.,numpy.absolute(guess1[5]),t_new_1) * s_matrix_3_b
	
	#K4_2 =  matern(guess1[6],3.,numpy.absolute(guess1[7]),t_new_1) * s_matrix_3_b

	K5_2 =  linear1(guess1[6],guess1[7],t_mults_new_1,s_matrix_1_b)
	K6_2 =  linear1(guess1[8],guess1[9],t_mults_new_1,s_matrix_2_b)	
	K7_2 =  linear1(guess1[10],guess1[11],t_mults_new_1,s_matrix_3_b)

	K8_2 =  matern(guess1[12],1.,numpy.absolute(guess1[13]),t_new_1) * s_matrix_4_b
	
	K9_2 =  matern(guess1[14],1.,numpy.absolute(guess1[15]),t_new_1) * s_matrix_5_b
	
	K2_f = K1_2 + K2_2 +  K3_2 + K5_2 + K6_2 + K7_2 + K8_2 + K9_2 #+ K4_2 

	new_y_p = numpy.matmul(K_inv,K2_f) 
	new_y_p_2 = numpy.matmul(numpy.transpose(K2_f),new_y_p)

	#####
	
	K1_3 =  matern(guess1[0],3.,numpy.absolute(guess1[1]),t_matrix2_1) * s_matrix_1_c
	
	K2_3 =  matern(guess1[2],3.,numpy.absolute(guess1[3]),t_matrix2_1) * s_matrix_2_c
	
	K3_3 =  matern(guess1[4],3.,numpy.absolute(guess1[5]),t_matrix2_1) * s_matrix_3_c
	
	#K4_3 =  matern(guess1[6],3.,numpy.absolute(guess1[7]),t_matrix2_1) * s_matrix_3_c

	K5_3 =  linear1(guess1[6],guess1[7],t_matrix2_mult_1,s_matrix_1_c)
	K6_3 =  linear1(guess1[8],guess1[9],t_matrix2_mult_1,s_matrix_2_c)
	K7_3 =  linear1(guess1[10],guess1[11],t_matrix2_mult_1,s_matrix_3_c)
	
	K8_3 =  matern(guess1[12],1.,numpy.absolute(guess1[13]),t_matrix2_1) * s_matrix_4_c

	K9_3 =  matern(guess1[14],1.,numpy.absolute(guess1[15]),t_matrix2_1) * s_matrix_5_c
	
	K_2 = K1_3 + K2_3 + K3_3 + K5_3 + K6_3 + K7_3 + K8_3 + K9_3 # + K4_3

	new_y = numpy.matmul(numpy.transpose(K2_f),numpy.matmul(K_inv,numpy.reshape(y_1,(-1,1))))

	newy_mat[0,:] = numpy.ndarray.flatten(new_y)
	newy_p_mat[0,:] = numpy.sqrt(numpy.diag(K_2 - new_y_p_2))
	
	output_matrix_1[MCMC_index,:] = newy_mat[0,:] + (numpy.random.standard_normal(size=len(newy_p_mat[0,:])) * newy_p_mat[0,:])
	output_matrix_5[MCMC_index,:] = numpy.random.multivariate_normal(mean=numpy.ndarray.flatten(new_y), cov=(K_2 - new_y_p_2), size=1)
		
	################################

	#####
	
	K2_f = K1_2 + K2_2 +  K3_2 + K5_2 + K6_2 + K7_2 #+ K4_2 

	new_y_p = numpy.matmul(K_inv,K2_f) 
	new_y_p_2 = numpy.matmul(numpy.transpose(K2_f),new_y_p)

	#####
	
	K_2 = K1_3 + K2_3 + K3_3 + K5_3 + K6_3 + K7_3 #+ K4_3 
	
	new_y = numpy.matmul(numpy.transpose(K2_f),numpy.matmul(K_inv,numpy.reshape(y_1,(-1,1))))

	newy_mat[0,:] = numpy.ndarray.flatten(new_y)
	newy_p_mat[0,:] = numpy.sqrt(numpy.diag(K_2 - new_y_p_2))
	
	output_matrix_2[MCMC_index,:] = newy_mat[0,:] + (numpy.random.standard_normal(size=len(newy_p_mat[0,:])) * newy_p_mat[0,:])
	output_matrix_6[MCMC_index,:] = numpy.random.multivariate_normal(mean=numpy.ndarray.flatten(new_y), cov=(K_2 - new_y_p_2), size=1)

	################################

	#####

	K2_f = K1_2 + K3_2 + K5_2 + K7_2 #+ K4_2 

	new_y_p = numpy.matmul(K_inv,K2_f) 
	new_y_p_2 = numpy.matmul(numpy.transpose(K2_f),new_y_p)

	#####

	K_2 = K1_3 + K3_3 + K5_3 + K7_3 # + K4_3 
	
	new_y = numpy.matmul(numpy.transpose(K2_f),numpy.matmul(K_inv,numpy.reshape(y_1,(-1,1))))

	newy_mat[0,:] = numpy.ndarray.flatten(new_y)
	newy_p_mat[0,:] = numpy.sqrt(numpy.diag(K_2 - new_y_p_2))
	
	output_matrix_3[MCMC_index,:] = newy_mat[0,:] + (numpy.random.standard_normal(size=len(newy_p_mat[0,:])) * newy_p_mat[0,:])
	output_matrix_8[MCMC_index,:] = numpy.random.multivariate_normal(mean=numpy.ndarray.flatten(new_y), cov=(K_2 - new_y_p_2), size=1)
	
	#####
	
	output_matrix_7[MCMC_index,:] = xes1_t_1 + mean_x
	
	#####

	for n in range(0,len(unique_data_types)):

		typ_idx = numpy.where(s_vals2 == unique_data_types[n])[0]
		output_matrix_1[MCMC_index,typ_idx] += type_mean[n]
		output_matrix_2[MCMC_index,typ_idx] += type_mean[n]
		output_matrix_3[MCMC_index,typ_idx] += type_mean[n]
		output_matrix_5[MCMC_index,typ_idx] += type_mean[n]
		output_matrix_6[MCMC_index,typ_idx] += type_mean[n]
		output_matrix_8[MCMC_index,typ_idx] += type_mean[n]

	if MCMC_index % 25 ==0:
	
		t2 = float(time.time())
		time_per = (t2 - t1) / (float(MCMC_index) + 1.)
		
		sys.stdout.write("\r i: %s   " % (str(numpy.round(time_per,5)))) 
		
		print ("...")

		with open(file_out_1, 'w',newline="") as csvfile2:
			writer = csv.writer(csvfile2)
			writer.writerows(output_matrix_1)	
		
		with open(file_out_2, 'w',newline="") as csvfile2:
			writer = csv.writer(csvfile2)
			writer.writerows(output_matrix_2)	

		with open(file_out_3, 'w',newline="") as csvfile2:
			writer = csv.writer(csvfile2)
			writer.writerows(output_matrix_3)
		
		with open(file_out_4, 'w',newline="") as csvfile2:
			writer = csv.writer(csvfile2)
			writer.writerows(output_matrix_7)				
		
		with open(file_out_5, 'w',newline="") as csvfile2:
			writer = csv.writer(csvfile2)
			writer.writerows(output_matrix_5)
		
		with open(file_out_6, 'w',newline="") as csvfile2:
			writer = csv.writer(csvfile2)
			writer.writerows(output_matrix_6)
		
		with open(file_out_8, 'w',newline="") as csvfile2:
			writer = csv.writer(csvfile2)
			writer.writerows(output_matrix_8)
		
		
print (time.strftime("%H:%M:%S"))

with open(file_out_1, 'w',newline="") as csvfile2:
	writer = csv.writer(csvfile2)
	writer.writerows(output_matrix_1)	

with open(file_out_2, 'w',newline="") as csvfile2:
	writer = csv.writer(csvfile2)
	writer.writerows(output_matrix_2)	

with open(file_out_3, 'w',newline="") as csvfile2:
	writer = csv.writer(csvfile2)
	writer.writerows(output_matrix_3)

with open(file_out_4, 'w',newline="") as csvfile2:
	writer = csv.writer(csvfile2)
	writer.writerows(output_matrix_7)				

with open(file_out_5, 'w',newline="") as csvfile2:
	writer = csv.writer(csvfile2)
	writer.writerows(output_matrix_5)

with open(file_out_6, 'w',newline="") as csvfile2:
	writer = csv.writer(csvfile2)
	writer.writerows(output_matrix_6)

with open(file_out_8, 'w',newline="") as csvfile2:
	writer = csv.writer(csvfile2)
	writer.writerows(output_matrix_8)