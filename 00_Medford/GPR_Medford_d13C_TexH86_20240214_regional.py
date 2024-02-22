import os
import sys
import datetime
import re
import shutil
import time
from time import gmtime, strftime
import csv
import numpy
import random
import loess2D
import glob
import scipy

from numpy import matrix
from numpy import genfromtxt
from numpy import linalg

backslash = '\\'

wkspc = str(os.getcwd()).replace(backslash,"/") + "/"

print (str(__file__))

print (time.strftime("%H:%M:%S"))

wkspc_data = wkspc + '01_Data/'

gpsdata = glob.glob(wkspc_data + "*.csv")

print (wkspc_data)

for n in range(0,len(gpsdata)):
	SL_table = gpsdata[n]
	SL_data = numpy.genfromtxt(SL_table,delimiter=',')
	
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
		SL_data_2[:,0] = SL_data_2[:,0] *1.0 # - 56.2
		w1 = numpy.where((SL_data_2[:,0]>=51.02) & (SL_data_2[:,0]<=60.98))[0]			
	
	
	if float(SL_table.split("\\")[1][4:6]) >= 6.:
		w1 = numpy.where((SL_data_2[:,0]>=-100000000.0) & (SL_data_2[:,0]<=100000000.))[0]			
		
	if n == 0:
		SL_data_combined = SL_data_2[w1,:]	
	
	if n > 0:
		SL_data_combined = numpy.append(SL_data_combined,SL_data_2[w1,:],axis=0)


sites = numpy.unique(SL_data_combined[:,3])

print(sites)

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
	
def linear3(sigma3,sigma4,s_mat2,s_mat3):
	
	ret_val = ((sigma3**2.)* s_mat2) + ((sigma4**2.)* s_mat3)
	
	return ret_val	

def MATERN_X_2(hyp1_in,hyp2_in,t_matrix_i,mat_deg_in,output_mat):

	ls_time_t = 1.0
		
	new_dists_t = numpy.sqrt((t_matrix_i/(ls_time_t))**2.)
	
	return matern(hyp1_in,mat_deg_in,numpy.absolute(hyp2_in),new_dists_t) 

def WHE_NSE(hyp1_in,hyp2_in,hyp3_in,hyp4_in,t_matrix_i,noise_mat_in,s_mat1,s_mat2,s_mat3,s_mat4):
	
	return (noise_mat_in) + ((s_mat1*(numpy.identity(len(t_matrix_i[:,0])))) * (hyp1_in **2.)) +  ((s_mat2*(numpy.identity(len(t_matrix_i[:,0])))) * (hyp2_in **2.)) + ((s_mat3*(numpy.identity(len(t_matrix_i[:,0])))) * (hyp3_in **2.)) + ((s_mat4*(numpy.identity(len(t_matrix_i[:,0])))) * (hyp4_in **2.))

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

y_1 = numpy.reshape(xes2_sl_1,(-1,1))	
y_transpose_1 = numpy.transpose(y_1)

noise_mat_1 = (1.*(numpy.identity(n_SL_1)))

for index1 in range(0,n_SL_1):
	noise_mat_1[index1,index1] = noise_mat_1[index1,index1] * (xes4_error_1[index1]**2.)

###Global d13C matern

s_matrix_1_tmp = numpy.repeat(numpy.reshape(xes5_type_2,(1,-1)),n_SL_1,axis=0)
s_matrix_2_tmp = numpy.repeat(numpy.reshape(xes5_type_2,(-1,1)),n_SL_1,axis=1)

w1 = numpy.where((s_matrix_1_tmp!=2.)&(s_matrix_2_tmp!=2.))

s_matrix_1 = numpy.zeros(numpy.shape(s_matrix_1_tmp))
s_matrix_1[w1[0],w1[1]] = s_matrix_1[w1[0],w1[1]] + 1.0

###NJ d13C bulk and d13C foram  and TEX86 matern

s_matrix_1_tmp = numpy.repeat(numpy.reshape(xes5_type_2,(1,-1)),n_SL_1,axis=0)
s_matrix_2_tmp = numpy.repeat(numpy.reshape(xes5_type_2,(-1,1)),n_SL_1,axis=1)

w1 = numpy.where((s_matrix_1_tmp==s_matrix_2_tmp)&((s_matrix_2_tmp!=5.)&(s_matrix_2_tmp!=2.)))

s_matrix_2 = numpy.zeros(numpy.shape(s_matrix_1_tmp))
s_matrix_2[w1[0],w1[1]] = s_matrix_2[w1[0],w1[1]] + 1.0

s_matrix_1_tmp = numpy.repeat(numpy.reshape(xes5_type_2,(1,-1)),n_SL_1,axis=0)
s_matrix_2_tmp = numpy.repeat(numpy.reshape(xes5_type_2,(-1,1)),n_SL_1,axis=1)

w1 = numpy.where((s_matrix_1_tmp==s_matrix_2_tmp)&((s_matrix_2_tmp==2.)))

s_matrix_3 = numpy.zeros(numpy.shape(s_matrix_1_tmp))
s_matrix_3[w1[0],w1[1]] = s_matrix_3[w1[0],w1[1]] + 1.0

###Local matern

s_matrix_1_tmp = numpy.repeat(numpy.reshape(xes5_type_2,(1,-1)),n_SL_1,axis=0)
s_matrix_2_tmp = numpy.repeat(numpy.reshape(xes5_type_2,(-1,1)),n_SL_1,axis=1)

w1_tmp = ((s_matrix_1_tmp==s_matrix_2_tmp)&(s_matrix_2_tmp!=2.))

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

t_matrix_1 = numpy.repeat(numpy.reshape(xes1_t_1,(1,-1)),n_SL_1,axis=0) - numpy.repeat(numpy.reshape(xes1_t_1,(-1,1)),n_SL_1,axis=1)	
new_dists_t_1 = numpy.sqrt((t_matrix_1/(1.))**2.)
new_mults_t_1 = numpy.repeat(numpy.reshape(xes1_t_1,(1,-1)),n_SL_1,axis=0) * numpy.repeat(numpy.reshape(xes1_t_1,(-1,1)),n_SL_1,axis=1)
	
def optimize_MLE_merge_guess(guess1):
	
	global noise_mat_1
	global y_1
	global y_transpose_1

	K1 = matern(guess1[0],3.,numpy.absolute(guess1[1]),new_dists_t_1) * s_matrix_1
    
	K2 = matern(guess1[2],3.,numpy.absolute(guess1[3]),new_dists_t_1) * s_matrix_2
	
	K3 = matern(guess1[4],3.,numpy.absolute(guess1[5]),new_dists_t_1) * s_matrix_3
	
	#K4 = matern(guess1[6],3.,numpy.absolute(guess1[7]),new_dists_t_1) * s_matrix_3
	
	K5 = linear1(guess1[6],guess1[7],new_mults_t_1,s_matrix_1)
	
	K6 = linear1(guess1[8],guess1[9],new_mults_t_1,s_matrix_2)
	
	K7 = linear1(guess1[10],guess1[11],new_mults_t_1,s_matrix_3)

	#K8 = matern(guess1[6],1.,numpy.absolute(guess1[7]),new_dists_t_1) * s_matrix_4
	
	#K9 = matern(guess1[8],1.,numpy.absolute(guess1[9]),new_dists_t_1) * s_matrix_5
	
	WN = WHE_NSE(guess1[12],guess1[13],guess1[14],guess1[15],t_matrix_1,noise_mat_1,s_matrix_9,s_matrix_10,s_matrix_11,s_matrix_12)
	
	K_1 = K1 + K2 + K3 + K6 + K7 + WN + K5  #+ K4 
	
	K_inv = numpy.linalg.inv(K_1)
	matmul_tmp = numpy.matmul(y_transpose_1,K_inv)
	term1 = (-1./2.) * numpy.matmul(matmul_tmp,y_1)
	det_k = numpy.linalg.slogdet(K_1)[1]
	term2 = (1./2.) * det_k
	term3 = (float(len(y_1))/2.) * numpy.log(2.*numpy.pi)
	opt_outs_CO2_1 = term1 - term2 - term3

	opt_outs_CO2 = opt_outs_CO2_1 * 1.

	if opt_outs_CO2 > 0:
		opt_outs_CO2 = numpy.nan
	
	opt_outs_CO2 = opt_outs_CO2 * -1.
	
	return opt_outs_CO2
	
MCMC_iters = 200001

guess_orig = numpy.array([\
1.,1.,\
1.,1.,\

1.,1.,\

0.1,0.1,\
0.1,0.1,\
0.1,0.1,\

0.001,0.001,0.01,0.001])

guess_file = wkspc + 'modes_K.csv'	
guess_orig_tmp = numpy.genfromtxt(guess_file,delimiter=",")
guess_orig[0:2] = guess_orig_tmp[0,0:2]
guess_orig[4:] = guess_orig_tmp[0,2:]

guess_orig_2 = numpy.array([\
0.,0.,\
1.,1.,\

0.,0.,\

0.0,0.0,\
0.0,0.0,\
0.0,0.0,\

0.01,0.01,0.1,0.01])

stepsizes = guess_orig_2/1.5 

old_alpha = guess_orig*1.0

output_matrix_A = numpy.zeros((MCMC_iters,len(guess_orig)))* numpy.nan

loglik_output = numpy.zeros((MCMC_iters,2))
accept_output = numpy.zeros((MCMC_iters,2)) * numpy.nan

file_out = wkspc + 'posterior_hyperparams' + str(sys.argv[1]) + '.csv'
file_out2 = wkspc + 'accept_array' + str(sys.argv[1]) + '.csv'

print ("run Medford 'regional'")

t1 = float(time.time())

index_to_change = 0	

positive_indices = numpy.array([0,1,2,3,4,5,7,9,11,12,13,14,15])

for n in range(MCMC_iters):

	if n % 100 == 0:
	
		if n > 0:
			t2 = float(time.time())
			
			time_per_iter = (t2 - t1) / (float(n))
			accept_output[n,1] = time_per_iter
			time_remaining = time_per_iter * (MCMC_iters - n)

			days = time_remaining // (24 * 3600)
			time_remaining = time_remaining % (24 * 3600)
			hours = time_remaining // 3600
			time_remaining %= 3600
			minutes = time_remaining // 60
			time_remaining %= 60
			seconds = time_remaining

			print(f"Regional - estimated time remaining: {int(days)} days, {int(hours)} hours, {int(minutes)} minutes, {int(seconds)} seconds")
			
			print ("...")
			print ("loglik: ", numpy.mean(loglik_output[n-1,0]))
			print ("accept rate: ", numpy.mean(accept_output[0:n,0]))
			print ("process time rate: ",time_per_iter)
			print ("###")
		
			numpy.savetxt(file_out,output_matrix_A,delimiter=',',fmt='%10.8e')
			numpy.savetxt(file_out2,accept_output,delimiter=',',fmt='%10.5f')	
				
		
	if n > 0:
		old_alpha  = output_matrix_A[n-1,:]
		old_loglik = loglik_output[n-1,0]
		
	new_alpha = old_alpha * 1.0

	new_alpha = numpy.random.normal(loc = old_alpha, scale = stepsizes)
	
	new_alpha[positive_indices] = numpy.absolute(new_alpha[positive_indices])

	index_to_change = index_to_change + 1
	
	if index_to_change == len(new_alpha):
		index_to_change = 0	

	new_loglik = optimize_MLE_merge_guess(new_alpha)
	
	if n == 0:
		old_loglik = new_loglik * .9999999999999
		
	if numpy.isnan(new_loglik) == False:
		if (new_loglik < old_loglik):
			output_matrix_A[n,:] = new_alpha
			loglik_output[n,0] = new_loglik
			accept_output[n,0] = 1.0

		else:			
			u = numpy.random.uniform(0.0,1.0)
			
			if (u < numpy.exp(old_loglik - new_loglik)):
				output_matrix_A[n,:] = new_alpha				
				loglik_output[n,0] = new_loglik
				accept_output[n,0] = 1.0

			else:
				output_matrix_A[n,:] = old_alpha
				loglik_output[n,0] = old_loglik
				accept_output[n,0] = 0.0

	else:
		output_matrix_A[n,:] = old_alpha
		loglik_output[n,0] = old_loglik
		accept_output[n,0] = 0.0

	

numpy.savetxt(file_out,output_matrix_A,delimiter=',',fmt='%10.8e')
numpy.savetxt(file_out2,accept_output,delimiter=',',fmt='%10.5f')