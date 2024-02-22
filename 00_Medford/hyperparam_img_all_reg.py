import os
import sys

import csv
import numpy
import loess2D

import matplotlib.pyplot as plt

backslash = '\\'

wkspc = str(os.getcwd()).replace(backslash,"/") + "/"

for index_Kdat in range(1,2):

	hyperparams_tmp = wkspc + "posterior_hyperparams" + str(index_Kdat) + ".csv"
	hyperparams_tmp = numpy.genfromtxt(hyperparams_tmp,delimiter=',')
	if index_Kdat == 1:
		K_data = hyperparams_tmp[0:,:]*1.0

hyperparams = K_data*1.0

min_iter = 100000

fig = plt.figure(1,figsize=(40,20))
for n in range(0,len(hyperparams[0,:])):

	hyp_3 = numpy.delete(hyperparams,numpy.where(numpy.isnan(hyperparams))[0],axis=0)
	ax1 = plt.subplot(5,5,n+1)
	ax1.plot(hyp_3[min_iter:,n])

pltname = wkspc +  'K_path_reg.png'

plt.tight_layout()

plt.savefig(pltname, dpi = 300)

fig = plt.figure(2,figsize=(40,20))

percs = numpy.zeros((3,len(hyperparams[0,:])))
modes2 = numpy.zeros((2,len(hyperparams[0,:])))

for n in range(0,len(hyperparams[0,:])):
	
	ax1 = plt.subplot(5,5,n+1)

	histo1 = numpy.histogram(hyp_3[min_iter:,n],bins=50)

	bin_places = (histo1[1][1:] + histo1[1][0:-1])/2.

	bin_spacing = bin_places[1] - bin_places[0]

	bin_fill_high = numpy.arange(0.0,bin_spacing*100.,bin_spacing) + numpy.max(bin_places) + bin_spacing
	bin_fill_low = numpy.arange(0.0,bin_spacing*100.,bin_spacing) - (bin_spacing*100.) +  numpy.min(bin_places)
	bin_places_tmp = numpy.append(bin_fill_low,bin_places)
	bin_places_tmp = numpy.append(bin_places_tmp,bin_fill_high)

	bin_fill_low_2 = numpy.zeros(len(bin_fill_low))
	bin_fill_high_2 = numpy.zeros(len(bin_fill_low))	
	bin_values_tmp = numpy.append(bin_fill_low_2,histo1[0])
	bin_values_tmp = numpy.append(bin_values_tmp,bin_fill_high_2)

	tmp_tmp = loess2D.loess_int(bin_places_tmp,bin_places_tmp * 0.0,bin_places_tmp,bin_places_tmp*0.0,bin_values_tmp,3,5.*bin_spacing,50.*bin_spacing)[2]

	zeros = numpy.where(tmp_tmp < 0.0)[0]

	tmp_tmp[zeros] = 0.0

	p1_tmp = numpy.max(tmp_tmp)
	w1 = numpy.where(tmp_tmp == p1_tmp)[0]
	p1 = bin_places_tmp[w1]	

	percs[0,n] = p1
	
	p16, p50, p84 = numpy.percentile(hyp_3[min_iter:,n], [16, 50, 84])
	p0, p100 = numpy.percentile(hyp_3[min_iter:,n], [0, 100])
	percs[0,n] = p50
	percs[1, n] = p84 - p1
	percs[2, n] = p1 - p16

	std2 = (percs[2,n] + percs[1,n])/2.
	dist_2 = numpy.random.normal(loc = p1, scale = std2,size=len(hyp_3[min_iter:,n]))

	p1_tmp = numpy.max(tmp_tmp)
	w1 = numpy.where(tmp_tmp == p1_tmp)[0]
	p1 = bin_places_tmp[w1]
	p2 = numpy.std(hyp_3[min_iter:,n])
	ax1.hist(hyp_3[min_iter:,n],bins=50)

	p3 =  numpy.max(numpy.histogram(hyp_3[min_iter:,n],bins=50)[0])

	ax1.plot([p1,p1],[0,p3],'k',linewidth=2.0)
	ax1.plot(bin_places_tmp,tmp_tmp,'k')

	mean1 = numpy.mean(hyp_3[min_iter:,n])
	std1 = numpy.std(hyp_3[min_iter:,n])

	ax1.plot([p1,p1],[0,p3],'b',linewidth=2.0)
	ax1.plot([p1+percs[1, n],p1+percs[1, n]],[0,p3],color='y',linewidth=1.5)
	ax1.plot([p1-percs[2, n],p1-percs[2, n]],[0,p3],color='y',linewidth=1.5)

	modes2[0,n] = mean1
	modes2[1,n] = std1
	ax1.plot([mean1,mean1],[0,p3],'r',linewidth=1.5)
	ax1.plot([mean1+std1,mean1+std1],[0,p3],'r',linewidth=1.5)
	ax1.plot([mean1-std1,mean1-std1],[0,p3],'r',linewidth=1.5)

	ax1.set_ylabel("frequency")

	ax1.set_xlim(p0,p100)

	percs[1, n] = p84
	percs[2, n] = p16

f_name1 = wkspc + 'modes_K_reg.csv'	

with open(f_name1, 'w',newline="") as csvfile2:
	writer = csv.writer(csvfile2)
	writer.writerows(percs)	

f_name1 = wkspc + 'modes_K_prior_reg.csv'	

with open(f_name1, 'w',newline="") as csvfile2:
	writer = csv.writer(csvfile2)
	writer.writerows(modes2)	

ax1.set_xlabel("value")	

pltname = wkspc +  'K_distribution_reg.png'

plt.tight_layout()

plt.savefig(pltname, dpi = 300)