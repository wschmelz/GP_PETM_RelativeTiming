import math
import csv
import os
import sys
import numpy
import scipy
import glob
from scipy.spatial import Delaunay
from scipy import signal
from scipy.interpolate import interp1d

import time
from datetime import tzinfo, timedelta, datetime

from numpy import matrix
from numpy import linalg
from numpy import genfromtxt

import pylab as pl
import matplotlib.pyplot as plt
from matplotlib import collections  as mc
from matplotlib import transforms
from matplotlib.colors import colorConverter



backslash = '\\'


wkspc = str(os.getcwd()).replace(backslash,"/") + "/"


##data files

shore_displacements = wkspc + "00_000_00_00_BR_d13C_bulk.csv"
shore_displacements = numpy.genfromtxt(shore_displacements, delimiter=',')

shore_displacements_2 = numpy.zeros((len(shore_displacements[:,0]),2))

shore_displacements_2[:,0] = shore_displacements[:,0]*1.0
shore_displacements_2[:,1] = shore_displacements[:,1]*1.0


#shore_disp_vec = numpy.nanmean(shore_displacements_2[:,1],axis=1)

fig = plt.figure()
ax1 = plt.subplot(211)

ax1.plot(shore_displacements_2[:,1],shore_displacements_2[:,0],'k',linewidth=1)

ax1.grid()
	
	
	
a = (shore_displacements_2[:,1] - numpy.mean(shore_displacements_2[:,1])) / (numpy.std(shore_displacements_2[:,1]) * len(shore_displacements_2[:,1]))
v = (shore_displacements_2[:,1] - numpy.mean(shore_displacements_2[:,1])) /  numpy.std(shore_displacements_2[:,1])
r = numpy.correlate (a,v,'full')

w1 = numpy.where(r==numpy.max(r))[0]
print (w1)
r2 = r[int(w1):]

print (r2)

w2 = numpy.where(r2>0.0)[0]

integral = numpy.trapz(r2[w2]*.05)
print (integral)
ax1 = plt.subplot(212)
ax1.plot(r2,'k',linewidth=1)
ax1.grid()
go = 0.0
for n in range(0,len(r2)):
	if (r2[n] > 0.0) and go==0.0:
		ax1.plot([n,n],[0,r2[n]],'k',linewidth=1,alpha=0.05)
	if r2[n] < 0.0:
		go =1.0
'''
ax1 = plt.subplot(313)
ax1.plot(integral,'k',linewidth=1)
ax1.grid()
'''

N_prime = (len(shore_displacements_2[:,1])) / integral

print (N_prime)
	
#print (shore_disp_vec)

plt.show()

