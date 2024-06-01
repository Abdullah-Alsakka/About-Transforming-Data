#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 17:41:24 2022

@author: mike

This curve_fit regression routine of Python scipy, uses the data, as luminosity distance vs. 
recession velocity (1/1+z), from Brout et al. 2022, 'The Pantheon+ Analysis: Cosmological Constraints'
Astrophys. J. vol. 938, 110. The model selected is the E-DS (from Oeztas, Smith, Paul, Int J Theor Phys 2008). 
This model presents only one parameter, the Hubble constant. No estimation is possible for either 
the normalized matter density, nor dark energy nor space density.
"""
print()
print("This is our E-DS model, a version of the Einstein-deSitter model of cosmology. It is not possible" )
print("to estimate the matter density, space density nor dark energy.")

import numpy as np
import csv
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import math
from sklearn.metrics import r2_score
from astropy.stats.info_theory import bayesian_info_criterion

# open data file
with open("DATA2B.csv","r") as i:
    rawdata = list(csv.reader(i, delimiter = ","))
    
# ignore the first row
exampledata = np.array(rawdata[1:],dtype=float)

xdata = exampledata[:,1]
ydata = exampledata[:,4]
error = exampledata[:,7]

# define the constant
litesped = 299793

# define the function, where Hu is the Hubble constant, the only parameter
def func(x,Hu):
    return ((litesped/(Hu*x))*np.sinh(1-x))

# The intial guess for the Hubble constant
p0 = [65]

# evaluate the function where bnds are the upper and lower limit allowed for Hu
funcdata = func(xdata,p0)
bnds = (50.0, 80.0)

# curve fit model to the data where absolute_sigma = False means the standard deviations are normalized
params, pcov = curve_fit(func,xdata,ydata,bounds = bnds,sigma = error, absolute_sigma = False)
perr = np.sqrt(np.diag(pcov))

# unpacking the Hubble parameter and the estimated fit error
Hubble, = params
Error, = perr

# rounding the above two values to 2 decimal places
normHub = np.round(Hubble,2)
normError = np.round(Error,2)

#evaluate the function using the best fit parameter
funcdata2 = func(xdata,Hubble)

# calculate the statistical fitness, using N=1702 as the number of data pairs and P=1 as the parameter count
P=1
N=1702
e = 2.718281

#Calculate the chi^2 according to astronomers
newxsqrd = sum(((ydata[1:-1] - func(xdata,Hubble)[1:-1])**2)/(error[1:-1]**2))
newxsqrded = round(newxsqrd/(N-P),2)

# estimating the goodness of fit in the more common manner
chisq = sum((ydata[1:-1] - func(xdata,Hubble)[1:-1])**2/func(xdata,Hubble)[1:-1])
normchisquar = round((chisq/(N-P)),2) #rounded to 2 digits

#The usual method for BIC calculation is
SSE = sum((ydata - func(xdata,Hubble))**2)
log_SSE = math.log(e,SSE)
small_bic = bayesian_info_criterion(log_SSE, P, N)
rBIC = round(small_bic,2)

# calculation of residuals
residuals = ydata - func(xdata,Hubble)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((ydata-np.mean(ydata))**2)

#routine for calculating r squared
ycalc = func(xdata,Hubble)
R_sqrd = r2_score(ydata, ycalc)
R_square = round(R_sqrd,4)

#Calculation of the weighted F-statistic
SSEw = sum((1/error)*(residuals)**2)
SSM = sum((1/error)*(ydata - np.mean(ydata))**2) 
MSR = (SSM - SSEw)/(P)
MSE = SSEw/(N-P)
Fstat = MSR/MSE
rFstat = round(Fstat,1)

# plot of imported data
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['lines.linewidth'] = 3
plt.xlim(0.30,1.0)
plt.ylim(0.0, 20000.0)
plt.xscale("linear")
plt.yscale("linear")
fig, ax = plt.subplots()
ax.tick_params(axis="y", direction='in', length=14)
ax.tick_params(axis="x", direction='in', length=10)
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(3)
    ax.tick_params(width=3)
plt.xlabel("Recession velocity, \u03BE", fontsize=18)
plt.ylabel("Luminosity distance (Mpc)", fontsize=18)
plt.rc('xtick', labelsize=14) 
plt.rc('ytick', labelsize=14)
plt.errorbar(xdata, ydata, yerr=error, fmt='.k', capsize = 5)
plt.plot(xdata, funcdata2, color = "green", label = "$D_L$E-DS model")
plt.legend(loc='best', fancybox=True, shadow=False)
plt.show()

#print results
print("\n")
print("The estimated Hubble constant is: ", normHub)
print("The S.D. of the Hubble constant is ", normError)
print("Note that values for \u03A9_m and \u03A9_k and \u03A9_\u039B cannot be calculated with this model.")
print()
print('The r\u00b2 is:', R_square)
print('The weighted F-statistic is:', rFstat)
print("The reduced goodness of fit, according to astronomers, \u03C7\u00b2 estimate: ", newxsqrded)
print("The common reduced goodness of fit \u03C7\u00b2 estimate: ", normchisquar)
print("The BIC estimate is: ",rBIC)
print()

#Routines to save figues in eps and pdf formats
fig.savefig("DL-E-DS.eps", format="eps", dpi=2000, bbox_inches="tight", transparent=True)
fig.savefig("DL-E-DS.pdf", format="pdf", dpi=2000, bbox_inches="tight", transparent=True)
