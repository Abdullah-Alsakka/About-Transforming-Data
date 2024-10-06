#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 17:41:24 2022

@author: mike

This curve_fit regression routine of Python scipy, uses the data, as mag vs redshift (z), from Brout et al. 2022, 
'The Pantheon+ Analysis: Cosmological Constraints' Astrophys. J. vol. 938, 110 AND the TRGB data, from GS Anand et al. 
2022, 'Comparing Tip of the Red Giant Branch Distance Scales:' Astrophys. J. vol. 932, 15. The model selected is the 
Melia 2012 solution, not formally the Freidmann-Lemaitre-Robertson-Walker (FLRW) model. This model presents 
only one parameter, the Hubble constant. No estimation is possible for either the normalized matter density, which is 
presumed to be about 1, nor dark energy.")
"""
print()
print("This is the R_h modeled using distance mag data as ")
print("luminosity distances, D_L, vs. the expansion factor.")
print("Estimates for the matter density and dark energy ")
print("(cosmological constant) cannot be made with this model")

# import the data and library files
import numpy as np
import csv
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import math
from sklearn.metrics import r2_score
from astropy.stats.info_theory import bayesian_info_criterion

# open data file and extract the data
with open("Freed_TRGB_SNe_mag_data.csv","r") as i:
    rawdata = list(csv.reader(i, delimiter = ","))
    
# ignore the top row    
exampledata = np.array(rawdata[1:],dtype=float)

xdata = exampledata[:,0]
ydata = exampledata[:,1]
error = exampledata[:,2]

# define the function, where Hu is the Hubble constant and the only parameter
def func(x,Hu):
    return (litesped*(1+x)/Hu)*np.log((1+x))

def func2(x,Hu):
    return 5*np.log10(func(x,Hu)) + 25

# define the constant
litesped = 299793

# The intial guess for the Hubble constant
p0 = [60]

# evaluate the function where bnds are the upper and lower limit allowed for Hu
funcdata = func2(xdata,p0)
bnds = (60.0, 80.0)

# curve fit nodel to the data where absolute_sigma = False means the standard deviations are normlaized
params, pcov = curve_fit(func2,xdata,ydata,bounds = bnds, sigma = error, absolute_sigma = False)
perr = np.sqrt(np.diag(pcov))

# unpacking the Hubble parameter and the estimated fit error
Hubble, = params
Error, = perr

# rounding the above two values to 2 decimal places
normHubble = np.round(Hubble,2)
normError = np.round(Error,2)

# calculate the statistical fitness, using N=156 as the number of data pairs and P=1 as the degree of freedom (parameter count)
P=1
N=1719
e = 2.718281

#Calculate the chi^2 according to astronomers
newxsqrd = sum(((ydata - func2(xdata,Hubble))**2)/error**2)
newxsqrded = np.round(newxsqrd/(N-P),2)
"""
#Calculate the chi^2 according in the common manner
normxsqrd = sum(((ydata - func2(xdata,Hubble))**2)/func2(xdata,Hubble))
normxsqrded = np.round(normxsqrd/(N-P),6) #rounded to 6 digits
"""
#The usual method for BIC calculation is
SSE = sum((ydata - func(xdata,Hubble))**2)
log_SSE = math.log(e,SSE)
small_bic = bayesian_info_criterion(log_SSE, P, N)
rBIC = round(small_bic,2)

# calculation of residuals
residuals = ydata - func2(xdata,Hubble)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((ydata-np.mean(ydata))**2)

#easy routine for calculating r squared
ycalc = func2(xdata,Hubble)
R_sqrd = r2_score(ydata, ycalc)
R_square = round(R_sqrd,4)

#Calculation of the weighted F-statistic
SSEw = sum((1/error)*(residuals)**2)
SSM = sum((1/error)*(ydata - np.mean(ydata))**2) 
MSR = (SSM - SSEw)/(P)
MSE = SSEw/(N-P)
Fstat = MSR/MSE
rFstat = round(Fstat,1)

# r squared calculation
r_squared = 1 - (ss_res/ss_tot)
r2 = round(r_squared,3)
r2adjusted = np.round(1-(((1-r2)*(len(ydata)-1))/(len(ydata)-P-1)),3)

# plot of imported data
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['lines.linewidth'] = 3
plt.xlim(0.0,1.8)
plt.ylim(32.0, 46.0)
plt.xscale("linear")
plt.yscale("linear")
fig, ax = plt.subplots()
ax.tick_params(axis="y", direction='in', length=14)
ax.tick_params(axis="x", direction='in', length=10)
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(3)
    ax.tick_params(width=3)
plt.xlabel("Redshift z", fontsize=18)
plt.ylabel("$\mu$ (distance mag, no units)", fontsize=18)
plt.rc('xtick', labelsize=14) 
plt.rc('ytick', labelsize=14)
plt.errorbar(xdata, ydata, yerr=error, fmt='.k', capsize = 5)
plt.plot(xdata, funcdata, color = "green", label = "$magR_h$ model")
plt.legend(loc='best', fancybox=True, shadow=False)
plt.show()

#print results
print("\n")
print("The estimated Hubble constant is:", normHubble)
print("The S.D. of the Hubble constant is", normError)
print()
print('The r\u00b2 is:', R_square)
print('The weighed Fstat is:', rFstat)
print("And reduced goodness of fit, according to astronomers, \u03C7\u00b2 estimate:", newxsqrded)
#print("And common reduced goodness of fit \u03C7\u00b2 estimate:", normxsqrded)
print("The BIC estimate is:",rBIC)
print()

#Routines to save figues in eps and pdf formats
fig.savefig("magR_h.eps", format="eps", dpi=2000, bbox_inches="tight", transparent=True)
fig.savefig("magR_h.pdf", format="pdf", dpi=2000, bbox_inches="tight", transparent=True)
