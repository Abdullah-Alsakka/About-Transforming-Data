#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 17:41:24 2022
This model is described in section 2.5.2, euqation 8 and Appendix B1.
@author: mike
"""
print()
print("This curve_fit regression routine of Python scipy, uses the data, as mag vs redshift (z), from Brout et al. 2022, 'The Pantheon+ Analysis: Cosmological Constraints' Astrophys. J. vol. 938, 110. The model selected is the Melia 2012 solution, not the Freidmann-Lemaitre-Robertson-Walker (FLRW) model. This model presents only one parameter, the Hubble constant. No estimation is possible for either the normalized matter density, which is presumed to be about 1, nor dark energy.")
print()

# import the data and library files
import numpy as np
import csv
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import math

# open data file and extract the data
with open("DATA.csv","r") as i:
    rawdata = list(csv.reader(i, delimiter = ","))
    
# ignore the top row    
exampledata = np.array(rawdata[1:],dtype=float)

xdata = exampledata[:,0]
ydata = exampledata[:,2]
error = exampledata[:,3]

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
bnds = (50.0, 80.0)

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
N=1701
e = 2.718281

#Calculate the chi^2 according to astronomers
newxsqrd = sum(((ydata - func2(xdata,Hubble))**2)/error**2)
newxsqrded = np.round(newxsqrd/(N-P),2)

#Calculate the chi^2 according in the common manner
normxsqrd = sum(((ydata - func2(xdata,Hubble))**2)/func2(xdata,Hubble))
normxsqrded = np.round(normxsqrd/(N-P),6)

#another BIC value is calculated as 
SSE = sum((ydata - func2(xdata,Hubble))**2)
alt_BIC = N*math.log(e,SSE/N) + P*math.log(e,N)
normalt_BIC = round(alt_BIC,2)

# calculation of residuals
residuals = ydata - func2(xdata,Hubble)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((ydata-np.mean(ydata))**2)

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
plt.ylabel("$\mu$ (mag)", fontsize=18)
plt.rc('xtick', labelsize=14) 
plt.rc('ytick', labelsize=14)
plt.errorbar(xdata, ydata, yerr=error, fmt='.k', capsize = 5)
plt.plot(xdata, funcdata, color = "green", label = "$magR_h$ model")
plt.legend(loc='best', fancybox=True, shadow=False)
plt.show()

#print results
print("\n")
print("The estimated Hubble constant is: ", normHubble)
print("The S.D. of the Hubble constant is ", normError)
print()
print("The adjusted r\u00b2 is calculated to be: ",r2adjusted)
print("And reduced goodness of fit, according to astronomers, \u03C7\u00b2 estimate: ", newxsqrded)
print("And common reduced goodness of fit \u03C7\u00b2 estimate: ", normxsqrded)
print("The estimate for BIC is: ", normalt_BIC)
print()

#Routines to save figues in eps and pdf formats
fig.savefig("Rh_mag_data.eps", format="eps", dpi=2000, bbox_inches="tight", transparent=True)
fig.savefig("Rh_mag_data.pdf", format="pdf", dpi=2000, bbox_inches="tight", transparent=True)
