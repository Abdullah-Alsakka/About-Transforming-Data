#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 17:41:24 2022
This model is described in section 2.5.2 and Appendix B1.
@author: Mike
"""
print("This curve_fit regression routine of Python scipy, uses the 'distance mag' data, converted to luminosity distances, D_L, vs expansion factor, from Brout et al. 2022, 'The Pantheon+ Analysis: Cosmological Constraints' Astrophys. J. vol. 938, 110. The model selected is the Melia 2012, analytical solution with only one parameter, the Hubble constant. No estimation is possible for the matter density nor dark energy")
print()
# import data and library files
import numpy as np
import csv
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import math

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
e = 2.718281

# define the function, where b is the Hubble constant
def func(x,b):
    return (litesped/(b*x))*np.log(1/x)

# evaluate and plot function
funcdata = func(xdata,70) # The initial guess for the Hubble constant, 70, is the value just after xdata.

# the lower and upper bounds allowed for the Hubble constant
bnds = (50.0, 80.0)

# curve_fit the model to the data, note that when absolute_sigma = False the errors are "normalized"
params, pcov = curve_fit(func,xdata,ydata,bounds = bnds, sigma = error, absolute_sigma = False)
perr = np.sqrt(np.diag(pcov))
      
# unpacking the Hubble parameter and the estimated standard error
Hubble, = params
Error, = perr

# rounding the above two values to 2 decimal places
normHubble = round(Hubble,2)
normError = round(Error,2)

# calculate the statistical fitness, using N=1702 as the number of data pairs and P=1 as the degree of freedom (paramater count)

#set some constant values
P=1
N=1702

#Calculate the chi^2 according to astronomers
newxsqrd = sum(((ydata[1:-1] - func(xdata,Hubble)[1:-1])**2)/(error[1:-1]**2))
#normalised newxsqrded is calculated as
newxsqrded = round(newxsqrd/(N-P),3)

# since the error at the origin is 0 we have to ignore this only to estimate the common goodness of fit, but not the fit itself
chisq = sum((ydata[1:-1] - func(xdata,Hubble)[1:-1])**2/func(xdata,Hubble)[1:-1])
#normalised chisquar is calculated as
normchisquar = round((chisq/(N-P)),2) #rounded to 3 digits

#another BIC value is calculated as 
SSE = sum((ydata - func(xdata,Hubble))**2)
alt_BIC = N*math.log(e,SSE/N) + P*math.log(e,N)
normBIC = round(alt_BIC,2)

#calculation of residuals
residuals = ydata - func(xdata,Hubble)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((ydata-np.mean(ydata))**2)

#r squared calculation and round to 4 digits
r_squared = 1 - (ss_res/ss_tot)
r2 = round(r_squared,4)
r2adjusted = round(1-(((1-r2)*(len(ydata)-1))/(len(ydata)-P-1)),4)

#plot of data and results
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['lines.linewidth'] = 3
plt.xlim(0.3,1.0)
plt.ylim(0.0,15000)
plt.xscale("linear")
plt.yscale("linear")
fig, ax = plt.subplots()
ax.tick_params(axis="y", direction='in', length=8)
ax.tick_params(axis="x", direction='in', length=10)
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(3)
    ax.tick_params(width=3)
plt.xlabel("Expansion factor, \u03BE ", fontsize=18)
plt.ylabel("Luminosity distance (Mpc)", fontsize=18)
plt.rc('xtick', labelsize=14) 
plt.rc('ytick', labelsize=14)
plt.errorbar(xdata, ydata, yerr=error, fmt='.k', capsize = 5)
plt.plot(xdata, funcdata, color = "orange", label = "$D_LR_h$ model")
plt.legend(loc='best', fancybox=True, shadow=False)
plt.show()

#print results
print()
print("The estimated Hubble constant is: ", normHubble)
print("The S.D. of the Hubble constant is ", normError)
print()
print("The adjusted r\u00b2 is calculated to be: ",r2adjusted)
print("And reduced goodness of fit, according to astronomers, \u03C7\u00b2 estimate is: ", newxsqrded)
print("And the common reduced goodness of fit \u03C7\u00b2 estimate is: ", normchisquar)
print("The estimate for BIC is: ", normBIC)

#Routines to save figues in eps and pdf formats
fig.savefig("Melia_D_L_data.eps", format="eps", dpi=2000, bbox_inches="tight", transparent=True)
fig.savefig("Melia_D_L_data.pdf", format="pdf", dpi=2000, bbox_inches="tight", transparent=True)
