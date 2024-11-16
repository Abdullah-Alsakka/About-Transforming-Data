#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15, 09:41:24 2024

@author: mike

This curve_fit regression routine of Python scipy, uses the 18 data, 
converted to luminosity distances, D_L, vs. z, from Freedman et al. 2019, 
'The Carnegie-Chicago Hubble Program. VIII. An Independent Determination 
of the Hubble Constant Based on the Tip of the Red Giant Branch' 
Astrophys. J. vol. 938, 110. The model selected is the simplest solution 
with two parameters, the Hubble constant, Hu, and the intercept. 
No estimate of matter or dark energy is possible.
"""
print()
print("This is our Hubble model, a version of cosmology. It is not possible" )
print("to estimate the matter density, space density nor dark energy.")

import numpy as np
import csv
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import math
from sklearn.metrics import r2_score
from astropy.stats.info_theory import bayesian_info_criterion

#open data file
#with open("TRGB_data_Freedman_2019.csv","r") as i:
#    rawdata = list(csv.reader(i, delimiter = ","))
    
# or open another data file as
with open("TRGB_data_Freedman_2019.csv","r") as i:
    rawdata = list(csv.reader(i, delimiter = ","))
    
# ignore the first row
exampledata = np.array(rawdata[1:],dtype=float)

xdata = exampledata[:,8]
ydata = exampledata[:,4]
error = exampledata[:,7]

# define the constant (which is not used here)
litesped = 299792

# define the function, where Hu is the Hubble constant, the only parameter
def func(x,Hu,intercept):
    return (x/Hu)+intercept

# The intial guess for the Hubble constant and intercept
p0 = [65.0,0.01]

# evaluate the function where bnds are the upper and lower limit allowed for Hu and the intercept
bnds = ([60.0,0.000001],[95.0,1.0])

# curve fit model to the data where absolute_sigma = False means the standard deviations are normalized
params, pcov = curve_fit(func,xdata,ydata,p0,bounds = bnds,sigma = error, absolute_sigma = False)
perr = np.sqrt(np.diag(pcov))

# unpacking the Hubble parameter and the estimated fit error
Hubble,intercept = params
Error,intererror = perr

#funcdata = func(xdata,p0)

# rounding the above above values to 2 decimal places
normHub = np.round(Hubble,2)
normError = np.round(Error,2)
normInter = np.round(intercept,8)
normInterError = np.round(intererror,5)

#evaluate the function using the best fit parameter
funcdata2 = func(xdata,Hubble,intercept)

# calculate the statistical fitness, using N=19 as the number of data pairs 
# plus the intercept and P=1 as the parameter count
P=1
N=19
e = 2.718281

#Calculate the chi^2 according to astronomers
newxsqrd = sum(((ydata[1:-1] - func(xdata,Hubble,intercept)[1:-1])**2)/(error[1:-1]**2))
newxsqrded = round(newxsqrd/(N-P),2)

# estimating the goodness of fit in the more common manner
chisq = sum((ydata[1:-1] - func(xdata,Hubble,intercept)[1:-1])**2/func(xdata,Hubble,intercept)[1:-1])
normchisquar = round((chisq/(N-P)),2) #rounded to 2 digits

#The usual method for BIC calculation is
SSE = sum((ydata - func(xdata,Hubble,intercept))**2)
log_SSE = math.log(e,SSE)
small_bic = bayesian_info_criterion(log_SSE, P, N)
rBIC = round(small_bic,2)

# calculation of residuals
residuals = ydata - func(xdata,Hubble,intercept)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((ydata-np.mean(ydata))**2)

#routine for calculating r squared
ycalc = func(xdata,Hubble,intercept)
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
plt.xscale("linear")
plt.yscale("linear")
fig, ax = plt.subplots()
ax.tick_params(axis="y", direction='in', length=14)
ax.tick_params(axis="x", direction='in', length=10)
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(3)
    ax.tick_params(width=3)
plt.xlabel("Recession velocity (cz, km/sec)", fontsize=18)
plt.ylabel("Luminosity distance (Mpc)", fontsize=18)
plt.rc('xtick', labelsize=14) 
plt.rc('ytick', labelsize=14)
plt.errorbar(xdata, ydata, yerr=error, fmt='.k', capsize = 5)
plt.plot(xdata, funcdata2, color = "green", label = "Hubble correlation")
plt.legend(loc='best', fancybox=True, shadow=False)
plt.show()

#print results
print("\n")
print("The estimated Hubble constant is: ", normHub)
print("The S.D. of the Hubble constant is ", normError)
print("The estimated intercept is: ", normInter)
print("The S.D. of the intercept is ", normInterError)
print("Note that values for \u03A9_m and \u03A9_k and \u03A9_\u039B cannot be calculated with this model.")
print()
print('The r\u00b2 is:', R_square)
print('The weighted F-statistic is:', rFstat)
print("The reduced goodness of fit, according to astronomers, \u03C7\u00b2 estimate: ", newxsqrded)
#print("The common reduced goodness of fit \u03C7\u00b2 estimate: ", normchisquar)
print("The BIC estimate is: ",rBIC)
print()

#Routines to save figues in eps and pdf formats
fig.savefig("HubbleFreed.eps", format="eps", dpi=2000, bbox_inches="tight", transparent=True)
fig.savefig("HubbleFreed.pdf", format="pdf", dpi=2000, bbox_inches="tight", transparent=True)
