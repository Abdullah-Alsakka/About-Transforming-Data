#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 17:41:24 2022
@author: Mike

This curve_fit regression routine of Python scipy, uses the distance mag data, converted to 
luminosity distances, D_L, vs expansion factor, from Brout et al. 2022, 'The Pantheon+ Analysis: 
Cosmological Constraints' Astrophys. J. vol. 938, 110 AND the TRGB data, from GS Anand et al. 2022, 'Comparing Tip of the Red Giant 
Branch Distance Scales:' Astrophys. J. vol. 932, 15. The model selected here is a version of the 
Einstein-DeSitter solution with two parameters, the Hubble constant and the normalised space density. 
No estimate of matter density is possible.
"""
# import data and library files
import numpy as np
import csv
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import math
from sklearn.metrics import r2_score
from astropy.stats.info_theory import bayesian_info_criterion

print("This is our \u039BE-DS model, a version of the Einstein-DeSitter model of cosmology.")
print("This version presumes some elliptically curved space geometry.")

# open data file selecting the distance, distance standard deviation and recession velocity columns
with open("Freed_TRGB_DL_data.csv","r") as i:
    rawdata = list(csv.reader(i, delimiter = ","))
    
# ignoring the first row
exampledata = np.array(rawdata[1:],dtype=float)

xdata = exampledata[:,0]
ydata = exampledata[:,1]
error = exampledata[:,2]

# Define the function - the model to be examined, where x represents the independent variable
# and b (Hubble constant) and c (normalized spacetime) are the parameters to be estimated.
def func(x,b,c):
    return ((litesped/(b*x*np.sqrt(c)))*np.sinh(np.arctanh(np.sqrt(c)/(np.sqrt((1-c)*x**2)+c))-np.arctanh(np.sqrt(c))))

#The initial guesses of the model parameters
p0=[70.0,0.9]

# define the lightspeed constant
litesped = 299793

# curve fit model to the data, where the first pair are the lower bounds and the second pair the upper bounds.
bnds = ([60.0, 0.0001],[80.0,1.0])

# when absolute_sigma = False, the regression routine "normalizes" the standard deviations (errors)
params, pcov = curve_fit(func,xdata,ydata, p0, bounds = bnds, sigma = error, absolute_sigma = False)

# extracting the Hubble constant, normalized space density and rounding the values
ans_b, ans_c = params
rans_b = round(ans_b,2)
rans_c = round(ans_c,5)
ans_L = 1-ans_c
rans_L= round(ans_L,5)

# evaluate and plot the function
funcdata = func(xdata,ans_b,ans_c)

# extracting the S.D. of both above values
perr = np.sqrt(np.diag(pcov))
ans_b_SD, ans_c_SD = perr
rans_b_SD = round(ans_b_SD,2)
rans_c_SD = round(ans_c_SD,5)

# normalised chisquared where P is the number of parameters (2), N the number of data pairs and 
# normchisquar is calculated from 
P=2
N=1720
e = 2.718281

#Calculate the chi^2 according to astronomers
newxsqrd = sum(((ydata[1:-1] - func(xdata,ans_b,ans_c)[1:-1])**2)/(error[1:-1]**2))
newxsqrded = round(newxsqrd/(N-P),2)

#The usual method for BIC calculation is
SSE = sum((ydata - func(xdata,ans_b,ans_c))**2)
log_SSE = math.log(e,SSE)
small_bic = bayesian_info_criterion(log_SSE, P, N)
rBIC = round(small_bic,2)

# calculation of residuals
residuals = ydata - func(xdata,ans_b,ans_c)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((ydata-np.mean(ydata))**2)

#routine for calculating r squared
ycalc = func(xdata,ans_b,ans_c)
R_sqrd = r2_score(ydata, ycalc)
R_square = round(R_sqrd,4)

#Calculation of the weighted F-statistic
SSEw = sum((1/error)*(residuals)**2)
SSM = sum((1/error)*(ydata - np.mean(ydata))**2) 
MSR = (SSM - SSEw)/(P)
MSE = SSEw/(N-P)
Fstat = MSR/MSE
rFstat = round(Fstat,1)

# plot of imported data with best fit curve
plt.figure(1,dpi=240)
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['lines.linewidth'] = 3
plt.xlim(0.3,1.0)
plt.ylim(0,30000)
plt.xscale("linear")
plt.yscale("linear")
fig, ax = plt.subplots()
ax.tick_params(axis="y", direction='in', length=14)
ax.tick_params(axis="x", direction='in', length=10)
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(3)
    ax.tick_params(width=3)
plt.errorbar(xdata, ydata, yerr=error, fmt='.k', capsize = 4)
plt.xlabel("Expansion factor, \u03BE", fontsize = 18)
plt.ylabel("Luminosity distance (Mpc)", fontsize=18)
plt.plot(xdata, funcdata, color = "orange", label = "$D_L$\u039BE-DS Model")
plt.legend(loc='best', fancybox=True, shadow=False)
plt.show()

# print results
print()
print("The calculated Hubble constant with standard deviation are:", rans_b,",", rans_b_SD )
print("The calculated normalized spacetime density,\u03A9k, with standard deviation are:", rans_c,",", rans_c_SD )
print("Since this model presumes that \u03A9m is small so \u03A9\u039B is calculated to be about",rans_L )
print()
print('The r\u00b2 is:', R_square)
print('The weighted F-statistic is:', rFstat)
print("The reduced goodness of fit, according to astronomers, \u03C7\u00b2 is:", newxsqrded)
print("The BIC estimate is:",rBIC)
print()

# Routines to save figues in eps and pdf formats
fig.savefig("DL-LambdaE-DS.eps", format="eps", dpi=1200, bbox_inches="tight", transparent=True)
fig.savefig("DL-LambdaE-DS.pdf", format="pdf", dpi=1200, bbox_inches="tight", transparent=True)



















