#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 17:41:24 2022

@author: Mike
print()
This curve_fit regression routine of Python scipy, uses the 'distance mag' data, 
converted to luminosity distances, D_L, vs expansion factor, from Brout et al. 2022,
'The Pantheon+ Analysis: Cosmological Constraints' Astrophys. J. vol. 938, 110 AND the TRGB data, from GS Anand et al. 
2022, 'Comparing Tip of the Red Giant Branch Distance Scales:' Astrophys. J. vol. 932, 15.
The model selected here is the arctanh analytical solution with two parameters, the Hubble constant, Hu and the 
normalised matter density, O_m. No estimate of dark energy is possible but the normalised value for space influence
O_k is the remainder.
"""
print()
print("This is the arctanh model using the luminosity distance data, D_L vs. expansion factor.")
print("This model presumes elliptical space geometry with sinn(x) = sinh(x).")
# import data and library files
import numpy as np
import csv
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import math
from sklearn.metrics import r2_score
from astropy.stats.info_theory import bayesian_info_criterion


# open data file selecting the distance, distance standard deviation and recession velocity columns
with open("TRGB_D_L_DATA.csv","r") as i:
    rawdata = list(csv.reader(i, delimiter = ","))
    
# ignoring the first row
exampledata = np.array(rawdata[1:],dtype=float)

xdata = exampledata[:,1]
ydata = exampledata[:,4]
error = exampledata[:,7]

#define the function - the model to be examined, where x represents the independent variable values and b (Hubble constant) and c (normalized matter density) are the parameters.
def func(x,b,c):
    return (litesped/(x*b*np.sqrt(abs(1-c))))*np.sinh(2*(np.arctanh(np.sqrt(abs(1-c)))-np.arctanh(np.sqrt(abs(1-c))/np.sqrt((c/x)+ (1-c)))))

#The initial guesses of the model parameters
p0=[70.0,0.002]

# define the lightspeed constant
litesped = 299793

# curve fit model to the data, where the first pair are the lower bounds and the second pair the upper bounds. Note how the small lower limit is required for the matter density. 
bnds = ([60.0,0.0001],[80.0,1.00])

# when absolute_sigma = False, the regression routine "normalizes" the standard deviations (errors)
# the log transformation is not required with this model.
params, pcov = curve_fit(func,xdata,ydata, p0, bounds = bnds, sigma = error, absolute_sigma = False)

# extracting the Hubble constant, normalized matter density and rounding the values
ans_b, ans_c = params
rans_b = round(ans_b,2)
rans_c = round(ans_c,10)

# evaluate and plot the function
funcdata = func(xdata,ans_b,ans_c)

# extracting the S.D. of both above values
perr = np.sqrt(np.diag(pcov))
ans_b_SD, ans_c_SD = perr
rans_b_SD = round(ans_b_SD,2)
rans_c_SD = round(ans_c_SD,4)

# normalised chisquar where P is the number of parameters (2), N the number of data pairs and normchisquar is calculated from 
P=2
N=1717
e = 2.718281

#Calculate the chi^2 according to astronomers
newxsqrd = sum(((ydata[1:-1] - func(xdata,ans_b,ans_c)[1:-1])**2)/(error[1:-1]**2))
newxsqrded = round(newxsqrd/(N-P),2)
"""
# estimating the goodness of fit in the more common manner
chisq = sum((ydata[1:-1] - func(xdata,ans_b,ans_c)[1:-1])**2/func(xdata,ans_b,ans_c)[1:-1])
normchisquar = round((chisq/(N-P)),2) #rounded to 2 digits
"""
#The usual method for BIC calculation is
SSE = sum((ydata - func(xdata,ans_b,ans_c))**2)
log_SSE = math.log(e,SSE)
small_bic = bayesian_info_criterion(log_SSE, P, N)
rBIC = round(small_bic,2)

# calculation of residuals
residuals = ydata - func(xdata,ans_b,ans_c)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((ydata-np.mean(ydata))**2)

#easy routine for calculating r squared using the r2_score routine
ycalc = func(xdata,ans_b,ans_c)
R_sqrd = r2_score(ydata, ycalc)
R_square = round(R_sqrd,4)

#Calculation of the F-statistic (weighted)
SSEw = sum((1/error)*(residuals)**2)
SSM = sum((1/error)*(ydata - np.mean(ydata))**2) 
MSR = (SSM - SSEw)/(P)
MSE = SSEw/(N-P)
Fstat = MSR/MSE
rFstat = round(Fstat,1)

# plot of imported data with best fit curve
plt.figure(1,dpi=240)
plt.xscale("linear")
plt.yscale("linear")
plt.xlim(0.3,1.3)
plt.ylim(-1000,22000)
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['lines.linewidth'] = 3
fig, ax = plt.subplots()
ax.tick_params(axis="y", direction='in', length=14)
ax.tick_params(axis="x", direction='in', length=10)
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(2)
    ax.tick_params(width=2)
plt.xlabel("Expansion factor, \u03BE", fontsize = 18)
plt.ylabel("Luminosity distance (Mpc)", fontsize=18)
plt.errorbar(xdata, ydata, yerr=error, fmt='.k', capsize = 2)
plt.plot(xdata, funcdata, color = "orange", label = "$D_L$Arctanh model")
plt.legend(loc='best', fancybox=True, shadow=False)
plt.show()

# print results
print()
print("The Hubble constant with standard deviation are: ", rans_b,",", rans_b_SD )
print("The normalized matter density with standard deviation are: ", rans_c,",", rans_c_SD )
print()
print('The r\u00b2 is:', R_square)
print('The weighed F-statistic is:', rFstat)
print("The reduced goodness of fit, according to astronomers, \u03C7\u00b2 is: ", newxsqrded)
#print("The reduced goodness of fit in the more common manner \u03C7\u00b2 is: ", normchisquar)
print("The estimate for BIC is: ",rBIC)
print()

# Routines to save figues in eps and pdf formats
fig.savefig("DLArctanh.eps", format="eps", dpi=1200, bbox_inches="tight", transparent=True)
fig.savefig("DLArctanh.pdf", format="pdf", dpi=1200, bbox_inches="tight", transparent=True)



















