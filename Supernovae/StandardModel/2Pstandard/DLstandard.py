#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 17:26:53 2024

@author: mike


This curve_fit regression routine of Python scipy, uses the 'distance mag' data, 
converted to luminosity distances, D_L, vs expansion factor, from Brout et al. 2022, 
'The Pantheon+ Analysis: Cosmological Constraints' Astrophys. J. vol. 938, 110. 
The standard (LCDM) model used here requires numerical integration with two parameters, 
the Hubble constant, Hu and the normalised matter density, O_m with presumed Euclidean space 
geometry, that is sinn(x) = x. An estimate of the normalized cosmological constant 
(dark energy) is the remainder. This is sometimes termed the standard model of cosmology,
Riess et al. 2004, ApJ, vol. 607, p. 665-687 and Riess et al. 2022, ApJ Lett, vol. 934, p. L7 
"""

import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import integrate as intg
import math
from sklearn.metrics import r2_score
from astropy.stats.info_theory import bayesian_info_criterion

print("This is the current standard model of cosmology presuming Euclidean space geometry, sinn(x) = x.")
print("Only 2 parameters can be estinated with this model.")

# open data file
with open("DATA2B.csv","r") as i:
    rawdata = list(csv.reader(i, delimiter = ","))
    
# ignore the first row   
exampledata = np.array(rawdata[1:],dtype=float)

xdata = exampledata[:,1]
ydata = exampledata[:,4]
error = exampledata[:,7]

def integr(x,O_m):
    return intg.quad(lambda t: 1/(t*(np.sqrt((O_m/t)+(1-O_m)*t**2))), x, 1)[0]
    
def func2(x, O_m):
    return np.asarray([integr(xx,O_m) for xx in x])

litesped = 299793

# Hu is the Hubble constant
def func3(x,Hu,O_m):
    return (litesped/(x*Hu))*(func2(x,O_m))

init_guess = np.array([70,0.30])
bnds=([60,0.001],[80,1.0])

# the bnds are the two lower and higher bounds for the unknowns 
#(parameters), when absolute_sigma = False the errors are "normalized"
#func3 does not require transformation to use D_L vs. recession velocity data.
params, pcov = curve_fit(func3, xdata, ydata, p0 = init_guess, bounds = bnds, sigma = error, absolute_sigma = False)

#extracting the two parameters from the solution and rounding the values
ans_Hu, ans_O_m = params
Rans_Hu = round(ans_Hu,2)
Rans_O_m = round(ans_O_m,3)

# extracting the S.D. and rounding the values
perr = np.sqrt(np.diag(pcov))
ans_Hu_SD, ans_O_m_SD = np.sqrt(np.diag(pcov))
Rans_Hu_SD = round(ans_Hu_SD,2)
Rans_O_m_SD = round(ans_O_m_SD,3)

# where P is the number of parameters (2), N is the number of data pairs (1702)
P=2
N=1702
e = 2.718281

#Calculate the chi^2 according to astronomers
newxsqrd = sum(((ydata[1:-1] - func3(xdata,ans_Hu,ans_O_m)[1:-1])**2)/(error[1:-1]**2))
newxsqrded = np.round(newxsqrd/(N-P),2)

#The usual method for BIC calculation is
SSE = sum((ydata - func3(xdata,ans_Hu,ans_O_m))**2)
log_SSE = math.log(e,SSE)
small_bic = bayesian_info_criterion(log_SSE, P, N)
rBIC = round(small_bic,2)

#calculation of residuals
residuals = ydata - func3(xdata,ans_Hu,ans_O_m)
#residuals_lsq = data - data_fit_lsq
ss_res = np.sum(residuals**2)
ss_tot = np.sum((ydata-np.mean(ydata))**2)

#routine for calculating r squared
ycalc = func3(xdata,ans_Hu,ans_O_m)
R_sqrd = r2_score(ydata, ycalc)
R_square = round(R_sqrd,4)

#Calculation of the weighted F-statistic
SSEw = sum((1/error)*(residuals)**2)
SSM = sum((1/error)*(ydata - np.mean(ydata))**2)
MSR = (SSM - SSEw)/(P)
MSE = SSEw/(N-P)
Fstat = MSR/MSE
rFstat = round(Fstat,1)

#plot of imported data and model
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['lines.linewidth'] = 3
plt.figure(1,dpi=240)
plt.xlabel("Expansion factor")
plt.ylabel("Distance (Mpc)")
plt.xlim(0.3,1)
plt.ylim(0.0,16000)
plt.xscale("linear")
plt.yscale("linear")
fig, ax = plt.subplots()
plt.xlabel("Expansion factor, \u03BE", fontsize=18)
plt.ylabel("Luminosity distance (Mpc)", fontsize=18)
ax.tick_params(axis="y", direction='in', length=8)
ax.tick_params(axis="x", direction='in', length=8)
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(3)
    ax.tick_params(width=3)
plt.errorbar(xdata, ydata, yerr=error, fmt='.k', capsize = 4)
plt.plot(xdata, func3(xdata,ans_Hu,ans_O_m), color = "orange", label = "$D_L$standard model")
plt.legend(loc='best', fancybox=True, shadow=False)

#print results
print()
print("The calculated Hubble constant with S.D. is:", Rans_Hu,",", Rans_Hu_SD)
print("The calculated matter density with S.D. is:", Rans_O_m,",",Rans_O_m_SD)
print()
print('The r\u00b2 is:', R_square)
print('The weighted F-statistic is:', rFstat)
print("The reduced goodness of fit, according to astronomers \u03C7\u00b2, is:", newxsqrded)
print("The BIC estimate is:",rBIC)
print()

#Saving the plots in two different formats
fig.savefig("DLstandard.eps", format="eps", dpi=2000, bbox_inches="tight", transparent=True)
fig.savefig("DLstandard.pdf", format="pdf", dpi=2000, bbox_inches="tight", transparent=True)