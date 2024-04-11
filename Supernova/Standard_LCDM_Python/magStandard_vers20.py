#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 17:53:20 2022

@author: Mike
This curve_fit regression routine of Python scipy, uses the mag vs redshift (z) data from 
Brout et al. 2022, 'The Pantheon+ Analysis: Cosmological Constraints' Astrophys. J. vol. 938, 110. 
This variation of the standard (LCDM) model used here has two parameters: Hubble constant, Hu, 
normalised matter density, O_m; the cosmological constant, \Omega_L, is the remainder of 
information in a universe with Euclidean space geometry.
"""
print()
print("This is the magStandard model presuming Euclidean space geometry, sinn(x) = x.")

# import the data file and the Python 3 libraries
import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import integrate as intg
import math
from sklearn.metrics import r2_score
from astropy.stats.info_theory import bayesian_info_criterion

# open data file
with open("DATA.csv","r") as i:
    rawdata = list(csv.reader(i, delimiter = ","))
    
# ignore the top row    
exampledata = np.array(rawdata[1:],dtype=float)

xdata = exampledata[:,0]
ydata = exampledata[:,2]
error = exampledata[:,3]

# initial guess for the normalized matter density, O_m
O_m = 0.30

# where t is the "dummy" variable during integration
def integr(x,O_m):
    return intg.quad(lambda t: (1/(np.sqrt(((1+t)**2)*(1+O_m*t) - t*(2+t)*(1-O_m)))), 0, x)[0]
    
def func2(x, O_m):
    return np.asarray([integr(xx,O_m) for xx in x]) 

# specify the speed of light
litesped = 299793

def func3(x,Hu,O_m):
    return 5*(np.log10((litesped*(1+x)/Hu)*(func2(x,O_m)))) + 25

# guesses for the Hubble constant, Hu, and the normalized matter density, O_m
init_guess = np.array([70,0.30])

# allowed range for the two parameters
bnds=([50,0.01],[80,1.0])

# fitting the model to the data, note that when choosing absolute_sigma = False the standard deviations (error) are normalized
params, pcov = curve_fit(func3, xdata, ydata,p0 = init_guess, bounds = bnds, sigma = error, absolute_sigma = False)

# extracting and rounding the parameter, Hu and O_m, values
ans_Hu, ans_O_m = params
rans_Hu = round(ans_Hu,2)
rans_O_m = round(ans_O_m,3)

# extracting and rounding the estimated standard deviations.
perr = np.sqrt(np.diag(pcov))
SD_Hu, SD_O_m = perr
rSD_Hu = round(SD_Hu,2)
rSD_O_m = round(SD_O_m,3)

# normalised chisquar is calculated for 1701 data pairs with P the parameter count (2) as
P=2
N=1701
e=2.71828183

#Calculate the reduced chi^2 according to astronomers
newxsqrd = sum(((ydata - func3(xdata,ans_Hu,ans_O_m))**2)/(error**2))
newxsqrded = np.round(newxsqrd/(N-P),2)

#Calculate the reduced chi^2 as commonly done
xsqrd = sum(((ydata - func3(xdata,ans_Hu,ans_O_m))**2)/func3(xdata,ans_Hu,ans_O_m))
normxsqrd = np.round(xsqrd/(N-P),6)

#The usual method for BIC calculation is
SSE = sum((ydata - func3(xdata,ans_Hu,ans_O_m))**2)
log_SSE = math.log(e,SSE)
small_bic = bayesian_info_criterion(log_SSE, P, N)
rBIC = round(small_bic,2)

# calculation of residuals
residuals = ydata - func3(xdata,ans_Hu,ans_O_m)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((ydata-np.mean(ydata))**2)

#easy routine for calculating r squared
ycalc = func3(xdata,ans_Hu,ans_O_m)
R_sqrd = r2_score(ydata, ycalc)
R_square = round(R_sqrd,4)

#plot of imported data and model
plt.figure(1,dpi=240)
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['lines.linewidth'] = 3
plt.xlim(0.0,1.8)
plt.ylim(32.0,46.0)
plt.xscale("linear")
plt.yscale("linear")
fig, ax = plt.subplots()
ax.tick_params(axis="y", direction='in', length=14)
ax.tick_params(axis="x", direction='in', length=10)
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(3)
    ax.tick_params(width=3)
plt.errorbar(xdata, ydata, yerr=error, fmt='.k', capsize = 4)
plt.plot(xdata, func3(xdata,ans_Hu,ans_O_m), color = "green", label = "magStandard model")
plt.xlabel("Redshift z", fontsize = 18)
plt.ylabel("$\mu$ (mag)", fontsize = 18)
plt.legend(loc='best', fancybox=True, shadow=False)

#print results
print()
print("The calculated Hubble constant and S.D. are:", rans_Hu, ",",rSD_Hu)
print("The calculated normalised matter density and S.D. are:",rans_O_m, ",",rSD_O_m )
print()
print('The r\u00b2 is:', R_square)
print("The reduced goodness of fit, as per astronomers, \u03C7\u00b2 is:", newxsqrded)
print("The reduced goodness of fit, as commonly done, \u03C7\u00b2 is:", normxsqrd)
print("The BIC estimate is: ",rBIC)
print()

#commands to save plots in two different formats
fig.savefig("magStandard.eps", format="eps", dpi=2000, bbox_inches="tight", transparent=True)
fig.savefig("magStandard.pdf", format="pdf", dpi=2000, bbox_inches="tight", transparent=True)
