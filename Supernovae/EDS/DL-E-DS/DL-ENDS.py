#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 17:53:20 2022
@author: Mike

This curve_fit regression routine of Python scipy, uses data from 
Brout et al. 2022,'The Pantheon+ Analysis: Cosmological Constraints' Astrophys. J. vol. 938, 
110 after calculations of the D_L and recession velocities. The measured distances, D_L, 
are plotted vs. the recession velocity (1/1+z). "," This variation of the Einstein-Newcomb-
deSitter model has two parameters: Hubble constant, Hu, and normalised matter density, O_m; 
in a universe with apparant Euclidean geometry. This model is reviewed in Yershov, V.N. 2023, Universe, 
vol. 9(5), 204. The corrected version of his model is used here with the Python curve_fit 
regression routine.
"""

# import the data file and the Python 3 libraries
import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import integrate as intg
import math
from sklearn.metrics import r2_score
from astropy.stats.info_theory import bayesian_info_criterion

print("This is the DL-ENDS model, a version of the Einstein-Newcomb-deSitter model of cosmology.")

# open data file
with open("DATA2B.csv","r") as i:
    rawdata = list(csv.reader(i, delimiter = ","))
    
# ignore the first row   
exampledata = np.array(rawdata[1:],dtype=float)

xdata = exampledata[:,1]
ydata = exampledata[:,4]
error = exampledata[:,7]

# initial guess for the normalized matter density, O_m
O_m = 0.05

# where t is the "dummy" variable during integration
def integr(x,O_m):
    return intg.quad(lambda t: (1/(np.sqrt(((O_m/t)-O_m+1)))), x, 1)[0]
    
def func2(x, O_m):
    return np.asarray([integr(xx,O_m) for xx in x]) 

# specify the speed of light
litesped = 299793

# guesses for the Hubble constant, Hu, and the normalized matter density, O_m
init_guess = np.array([70,0.05])

def func3(x,Hu,O_m):
    return (litesped/(x*Hu)*(func2(x,O_m)))

# allowed range for the two parameters
bnds=([60,0.0001],[80,1.0])

# fitting the model to the data, note that when choosing absolute_sigma = False the standard deviations (error) are normalized for use in weighted regression.
params, pcov = curve_fit(func3, xdata, ydata,p0 = init_guess, bounds = bnds, sigma = error, absolute_sigma = False)

# extracting and rounding the parameter, Hu and O_m, values
ans_Hu, ans_O_m = params
rans_Hu = round(ans_Hu,2)
rans_O_m = round(ans_O_m,4)

# extracting and rounding the estimated standard deviations.
perr = np.sqrt(np.diag(pcov))
SD_Hu, SD_O_m = perr
rSD_Hu = round(SD_Hu,2)
rSD_O_m = round(SD_O_m,4)

# normalised chisquared is calculated for 1702 data pairs with P the parameter count (2) as
P=2
N=1702
e=2.71828183

#Calculate the chi^2 according to astronomers
newxsqrd = sum(((ydata[1:-1] - func3(xdata,ans_Hu,ans_O_m)[1:-1])**2)/(error[1:-1]**2))
newxsqrded = round(newxsqrd/(N-P),2)

#The usual method for BIC calculation is
SSE = sum((ydata - func3(xdata,ans_Hu,ans_O_m))**2)
log_SSE = math.log(e,SSE)
small_bic = bayesian_info_criterion(log_SSE, P, N)
rBIC = round(small_bic,2)

# calculation of residuals
residuals = ydata - func3(xdata,ans_Hu,ans_O_m)
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
plt.figure(1,dpi=240)
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['lines.linewidth'] = 3
plt.xlim(0.3,1.0)
plt.ylim(0.0,30000)
plt.xscale("linear")
plt.yscale("linear")
fig, ax = plt.subplots()
ax.tick_params(axis="y", direction='in', length=14)
ax.tick_params(axis="x", direction='in', length=10)
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(3)
    ax.tick_params(width=3)
plt.errorbar(xdata, ydata, yerr=error, fmt='.k', capsize = 4)
plt.plot(xdata, func3(xdata,ans_Hu,ans_O_m), color = "orange", label = "$D_L$ENDS model")
plt.xlabel("Expansion factor, \u03BE", fontsize = 18)
plt.ylabel("Luminosity distance (Mpc)", fontsize = 18)
plt.legend(loc='best', fancybox=True, shadow=False)

#print results
print()
print("The calculated Hubble constant and S.D. are: ", rans_Hu, ",",rSD_Hu)
print("The calculated normalised matter density and S.D. are: ",rans_O_m, ",",rSD_O_m )
print("Note that values for \u03A9k and \u03A9\u039B cannot be calculated with this model.")
print()
print('The r\u00b2 is:', R_square)
print('The weighted F-statistic is:', rFstat)
print("The reduced goodness of fit, as per astronomers, \u03C7\u00b2 is: ", newxsqrded)
print("The BIC estimate is: ",rBIC)
print()

#commands to save plots in two different formats
fig.savefig("DL-ENDS.eps", format="eps", dpi=2000, bbox_inches="tight", transparent=True)
fig.savefig("DL-ENDS.pdf", format="pdf", dpi=2000, bbox_inches="tight", transparent=True)
