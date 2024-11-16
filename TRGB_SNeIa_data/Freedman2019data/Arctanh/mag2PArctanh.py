#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 17:41:24 2022

@author: Mike
This curve_fit regression routine of Python scipy, uses the distance mag (\mu) vs redshift (z)
data, from Brout et al. 2022, 'The Pantheon+ Analysis: Cosmological Constraints'
Astrophys. J. vol. 938, 110 AND the TRGB data, from GS Anand et al. 
2022, 'Comparing Tip of the Red Giant Branch Distance Scales:' Astrophys. J. vol. 932, 15. 
This is the arctanh, analytical solution to the Friedmann-Lemaitre-Roberston-Walker (FLRW) model 
with two parameters, the Hubble constant, Hu and the normalised matter density, O_m. 
No estimation is possible for \Omega_L, the normalised cosmological constant (dark energy) but O_k, 
the normalised contribution from space is estimated, this is the remainder.
"""

print()
print("This is the arctanh model using the distance mag data vs. redshift, z.")
print("This model presumes elliptical space geometry with sinn(x) = sinh(x).")

# import data and Python 3 library files
import numpy as np
import csv
import math
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from astropy.stats.info_theory import bayesian_info_criterion

# open data file
with open("TRGB_mag_dataB.csv","r") as i:
    rawdata = list(csv.reader(i, delimiter = ","))
    
# ignore the first row    
exampledata = np.array(rawdata[1:],dtype=float)

xdata = exampledata[:,0]
ydata = exampledata[:,2]
error = exampledata[:,3]

# define the function - the model to be examined, where x represents the independent
# variable values and b (Hubble constant) and c (matter density) are the parameters 
# to be estimated.
def func(x,b,c):
    return (((litesped*(1+x))/(b*np.sqrt(np.abs(1-c))))*np.sinh(2*(np.arctanh(np.sqrt(np.abs(1-c)))-np.arctanh(np.sqrt(np.abs(1-c))/np.sqrt(((c*(1+x))+ (1-c)))))))

# log transformation is required for use with distance mag and redshift data.
def func2(x,b,c):
    return 5*np.log10(func(x,b,c)) + 25

# the initial guesses of the model parameters
p0=[70.0,0.002]

# specify the constant speed of light
litesped = 299792

# curve fit the model to the data, the bnds are the lower and upper bounds for the two parameters
bnds = ([60.0, 0.00001],[80.0,1.0])
params, pcov = curve_fit(func2,xdata,ydata, p0, bounds = bnds, sigma = error, absolute_sigma = False)

# extracting the two parameter values and rounding the values
ans_b, ans_c = params
rans_b = round(ans_b, 2)
rans_c = round(ans_c, 5)

# extracting the two standard deviations and rounding the values
perr = np.sqrt(np.diag(pcov))
ans_bSD, ans_cSD = perr
rans_bSD = round(ans_bSD,2)
rans_cSD = round(ans_cSD,5)

# normalised chisquar where P is the number of parameters (2) and N the number of data pairs
P=2
N=1719
e = 2.718281

#Calculate the chi^2 according to astronomers
newxsqrd = sum(((ydata - func2(xdata,ans_b,ans_c))**2)/error**2)
newxsqrded = np.round(newxsqrd/(N-P),2)
"""
# estimating the goodness of fit in the common manner
chisq = sum(((ydata - func2(xdata,ans_b,ans_c))**2)/func2(xdata,ans_b,ans_c))
normchisquar = round((chisq/(N-P)),5) #rounded to 2 digits
"""
#Calculation of the weighted F-statistic
SSEw = sum((1/error)*(ydata - func2(xdata,ans_b,ans_c))**2)
yAve = sum(ydata)/N
SSM = sum((1/error)*(ydata - yAve)**2)

MSR = (SSM - SSEw)/(P)
MSE = SSEw/(N-P)
Fstat = MSR/MSE
rFstat = round(Fstat,1)

#The usual method for BIC calculation is
SSE = sum((ydata - func2(xdata,ans_b,ans_c))**2)
log_SSE = math.log(e,SSE)
small_bic = bayesian_info_criterion(log_SSE, P, N)
rBIC = round(small_bic,2)

# calculation of residuals
residuals = ydata - func2(xdata,ans_b,ans_c)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((ydata-np.mean(ydata))**2)

#routine for calculating r squared
ycalc = func2(xdata,ans_b,ans_c)
R_sqrd = r2_score(ydata, ycalc)
R_square = round(R_sqrd,4)

#Calculation of the weighted F-statistic
SSEw = sum((1/error)*(residuals)**2)
SSM = sum((1/error)*(ydata - np.mean(ydata))**2) 
MSR = (SSM - SSEw)/(P)
MSE = SSEw/(N-P)
Fstat = MSR/MSE
rFstat = round(Fstat,1)

#plt.plot(xdata,ydata)
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
plt.xlabel("Redshift, z", fontsize = 18)
plt.ylabel("\u03bc (distance mag)", fontsize = 16)
plt.plot(xdata, func2(xdata,ans_b,ans_c), color = "green", label = "mag2PArctanh model")
plt.legend(loc='best', fancybox=True, shadow=False)
plt.show()

#print results
print()
print("The calculated Hubble constant with S.D. is:", rans_b, ",",rans_bSD )
print("The normalised matter density with S.D. is:", rans_c,"," , rans_cSD)
print()
print('The r\u00b2 is:', R_square)
print('The weighted F-statistic is:', rFstat)
print("The reduced goodness of fit, according to astronomers, \u03C7\u00b2 is:", newxsqrded)
#print("The common reduced goodness of fit \u03C7\u00b2 is:", normchisquar)
print("The BIC estimate is:",rBIC)
print()

#Routines to save figues in eps and pdf formats
fig.savefig("mag2PArctanh.eps", format="eps", dpi=2000, bbox_inches="tight", transparent=True)
fig.savefig("mag2PArctanh.pdf", format="pdf", dpi=2000, bbox_inches="tight", transparent=True)


















