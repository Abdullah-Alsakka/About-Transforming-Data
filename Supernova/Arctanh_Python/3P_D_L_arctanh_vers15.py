#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 17:41:24 2022
This code referers to section 2.5.5 Arctanh model, equation 16 and section 3.4.
@author: Mike
"""
print()
print("This curve_fit regression routine of Python scipy, uses the 'distance mag' data, converted to luminosity distances, D_L, vs expansion factor, from Brout et al. 2022, 'The Pantheon+ Analysis: Cosmological Constraints' Astrophys. J. vol. 938, 110. The model selected here is the arctanh analytical solution with three parameters, the Hubble constant, Hu, the normalised matter density, O_m, and the spacetime parameter, O_k. No estimate of dark energy is possible.")

# import data and library files
import numpy as np
import csv
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import math

# open data file selecting the distance, distance standard deviation and recession velocity columns
with open("DATA2B.csv","r") as i:
    rawdata = list(csv.reader(i, delimiter = ","))
    
# ignoring the first row which are strings
exampledata = np.array(rawdata[1:],dtype=float)

xdata = exampledata[:,1]
ydata = exampledata[:,4]
error = exampledata[:,7]

#define the function - the model to be examined, where x represents the independent variable values with b (Hubble constant), c (normalized matter density), d (spacetime density) are the parameters to be estimated
def func(x,b,c,d):
    return (litesped/(x*b*np.sqrt(abs(d))))*np.sinh(2*(np.arctanh(np.sqrt(abs(d)))-np.arctanh(np.sqrt(abs(d))/np.sqrt((c/x)+ (d)))))

#The initial guesses of the model parameters
p0=[65.0,0.001,0.999]

# define the lightspeed constant
litesped = 299793

# curve fit model to the data, where the first pair are the lower bounds and the second pair the upper bounds 
bnds = ([50.0, 0.00001, 0.00001],[80.0, 1.0, 1.0])

# when absolute_sigma = False the regression routine "normalizes" the standard deviations (errors)
params, pcov = curve_fit(func,xdata,ydata, p0, bounds = bnds, sigma = error, absolute_sigma = False)

# extracting the Hubble constant, normalized matter density and rounding the values
ans_b, ans_c, ans_d = params
rans_b = round(ans_b,2)
rans_c = round(ans_c,5)
rans_d = round(ans_d,5)

# define the parameters
b=ans_b
c=ans_c
d=ans_d

# evaluate and plot function
funcdata = func(xdata,b,c,d)

# extracting the S.D. of the above values
perr = np.sqrt(np.diag(pcov))
ans_b_SD, ans_c_SD, ans_d_SD = perr
rans_b_SD = round(ans_b_SD,2)
rans_c_SD = round(ans_c_SD,5)
rans_d_SD = round(ans_d_SD,5)

# normalised chisquar where P is the number of parameters (3), N is the number of data pairs (1702) 
P=3
N=1702
e=2.71828183

#Calculate the chi^2 according to astronomers
newxsqrd = sum(((ydata[1:-1] - func(xdata,ans_b,ans_c,ans_d)[1:-1])**2)/(error[1:-1]**2))
newxsqrded = round(newxsqrd/(N-P),2)

# estimating the goodness of fit in the common manner
chisq = sum((ydata[1:-1] - func(xdata,ans_b,ans_c,ans_d)[1:-1])**2/func(xdata,ans_b,ans_c,ans_d)[1:-1])
normchisquar = round((chisq/(N-P)),2) #rounded to 2 digits

#The BIC value is calculated as 
SSE = sum((ydata - func(xdata,ans_b,ans_c,ans_d))**2)
alt_BIC = N*math.log(e,SSE/N) + P*math.log(e,N)
normalt_BIC = round(alt_BIC,2)

# calculation of residuals
residuals = ydata - func(xdata,ans_b,ans_c,ans_d)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((ydata-np.mean(ydata))**2)

# r squared calculation
r_squared = 1 - (ss_res/ss_tot)
r2 = round(r_squared,3)
r2adjusted = round(1-(((1-r2)*(len(ydata)-1))/(len(ydata)-P-1)),3)

# plot of imported data
plt.figure(1,dpi=240)
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['lines.linewidth'] = 3
plt.xlim(0.3,1.0)
plt.ylim(0.0,16000)
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
plt.plot(xdata, funcdata, color = "orange", label = "3Parctanh Model")
plt.legend(loc='best', fancybox=True, shadow=False)
plt.show()

# print results
print()
print("The calculated Hubble constant with standard deviation are: ", rans_b,",", rans_b_SD )
print("The calculated normalized matter density with standard deviation are: ", rans_c,",", rans_c_SD )
print("The calculated normalized spacetime density with standard deviation are: ", rans_d,",", rans_d_SD )
print()
print("The adjusted r\u00b2 is calculated to be: ",r2adjusted)
print("The reduced goodness of fit, according to astronomers, \u03C7\u00b2 is: ", newxsqrded)
print("The common reduced goodness of fit \u03C7\u00b2 is: ", normchisquar)
print("The estimate for BIC is: ", normalt_BIC)
print()

# Routines to save figues in eps and pdf formats
fig.savefig("3P_Arctanh_D_L_data.eps", format="eps", dpi=1200, bbox_inches="tight", transparent=True)
fig.savefig("3P_Arctanh_D_L_data.pdf", format="pdf", dpi=1200, bbox_inches="tight", transparent=True)



















