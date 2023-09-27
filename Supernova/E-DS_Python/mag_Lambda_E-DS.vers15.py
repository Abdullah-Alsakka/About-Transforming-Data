#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 17:41:24 2022
This code solves equation A26 of Appendix, section A3.2, E-DS universe with \Omega_m=0 and \Omega_{\Lambda}>0. This code uses distance mag and redshift data for solution.
@author: Mike
"""
print()
print("This curve_fit regression routine of Python scipy, uses the distance mag vs redshift, z, data, from Brout et al. 2022, 'The Pantheon+ Analysis: Cosmological Constraints' Astrophys. J. vol. 938, 110. This is the arctanh, analytical solution to the Einstein-DeSitter model with two parameters, the Hubble constant, Hu and the normalised spacetime density, \u03A9k. No estimation is possible for \u03A9m, the normalised matter density which is presume as 0. The value of the normalised contribution from the cosmological constant, \u039B, is the remainder.")
print()

# import data and Python 3 library files
import numpy as np
import csv
import math
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# open data file
with open("DATA.csv","r") as i:
    rawdata = list(csv.reader(i, delimiter = ","))
    
# ignore the first row    
exampledata = np.array(rawdata[1:],dtype=float)

xdata = exampledata[:,0]
ydata = exampledata[:,2]
error = exampledata[:,3]

# define the function - the model to be examined, where x represents the independent variable values and b (Hubble constant) and c (space density) are the parameters to be estimated
def func(x,b,c):
    return (((litesped*(1+x))/(b*np.sqrt(c)))*np.sinh(np.arctanh(np.sqrt(c)/(np.sqrt((1-c)/(1+x)**2)+c))-np.arctanh(np.sqrt(c))))

def func2(x,b,c):
    return 5*np.log10(func(x,b,c)) + 25

# the initial guesses of the model parameters
p0=[70.0,0.002]

# specify the constant speed of light
litesped = 299793

# curve fit the model to the data, the bnds are the lower and upper bounds for the two parameters
bnds = ([50.0, 0.0002],[80.0,1.0])
params, pcov = curve_fit(func2,xdata,ydata, p0, bounds = bnds, sigma = error, absolute_sigma = False)

# extracting the two parameter values and rounding the values
ans_b, ans_c = params
rans_b = round(ans_b, 2)
rans_c = round(ans_c, 5)
ans_d=1-ans_c
rans_d = round(ans_d, 5)

# specify the parameters for the plot with the model
b=ans_b
c=ans_c

# evaluate the function
func2data = func2(xdata,b,c)

# extracting the two standard deviations and rounding the values
perr = np.sqrt(np.diag(pcov))
ans_bSD, ans_cSD = perr
rans_bSD = round(ans_bSD,2)
rans_cSD = round(ans_cSD,5)

# normalised chisquar where P is the number of parameters (2) and N the number of data pairs
P=2
N=1701
e = 2.718281

#Calculate the chi^2 according to astronomers
newxsqrd = sum(((ydata - func2(xdata,ans_b,ans_c))**2)/error**2)
newxsqrded = np.round(newxsqrd/(N-P),2)

# estimating the goodness of fit in the common manner
chisq = sum(((ydata - func2(xdata,ans_b,ans_c))**2)/func2(xdata,ans_b,ans_c))
normchisquar = round((chisq/(N-P)),4) #rounded to 4 digits

#the BIC value is calculated as 
SSE = sum((ydata - func2(xdata,ans_b,ans_c))**2)
alt_BIC = N*math.log(e,SSE/N) + P*math.log(e,N)
normalt_BIC = round(alt_BIC,2)

# calculation of residuals
residuals = ydata - func2(xdata,ans_b,ans_c)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((ydata-np.mean(ydata))**2)

# r squared calculation
r_squared = 1 - (ss_res/ss_tot)
r2 = round(r_squared,3)
r2adjusted = round(1-(((1-r2)*(len(ydata)-1))/(len(ydata)-P-1)),3)

#plt.plot(xdata,ydata)
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['lines.linewidth'] = 3
plt.xlim(0.0,1.8)
plt.ylim(25.0,46.0)
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
plt.ylabel("$\mu$ (distance mag)", fontsize = 18)
plt.plot(xdata, func2(xdata,ans_b,ans_c), color = "green", label = "mag\u039BE-DS model")
plt.legend(loc='best', fancybox=True, shadow=False)
plt.show()

#print results
print()
print("The calculated Hubble constant with S.D. is: ", rans_b, ",",rans_bSD )
print("The normalised spacetime density, \u03A9k, with S.D. is: ", rans_c,"," , rans_cSD)
print("The normalised \u03A9\u039B is: ", rans_d,)
print()
print("The adjusted r\u00b2 is calculated to be: ",r2adjusted)
print("The reduced goodness of fit, according to astronomers, \u03C7\u00b2 is: ", newxsqrded)
print("The common reduced goodness of fit \u03C7\u00b2 is: ", normchisquar)
print("The estimate for BIC is: ", normalt_BIC)
print()

#Routines to save figues in eps and pdf formats
fig.savefig("mag_LE-DS.eps", format="eps", dpi=2000, bbox_inches="tight", transparent=True)
fig.savefig("mag_LE-DS.pdf", format="pdf", dpi=2000, bbox_inches="tight", transparent=True)


















