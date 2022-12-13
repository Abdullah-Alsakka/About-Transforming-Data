#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 12:47:21 2022

@author: mike
"""
print()
print("This routine models the gold SNe Ia data, as mag vs. z redshift, of Riess et al. 2004 with 156 data pairs. The Einstein-DeSitter model with only a single parameter, the Hubble constant, is employeed so estimates are not made for matter density, spacetime curvature or dark energy. The Python 3 least_squares robust regression routine is used with mag vs. redshift z data, the robust regression employees the loss='cauchy' specification.")
print()

import csv
import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

#open data file
with open("Gold_Riess_mag_2004.csv","r") as i:
    rawdata = list(csv.reader(i, delimiter = ","))
    
exampledata = np.array(rawdata[1:],dtype=float)

x = exampledata[:,1]
ydata = exampledata[:,2]
errors = exampledata[:,3] #The standard deviations are not used to calculate the best fit but only for plot display.

litesped = 299793

#Hu represents the Hubble constant value
params=[65] #Initial guess of the Hubble constant, Hu, value. The mag vs. redshift Einstein-deSitter model is the right-hand term of the "residual" equation.
def func1(params, x, y):
    Hu = params[0] # only a single parameter
    residual = ydata-(5*np.log10((litesped*(x+1))/Hu*(np.sinh(x/(1+x))))+25)
    return residual

#Application of the least_squares regression routine, note that bounds=(50,80) are the allowed search limits for the Hubble constant, loss='cauchy' denotes the robust method for handling the residual values not the standard deviations.
result2 = least_squares(func1, x0=params, jac='3-point',bounds=(50,80),method = 'trf', loss='cauchy',args=(x, ydata))
Hu, = result2.x

#yfit1 decribes the E-DS model
yfit1 = 5*np.log10((litesped*(x+1))/Hu*(np.sinh(x/(1+x))))+25
Hubble = np.round(Hu,2)

#calculation of residuals and the squared residuals 
residuals = ydata - yfit1
ss_res_lsq = np.sum(residuals**2)
ss_tot_lsq = np.sum((ydata-np.mean(ydata))**2)

#r squared with the parameter count P=1
P=1
#r_squared = 1 - (ss_res/ss_tot)
r_squared_lsq = 1 - (ss_res_lsq/ss_tot_lsq)

r2_lsq = round(r_squared_lsq,4)
r2adjusted = round(1-(((1-r2_lsq)*(len(ydata)-1))/(len(ydata)-P-1)),4)

#Below will be used to estimate the Covariance Matrix of the parameters using the following formula: Sigma = (J'J)^-1.
J = result2.jac
cov = np.linalg.inv(J.T.dot(J))

#To find the variance of the parameters one can then use:
var = np.sqrt(np.diagonal(cov))
StndDev, = var

#We write a comma after StndDev, because we are extracting this value from tuple. Below is just rounding off the value of the standard deviation to something one might believe - only two digits.
normSD = round(StndDev,2)

#calculate the statistical fitness, using 156 as the number of data pairs and 1 as the degree of freedom (paramater count).
chisq = sum((ydata-yfit1)**2/(yfit1))
chisquar = np.round(chisq,2)
normchisquar = np.round((chisquar/(156-P)),4)
"""
#The BIC value is calculated as; BIC from Bayesian Information Criteria
BIC = 156 * np.log10(chisq/156) + P*np.log10(156)
normBIC = round(BIC,2)
"""
#Plot of data and best model regression.
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['lines.linewidth'] = 3
plt.xlim(0.0,2.0)
plt.ylim(32,46)
plt.xscale("linear")
plt.yscale("linear")
fig, ax = plt.subplots()
ax.tick_params(axis="y", direction='in', length=10)
ax.tick_params(axis="x", direction='in', length=10)
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(3)
    ax.tick_params(width=2)
plt.errorbar(x, ydata, yerr=errors, fmt='.k', capsize = 5)
plt.plot(x, ydata, 'bo', label="SNe Ia data")
plt.plot(x, yfit1, color="green", label="E-DS model")
plt.xlabel('Redshift z', fontsize = 18)
plt.ylabel('mag (no units)', fontsize = 18)
plt.legend(loc='best', fancybox=True, shadow=False)
plt.title("E-DS model with mag vs. z data", fontsize = 18)
plt.show()

print()
print("The estimated Hubble constant and S.D. are:", Hubble, ",",normSD)
print ('The r\u00b2_adj is:'+str(r2adjusted))
print("The goodness of fit, \u03C7\u00b2, estimate is: ", chisquar)
print("The reduced goodness of fit, \u03C7\u00b2, guesstimate is: ", normchisquar)
#print("Reduced \u03C7\u00b2 = \u03C7\u00b2/(N-P), where N are the number of data pairs and P is the parameter count.")
print()
print("There is no value for the cosmological constant in this solution")

#Routines to save figues in eps and pdf formats
fig.savefig("magE-DS_least_squares.eps", format="eps", dpi=2000, bbox_inches="tight", transparent=True)
fig.savefig("magE-DS_least_squares.pdf", format="pdf", dpi=2000, bbox_inches="tight", transparent=True)
