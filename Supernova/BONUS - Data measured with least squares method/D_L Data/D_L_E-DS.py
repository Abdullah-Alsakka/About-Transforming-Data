#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 12:47:21 2022

@author: mike
"""
print()
print("This routine models the 156 gold SNe Ia data and the position of the earth, as D_L vs. expansion factor, of Riess et al. 2004. The model used here is the Einstein-DeSitter model with only a single parameter, the Hubble constant; no values are calulated for matter density, dark energy or spacetime curvature. The Python 3 least_squares robust regression routine is employed with the specification that the robust routine employees the loss='cauchy' specification.")

import csv
import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import math

#open data file
with open("Gold_Riess_D_L_2004.csv","r") as i:
    rawdata = list(csv.reader(i, delimiter = ","))
    
# ignore the string data in the first row
exampledata = np.array(rawdata[1:],dtype=float)

x = exampledata[:,3]
ydata = exampledata[:,4]
errors = exampledata[:,7] #the standard deviations are not used in the regression routine but only in the plot.

#the constant speed of light 
litesped = 299793

#Initial guess of the Hubble constant, Hu, value.
params=[70] 

# he Einstein-deSitter model is the right-hand term of the "residual" equation.
def func1(params, x, y):
    Hu = params[0] # params[1], params[2]
    residual = ydata-((litesped/(x*Hu))*np.sinh(1-x))
    return residual

#Application of the least_squares regression routine, note that bounds=(50,80) are the allowed search limits for the Hubble constant, loss='cauchy' denotes the robust method for handling the residual values not the standard deviations.
result2 = least_squares(func1, x0=params, jac='3-point',bounds=(50,80),method = 'trf', loss='cauchy',args=(x, ydata))
Hu, = result2.x

#yfit1 now decribes the E-DeS model
yfit1 = (litesped/(x*Hu))*np.sinh(1-x)
Hubble = np.round(Hu,2)

#calculation of residuals and the squared residuals 
residuals = ydata - yfit1
ss_res_lsq = np.sum(residuals**2)
ss_tot_lsq = np.sum((ydata-np.mean(ydata))**2)

#r squared = 1 - (ss_res/ss_tot)
r_squared_lsq = 1 - (ss_res_lsq/ss_tot_lsq)
#r2 = round(r_squared,3)
r2_lsq = round(r_squared_lsq,3)

# where P is the parameter count, the r2adjusted value rounded to 3 digits
P=1
N=157
e = 2.718281
r2adjusted = round(1-(((1-r2_lsq)*(len(ydata)-1))/(len(ydata)-P-1)),3)

#Below will be used to estimate the Covariance Matrix of the parameters using the following formula: Sigma = (J'J)^-1.
J = result2.jac
cov = np.linalg.inv(J.T.dot(J))

#To find the variance of the parameters one can then use:
var = np.sqrt(np.diagonal(cov))
StndDev, = var

#We write a comma after StndDev, because we are extracting this value from tuple. Below is just rounding off the value of the standard deviation to something one might believe - only three digits.
normSD = round(StndDev,3)

#calculate the statistical fitness, using 157 as the number of data pairs and 1 as the degree of freedom (parameter count). Note the values of the errors are those of the tablulated SNe Ia distances.
chisq = sum((ydata[1:-1]-yfit1[1:-1])**2/(errors[1:-1])**2)
chisquar = np.round(chisq,2)

#normalised chisquar is calculated from the number of data pairs (157) minus the number of free parameters (1). 
normchisquar = np.round((chisquar/(N-P)),2)

#The BIC value is calculated as; BIC from Bayesian Information Criteria
BIC = N * math.log(e,(chisq/N)) + P*math.log(e,N)
normBIC = round(BIC,2)

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['lines.linewidth'] = 3
plt.xlim(0.3,1.0)
plt.ylim(0.0,16000)
plt.xscale("linear")
plt.yscale("linear")
fig, ax = plt.subplots()
ax.tick_params(axis="y", direction='in', length=8)
ax.tick_params(axis="x", direction='in', length=10)
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(3)
    ax.tick_params(width=2)
plt.errorbar(x, ydata, yerr=errors, fmt='.k', capsize = 5)
plt.plot(x, ydata, 'bo', label="SNe Ia data")
plt.plot(x, yfit1, color="orange", label="E-DS model")
plt.xlabel('Expansion factor, \u03BE', fontsize = 18)
plt.ylabel('Luminosity distance (Mpc)', fontsize = 16)
plt.legend(loc='best', fancybox=True, shadow=False)
plt.title("Einstein-DeSitter model with D_L vs. exp. fact. data", fontsize = 16)
plt.show()

print()
print("The estimated Hubble constant is:", Hubble)
print("The S.D. of the Hubble constant is", normSD)
print()
print ('The r\u00b2_adj is:'+str(r2adjusted))
#print("The goodness of fit, \u03C7\u00b2, estimate is:", chisquar)
print("The reduced goodness of fit, \u03C7\u00b2, estimate is:", normchisquar)
print("The BIC value is:", normBIC)
print()
print("There is no value for the cosmological constant in this solution")

#Routines to save figues in eps and pdf formats
fig.savefig("E-DS_least_squares_D_L_data.eps", format="eps", dpi=2000, bbox_inches="tight", transparent=True)
fig.savefig("E-DS_least_squares_D_L_data.pdf", format="pdf", dpi=2000, bbox_inches="tight", transparent=True)
