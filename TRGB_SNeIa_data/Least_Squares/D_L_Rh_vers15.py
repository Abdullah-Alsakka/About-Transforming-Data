#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 12:47:21 2022
This refers to Eqs. 7 of the manuscript.
@author: Mike
"""
print()
print("This curve_fit regression routine of Python scipy, uses the 'distance mag' data, converted to luminosity distances, D_L,") 
print("vs expansion factor, from Brout et al. 2022, 'The Pantheon+ Analysis: Cosmological Constraints' Astrophys. J. vol. 938, 110.")
print("The model used here is the R_h, model with only a single parameter, the Hubble constant; no values are calculated for") 
print("matter density, dark energy or space geometry. The Python 3 least_squares robust regression routine is employed with the") 
print("specification that the routine specifies loss='cauchy'.")

import csv
import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import math

#open data file
with open("DATAE.csv","r") as i:
    rawdata = list(csv.reader(i, delimiter = ","))
    
# ignore the string data in the first row
exampledata = np.array(rawdata[1:],dtype=float)

x = exampledata[:,0]
ydata = exampledata[:,4]
errors = exampledata[:,7] #the standard deviations are not used in the regression routine but only in the plot.

#the constant speed of light 
litesped = 299793

#Initial guess of the Hubble constant, Hu, value.
params=[70] 

# The R_h model is the right-hand term of the "residual" equation.
def func1(params, x, y):
    Hu = params[0] # params[1], params[2]
    residual = ydata-(litesped*(1+x)/(Hu))*np.log(1+x)
    return residual

#Application of the least_squares regression routine, note that bounds=(50,80) are the allowed search limits for the Hubble constant, loss='cauchy' denotes the robust method for handling the residual values not the standard deviations.
result2 = least_squares(func1, x0=params, jac='3-point', bounds=(50,90), method = 'trf', loss='linear', args=(x, ydata))
Hu, = result2.x

#yfit1 now describes the Rh model
yfit1 = (litesped*(1+x)/(Hu))*np.log(1+x)
Hubble = np.round(Hu,2)

#calculation of residuals and the squared residuals 
residuals = ydata - yfit1
ss_res_lsq = np.sum(residuals**2)
ss_tot_lsq = np.sum((ydata-np.mean(ydata))**2)

#r squared = 1 - (ss_res/ss_tot)
r_squared_lsq = 1 - (ss_res_lsq/ss_tot_lsq)
#r2 = round(r_squared,3)
r2_lsq = round(r_squared_lsq,3)

# where P is the parameter count, N the number of data pairs (1720), the r^2 adjusted value rounded to 4 digits
P=1
N=1720
e = 2.718281
r2adjusted = round(1-(((1-r2_lsq)*(len(ydata)-1))/(len(ydata)-P-1)),4)

#Below will be used to estimate the Covariance Matrix of the parameters using the following formula: Sigma = (J'J)^-1.
J = result2.jac
cov = np.linalg.inv(J.T.dot(J))

#To find the variance of the parameters one can then use:
perr = np.sqrt(np.diagonal(cov))
StndDev, = perr

#We write a comma after StndDev, because we are extracting this value from tuple. Below is just rounding off the value of the standard deviation to something one might believe to only 3 digits.
normSD = round(StndDev,3)

#calculate the statistical fitness, using 1720 as the number of data pairs and 1 as the degree of freedom (parameter count). 
#Note the error values are of the SNe Ia distances.
chisq = sum((ydata-yfit1)**2/(errors)**2)
normchisquar = np.round((chisq/(N-P)),2)

#calculate the statistical fitness, in the more usual manner, with 1720 as the number of data pairs and 3 as the degree of freedom (paramater count)
chisqed = sum(((ydata[1:-1]-yfit1[1:-1])**2)/yfit1[1:-1])
normchisquared = np.round((chisqed/(N-P)),2)

#The value for BIC is calculated as
SSE = sum((ydata-yfit1)**2)
alt_BIC = math.log(e,(SSE/N)) + P*math.log(e,N)
alt_BIC = round(alt_BIC,2)

#Calculation of the weighted F-statistic
SSEw = sum((1/errors)*(residuals)**2)
SSM = sum((1/errors)*(ydata - np.mean(ydata))**2) 
MSR = (SSM - SSEw)/(P)
MSE = SSEw/(N-P)
Fstat = MSR/MSE
rFstat = round(Fstat,1)

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
plt.plot(x, yfit1, color="orange", label="R_h model")
plt.xlabel('Redshift, z', fontsize = 18)
plt.ylabel('Luminosity distance (Mpc)', fontsize = 16)
plt.legend(loc='best', fancybox=True, shadow=False)
#plt.title("$R_h$ model with $D_L$ vs. redshift data", fontsize = 16)
plt.show()

print()
print("The estimated Hubble constant is:", Hubble)
print("The S.D. of the Hubble constant is", normSD)
print()
print ('The r\u00b2_adj is:'+str(r2adjusted))
print("The reduced goodness of fit as used by astronomers, \u03C7\u00b2, estimate is:", normchisquar)
print("The reduced goodness of fit as normally used, \u03C7\u00b2, estimate is:", normchisquared)
print("The BIC value is:", alt_BIC)
print("The estimate for the reduced F stat is: ", rFstat)
print()
print("There is no value for the cosmological constant in this solution")
print()
#Routines to save figues in eps and pdf formats
fig.savefig("Rh_least_squares_D_L_data.eps", format="eps", dpi=2000, bbox_inches="tight", transparent=True)
fig.savefig("Rh_least_squares_D_L_data.pdf", format="pdf", dpi=2000, bbox_inches="tight", transparent=True)
