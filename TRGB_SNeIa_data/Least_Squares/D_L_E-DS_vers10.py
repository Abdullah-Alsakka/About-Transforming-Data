#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 12:47:21 2022

@author: mike


This routine models the SNe Ia data, as D_L vs. z, of Brout et al. 2022, "The Pantheon+ Type Ia 
Supernova Sample: The Full Dataset and Light-Curve Release", arXIV 2112.03863 and the TRGB data of 
Freedman et al. 2019, "The Carnegie-Chicago Hubble Program. VIII." Apj vol. 882, 34. The model used is the 
Einstein-DeSitter model with only a single parameter, the Hubble constant. The Python 3 least_squares 
robust regression routine is employed and no estimate can be made for matter density or dark energy.")
"""
import csv
import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

#open data file
with open("DATAE.csv","r") as i:
    rawdata = list(csv.reader(i, delimiter = ","))
    
exampledata = np.array(rawdata[1:],dtype=float)

x = exampledata[:,0]
ydata = exampledata[:,4]
errors = exampledata[:,7]

litesped = 299793
N=1720
P=1
#Hu represents the Hubble constant params=[70] #Initial guess of the Hubble constant value. 
#The Einstein-deSitter model is the right-hand term of the "residual" equation.
def func1(params, x, y):
    Hu = params[0] # params[1], params[2]
    residual = ydata-((litesped*(1+x)/(Hu))*np.sinh(x/(1+x)))
    return residual

params=Hu
#Application of the least_squares regression routine
result2 = least_squares(func1, x0=params, jac='3-point',bounds=(50,100),method = 'trf', loss='linear', args=(x, ydata))
Hu, = result2.x
#yfit1 now decribes the E-DeS model
yfit1 = ((litesped*(1+x)/(Hu))*np.sinh(x/(1+x)))
Hubble = np.round(Hu,2)

#calculation of residuals and the squared residuals 
residuals = ydata - yfit1
ss_res_lsq = np.sum(residuals**2)
ss_tot_lsq = np.sum((ydata-np.mean(ydata))**2)

#r_squared = 1 - (ss_res/ss_tot)
r_squared_lsq = 1 - (ss_res_lsq/ss_tot_lsq)
#r2 = round(r_squared,3)
r2_lsq = round(r_squared_lsq,3)

#Below will be used to estimate the Covariance Matrix of the parameters using the following formula: Sigma = (J'J)^-1.
J = result2.jac
cov = np.linalg.inv(J.T.dot(J))

#To find the variance of the parameters one can then use:
var = np.sqrt(np.diagonal(cov))
StndDev, = var
#We write a comma after StndDev, because we are extracting this value from tuple. Below is just rounding off the value of the standard deviation to something one might believe - only three digits.
normSD = round(StndDev,3)

#calculate the statistical fitness, using 1720 as the number of data pairs and 1 as the degree of freedom (paramater count). Note the values of the errors are those of the tablulated SNe Ia distances.
chisq = sum((ydata-yfit1)**2/(errors**2))
chisquar = np.round(chisq,2)

#normalised chisquar is calculated from the number of data pairs (1720) minus the number of free parameters (1). 
normchisquar = np.round((chisquar/(N-1)),2)

#The BIC value is calculated as; BIC from Bayesian Information Criteria
BIC = 1720 * np.log10(chisq/N) + 1*np.log10(N)
normBIC = round(BIC,2)

#Calculation of the weighted F-statistic
SSEw = sum((1/errors)*(residuals)**2)
SSM = sum((1/errors)*(ydata - np.mean(ydata))**2) 
MSR = (SSM - SSEw)/(P)
MSE = SSEw/(N-P)
Fstat = MSR/MSE
rFstat = round(Fstat,1)

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['lines.linewidth'] = 3
plt.xlim(0.5,1.0)
plt.ylim(0.0,9000)
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
plt.plot(x, yfit1, color="orange", label="Einstein-DeSitter model")
plt.xlabel('Redshift, z', fontsize = 18)
plt.ylabel('Luminosity distance (Mpc)', fontsize = 18)
plt.title("Einstein-DeSitter model", fontsize = 18)
plt.legend(loc='best', fancybox=True, shadow=False)
plt.show()

print()
print("Loss function = linear")
print("The estimated Hubble constant is: ", Hubble)
print("The S.D. of the Hubble constant is ", normSD)
print()
print ('The r\u00b2 is calculated as:'+str(r2_lsq))
#print("The \u03C7\u00b2 guesstimate is: ", chisquar)
print("The reduced \u03C7\u00b2 guesstimate is: ", normchisquar)
print("Reduced \u03C7\u00b2 = \u03C7\u00b2/(N-P), where N are the number of data pairs and P is the parameter count.")
print("The guesstimate for BIC is: ", normBIC)
print("The value for the F-statistic is:" , rFstat )

#Routines to save figues in eps and pdf formats
fig.savefig("E-DS_least_squares_D_L_data.eps", format="eps", dpi=2000, bbox_inches="tight", transparent=True)
fig.savefig("E-DS_least_squares_D_L_data.pdf", format="pdf", dpi=2000, bbox_inches="tight", transparent=True)
