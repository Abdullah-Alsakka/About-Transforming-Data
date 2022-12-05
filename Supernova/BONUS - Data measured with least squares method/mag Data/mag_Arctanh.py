#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 12:47:21 2022

@author: mike
"""
print()
print("This routine models the gold SNe Ia data, as mag vs z, of Riess et al. 2004 with 156 data pairs. The model used is the mag arctanh analytical solution with two parameters, the Hubble constant, Hu, and the normalized matter density (\u03A9m,O_m) with no influence of dark energy. The Python 3 least_squares regression routine is used with the robust loss='cauchy' specification. Integration is unnecessary because the FLRW model can be solved exactly for mag (ydata) vs redshift.")

#import the necessary modules (libraries)
import csv  #required to open the .csv file for reading
import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

#open data file
with open("Gold_Riess_mag_2004.csv","r") as i:
    rawdata = list(csv.reader(i, delimiter = ","))
    
#ignores the first row which is only string data    
exampledata = np.array(rawdata[1:],dtype=float)

x = exampledata[:,1]
ydata = exampledata[:,2]
errors = exampledata[:,3] #The standard deviations are not used to calculate the best fit but only for plot display.

#litesped value for the speed of light
litesped = 299793

#Hu represents the Hubble constant and O_m represents the normalized matter density. These are the initial guesses, to be replaced later in this program by a computer estimated values. 
Hu=70.0
O_m = 0.02

#The long equation on the right-hand side is the FLRW model for a Universe with the Hubble expansion, matter density and spacetime - no dark energy.
def func1(params, x, y):
    Hu = params[0] #the 2 lines to identify 2 parameters
    O_m = params[1] #the 2 lines to identify 2 parameters
    residual = ydata-(5*np.log10((litesped*(1+x)/(Hu*np.sqrt(abs(1-O_m))))*np.sinh(2*(np.arctanh(np.sqrt(abs(1-O_m)))-np.arctanh(np.sqrt(abs(1-O_m))/np.sqrt((O_m/(1/(1+x)))+(1-O_m))))))+25)
    return residual

params=[75,0.02] #Initial guesses again: Hubble constant value and normalized matter density

#The least_squares regression routine used; x0, the parameter guesses; jac, the regression routine chosen; bounds, allowed lower, then upper bounds for params; loss, 'robust' treatment of outlier data pairs; args, x and y data. 
result2 = least_squares(func1, x0=params, jac='3-point',bounds=((50,0.001),(80,0.99)),loss='cauchy',args=(x, ydata))

#presenting the Hu and O_m parameters and rounding the values
Hu,O_m = result2.x
Hubble = np.round(Hu,2)
mattdens = np.round(O_m,3)

#the yfit1 function used to estimate goodness of fit and r**2 values with the Hu and O_m estimates
yfit1 = 5*np.log10((litesped*(1+x)/(Hu*np.sqrt(abs(1-O_m))))*np.sinh(2*(np.arctanh(np.sqrt(abs(1-O_m)))-np.arctanh(np.sqrt(abs(1-O_m))/np.sqrt((O_m/(1/(1+x)))+(1-O_m))))))+25

#calculation of residuals
residuals = ydata - yfit1
ss_res_lsq = np.sum(residuals**2)
ss_tot_lsq = np.sum((ydata-np.mean(ydata))**2)

P=2
#r squared with the parameter count of P=2, round to 3 digits
r_squared_lsq = 1 - (ss_res_lsq/ss_tot_lsq)
r2_lsq = round(r_squared_lsq,4)
r2adjusted = round(1-(((1-r2_lsq)*(len(ydata)-1))/(len(ydata)-P-1)),3)

#This can be used to estimate the parameter S.D. using the following formula: Sigma = (J'J)^-1.
J = result2.jac
cov = np.linalg.inv(J.T.dot(J))

#To find the S.D. of the parameters one can then use:
var = np.sqrt(np.diagonal(cov))
StndDev,O_mStndDev = var
normSD = round(StndDev,2)
normO_mStndDev = round(O_mStndDev,3)

#calculate the statistical fitness, using 156 as the number of data pairs and 2 as the degree of freedom (parameter count)
chisq = sum(((ydata-yfit1)**2)/yfit1)
chisquar = np.round(chisq,3)

#normalized chisquar is calculated from the number of data pairs (156) minus the number of free parameters (2) with P=2. 
normchisquar = np.round((chisquar/(156-P)),4)
"""
#The BIC value is calculated as; BIC from Bayesian Information Criteria
BIC = 156 * np.log10(chisq/156) + 2*np.log10(156)
normBIC = round(BIC,2)
"""
#Plot of data and best curve regression
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['lines.linewidth'] = 3
plt.xlim(0,2)
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
plt.plot(x, yfit1, color="green", label="magarctanh model")
plt.xlabel('Redshift z', fontsize = 18)
plt.ylabel('mag (no units)', fontsize = 18)
plt.legend(loc='best', fancybox=True, shadow=False)
plt.title("magarctanh model with mag vs. z data", fontsize = 18)
plt.show()

print()
print("The estimated Hubble constant and S.D are: ", Hubble,",", normSD)
print("The estimated \u03A9m and S.D. are: ", mattdens,",",normO_mStndDev)
print ('The calculated r\u00b2_adj is:'+str(r2adjusted))
print("The goodness of fit, \u03C7\u00b2, guesstimate is: ", chisquar)
print("The reduced goodness of fit, \u03C7\u00b2, estimate is: ", normchisquar)
#print("Reduced \u03C7\u00b2 = \u03C7\u00b2/(N-P), where N are the number of data pairs and P is the paramter count.")
#print("The guesstimate for BIC is: ", normBIC)
#print("BIC is shorthand for Bayesian Information Criteria")

#Routines to save figues in eps and pdf formats
fig.savefig("magArctan_least_squares.eps", format="eps", dpi=2000, bbox_inches="tight", transparent=True)
fig.savefig("magArctan_least_squares.pdf", format="pdf", dpi=2000, bbox_inches="tight", transparent=True)

 