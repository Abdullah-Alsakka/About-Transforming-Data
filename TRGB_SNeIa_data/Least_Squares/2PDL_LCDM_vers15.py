#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 12:47:21 2022

@author: mike



This curve_fit regression routine of Python scipy, uses the data, as D_L vs redshift (z), from Brout et al. 2022,
The Pantheon+ Analysis: Cosmological Constraints' Astrophys. J. vol. 938, 110 and the 18 TRGB data of W. Freedman 2019
'The Carnegie-Chicago Hubble Program. VIII.' vol. 882, 34. The model used is the flat, LCDM integrated solution,
the standard model of cosmology, with two parameters, the Hubble constant, Hu, and the normalized matter density
\u03A9m,O_m). The value for the normalized cosmologial constant can be calculated from these results.
The least_squares Python 3 regression is used specifying the loss='cauchy' or linear robust function.
Integration is necessary because the FLRW model cannot be solved exactly with D_L vs. redshift z data.
"""
print()
#import the necessary modules (libraries)
import csv  #required to open the .csv file for reading
import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from scipy import integrate as intg
import math

#open data file
with open("DATAE.csv","r") as i:
    rawdata = list(csv.reader(i, delimiter = ","))
    
exampledata = np.array(rawdata[1:],dtype=float)

x = exampledata[:,0]
ydata = exampledata[:,4]
errors = exampledata[:,7] #the standard deviations of the ydata are not used in the regression but only for display on the plot.

#litesped value for the speed of light
litesped = 299793

#Hu represents the Hubble constant and O_m represents the normalized matter density. Here Hu is the initial guess, to be replaced later in this program by a computer estimated value. 
Hu=70.0

#We have to write our integration functions, the first two functions, integr and func2 deal with the numerical integration
def integr(x,O_m):
    return intg.quad(lambda t:1/(np.sqrt(((1+t)**2)*(1+O_m*t)- (t*(2+t)*(1-O_m)))),0,x)[0]
    
def func2(x,O_m):
    return np.asarray([integr(xx,O_m) for xx in x])

litesped = 299793

# func1 translates the integrated values into the values useful for regression.
# notice the np.sinh function is used here. This may be removed if wished.
def func1(params, x, y):
    Hu = params[0] #to identify parameter #1
    O_m = params[1] #to identify parameter #2
    residual = ydata-(((litesped*(1+x))/Hu)*np.sinh((func2(x,O_m))))
    return residual

params=[70,0.35] #Initial guesses: Hubble constant value and normalized matter density: the least_squares regression routine used; x0, the parameter guesses; jac, the regression routine chosen; bounds, allowed lower, then upper bounds for params; loss, treatment of outlier data pairs; args, x and y data. 
result2 = least_squares(func1, x0=params, jac='3-point',bounds=((50,0.0001),(80,1.0)),loss='linear',args=(x, ydata))

#extracting the Hu and O_m parameters and rounding the values
Hu,O_m = result2.x
Hubble = np.round(Hu,2)
mattdens = np.round(O_m,3)

#the yfit1 function used to estimate goodness of fit using the Hu and O_m estimates.
yfit1 = ((litesped*(1+x)/Hu)*np.sinh(func2(x,O_m)))

#calculation of residuals
residuals = ydata - yfit1
ss_res_lsq = np.sum(residuals**2)
ss_tot_lsq = np.sum((ydata-np.mean(ydata))**2)

# We specify that the parameter count as 2
P=2
N=1720
e = 2.718281

#r_squared = 1 - (ss_res/ss_tot)
r_squared_lsq = 1 - (ss_res_lsq/ss_tot_lsq)
#r2 = round(r_squared,3)
r2_lsq = round(r_squared_lsq,3)
r2adjusted = round(1-(((1-r2_lsq)*(len(ydata)-1))/(len(ydata)-P-1)),3)

#This shall be used to estimate the covariance matrix of the parameters using the following formula: Sigma = (J'J)^-1.
J = result2.jac
cov = np.linalg.inv(J.T.dot(J))

#To find the variance of the parameters one can then use:
perr = np.sqrt(np.diagonal(cov))
StndDev,O_mStndDev = perr
normSD = round(StndDev,2)
normO_mStndDev = round(O_mStndDev,3)

#calculate the statistical fitness, according to astronomers, using 1701 as the number of data pairs and P=2 as the degree of freedom (paramater count)
chisq = sum(((ydata-yfit1)**2)/errors**2)
normchisquar = np.round((chisq/(N-P)),4)

#calculate the statistical usual fitness, using 1701 as the number of data pairs and 1 as the degree of freedom (paramater count).
chisqed = sum(((ydata-yfit1)**2)/(yfit1))
normchisquared = np.round((chisqed/(N-P)),4)

#The BIC value is calculated as
SSE = sum((ydata-yfit1)**2)
alt_BIC = N * math.log(e,(SSE/N)) + P*math.log(e,N)
alt_BIC = round(alt_BIC,2)

#Calculation of the weighted F-statistic
SSEw = sum((1/errors)*(residuals)**2)
SSM = sum((1/errors)*(ydata - np.mean(ydata))**2) 
MSR = (SSM - SSEw)/(P)
MSE = SSEw/(N-P)
Fstat = MSR/MSE
rFstat = round(Fstat,1)

#Plot of data and best curve fit.
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['lines.linewidth'] = 3
plt.xlim(0.0,2.0)
plt.ylim(32.0,46.0)
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
plt.plot(x, yfit1, color="green", label="simple standard model")
plt.xlabel('Redshift z', fontsize = 18)
plt.ylabel('distance (Mpc)', fontsize = 18)
plt.legend(loc='best', fancybox=True, shadow=False)
plt.show()

print()
print("The estimated Hubble constant and S.D. are: ", Hubble, ","  , normSD)
print("The estimated \u03A9m and S.D. are: ", mattdens, ",", normO_mStndDev)
print()
print ('The calculated r\u00b2 is: '+str(r2adjusted))
print("The reduced goodness of fit, as per astronomers, \u03C7\u00b2, estimate is:", normchisquar)
print("The common reduced goodness of fit, \u03C7\u00b2, estimate is: ", normchisquared)
print("The estimate for BIC is: ", alt_BIC)
print("The estimate for the reduced F stat is: ", rFstat)

#Routines to save figues in eps and pdf formats
fig.savefig("2PLCDM_least_squares.eps", format="eps", dpi=2000, bbox_inches="tight", transparent=True)
fig.savefig("2PLCDM_least_squares.pdf", format="pdf", dpi=2000, bbox_inches="tight", transparent=True)

 
