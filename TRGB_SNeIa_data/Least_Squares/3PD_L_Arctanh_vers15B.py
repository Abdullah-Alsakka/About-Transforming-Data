#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 12:47:21 2022
This code refers to Eq. 16 of the manuscript.
@author: mike
"""
print()
print("This least_squares regression routine of Python scipy, uses the data, converted to luminosity distances, D_L, vs redshift, z, from Brout et al. 2022, 'The Pantheon+ Analysis: Cosmological Constraints' ApJ, vol. 938, 110 and the 18 TRGB data from W. Freedman, 2019 'The Carnegie-Chicago Hubble Program. VIII.' ApJ, vol. 882, 34. The model used is the arctanh analytical solution with three parameters, the Hubble constant, Hu, the normalized matter density (\u03A9m,O_m) and the spacetime parameter (\u03A9k,O_k), but no value for dark energy. The Python 3 least_squares robust regression specifies the loss='cauchy' or 'linear' routine. Integration is unnecessary because the FLRW model can be solved exactly using D_L (as the ydata) vs redshift.")

#import the necessary modules (libraries)
import csv  #required to open the .csv file for reading
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

#litesped value for the speed of light
litesped = 299793

#Hu = the Hubble constant and O_m = the normalized matter density and O_k = the spacetime parameter. Here Hu, O_m and O_k are the initial guesses, to be replaced later by computer estimated values. 
Hu=70.0
O_m = 0.03
O_k = 0.95
#Note the initial guesses, O_m+O_k cannot = (sum) to be precisely 1, otherwise the Python 3 regression fails.

#The long equation on the right-hand side is the FLRW model for a Universe with the Hubble expansion, matter density and spacetime, but no dark energy.
def func1(params, x, y):
    Hu = params[0] #the 3 lines to identify 3 parameters
    O_m = params[1] #the 3 lines to identify 3 parameters
    O_k = params[2] #the 3 lines to identify 3 parameters
    residual = ydata-((litesped*(1+x)/(Hu*np.sqrt(abs(O_k))))*np.sinh(2*(np.arctanh(np.sqrt(abs(O_k)))-np.arctanh(np.sqrt(abs(O_k))/np.sqrt((O_m*(1+x))+ (O_k))))))
    return residual

params=[70,0.03,0.95] #Initial guesses: Hubble constant value and normalized matter density: the least_squares regression routine used; x0, the parameter guesses; jac, the math routine used; bounds, allowed lower, then upper bounds for params; loss='cauchy', treatment of outlier data pairs; args, x and y data. 
result2 = least_squares(func1, x0=params, jac='3-point',bounds=((50,0.0001,0.0001),(80,1.00,1.00)),loss='cauchy',args=(x, ydata))

#presenting the Hu, O_m and O_k parameters and rounding the values
Hu,O_m,O_k = result2.x
Hubble = np.round(Hu,2)
mattdens = np.round(O_m,5)
spacetime = np.round(O_k,5)

#the yfit1 function used to estimate goodness of fit with the Hu, O_m and O_k estimates
yfit1 = ((litesped*(1+x)/(Hu*np.sqrt(abs(O_k))))*np.sinh(2*(np.arctanh(np.sqrt(abs(O_k)))-np.arctanh(np.sqrt(abs(O_k))/np.sqrt((O_m*(1+x))+ (O_k))))))

#calculation of residuals
resids = ydata - yfit1
#residuals_lsq = data - data_fit_lsq
ss_res_lsq = np.sum(resids**2)
ss_tot_lsq = np.sum((ydata-np.mean(ydata))**2)

#r_squared = 1 - (ss_res/ss_tot)
r_squared_lsq = 1 - (ss_res_lsq/ss_tot_lsq)
#r2 = round(r_squared,3)
r2_lsq = round(r_squared_lsq,4)

#We use P=3 for the parameter count, rounded to 3 digits
P=3
N=1720
e = 2.718281
r2adjusted = round(1-(((1-r2_lsq)*(len(ydata)-1))/(len(ydata)-P-1)),4)

#This can be used to estimate the Covariance Matrix of the parameters using the following formula: Sigma = (J'J)^-1.
J = result2.jac
cov = np.linalg.inv(J.T.dot(J))

#To find the variance of the parameters one then uses:
perr = np.sqrt(np.diagonal(cov))
StndDev,O_mStndDev,O_kStndDev = perr
normSD = round(StndDev,3)
normO_mStndDev = round(O_mStndDev,5)
normO_kStndDev = round(O_kStndDev,5)

#calculate the statistical fitness, using 1720 as the number of data pairs and 2 as the degree of freedom (paramater count)
chisq = sum(((ydata-yfit1)**2)/(errors**2))
chisquar = np.round(chisq,4)

#calculate the statistical fitness, in the more usual manner, with 1720 as the number of data pairs and 3 as the degree of freedom (paramater count)
chisqed = sum(((ydata[1:-1]-yfit1[1:-1])**2)/yfit1[1:-1])
chisquared = np.round(chisqed,4)

#normalised chisquar is calculated from the number of data pairs (1720) minus the number of free parameters (2). 
normchisquar = np.round((chisquar/(N-P)),3)
normchisquared = np.round((chisquared/(N-P)),3)

#The  BIC value is calculated as
SSE = sum((ydata-yfit1)**2)
alt_BIC = N * math.log(e,(SSE/N)) + P*math.log(e,N)
alt_BIC = round(alt_BIC,2)

#Calculation of the weighted F-statistic
SSEw = sum((1/errors)*(resids)**2)
SSM = sum((1/errors)*(ydata - np.mean(ydata))**2) 
MSR = (SSM - SSEw)/(P)
MSE = SSEw/(N-P)
Fstat = MSR/MSE
rFstat = round(Fstat,1)

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['lines.linewidth'] = 3
plt.xlim(0.3,1)
plt.ylim(0,16000)
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
plt.plot(x, yfit1, color="orange", label="3PArctanh model")
plt.xlabel('Redshift, z', fontsize = 18)
plt.ylabel('Luminosity distance (Mpc)', fontsize = 16)
plt.title("3P_Arctanh model, D_L vs. Exp. fact. data", fontsize = 18)
plt.legend(loc='best', fancybox=True, shadow=False)
plt.show()

print()
print("The estimated Hubble constant and S.D. are:", Hubble, ",",normSD)
print("The estimated \u03A9m and S.D. are:", mattdens, ",",normO_mStndDev)
print("The estimated \u03A9k and S.D. are:", spacetime, ",",normO_kStndDev)
print()
print ("The calculated r\u00b2_adj is:", r2adjusted)
print("The reduced goodness of fit according to astronomers, \u03C7\u00b2, estimate is:", normchisquar)
print("The reduced goodness of fit according to normal people, \u03C7\u00b2, is:", normchisquared)
print("The estimate for BIC is: ", alt_BIC)
print("The estimate for the reduced F stat is: ", rFstat)


#Routines to save figues in eps and pdf formats
fig.savefig("3PArctan_least_squares_D_L_data.eps", format="eps", dpi=2000, bbox_inches="tight", transparent=True)
fig.savefig("3PArctan_least_squares_D_L_data.pdf", format="pdf", dpi=2000, bbox_inches="tight", transparent=True)

 