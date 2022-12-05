#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 12:47:21 2022

@author: mike
"""
print()
print("This routine models the 156 gold SNe Ia data and the position of the earth, as D_L vs expansion factor, of Riess et al. 2004. The model used presumes the flat, LCDM integrated solution with two parameters, the standard model with the Hubble constant, Hu, and the normalized matter density (\u03A9m,O_m). The Python 3 least_squares routine with robust loss='cauchy' is used. Integration is necessary because the FLRW model cannot be solved exactly for D_L vs. expansion factor when the influence of dark energy is considered.")

#import the necessary modules (libraries)
import csv  #required to open the .csv file for reading
import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from scipy import integrate as intg

#open data file
with open("Gold_Riess_D_L_2004.csv","r") as i:
    rawdata = list(csv.reader(i, delimiter = ","))

# ignore the string data in the first row    
exampledata = np.array(rawdata[1:],dtype=float)

x = exampledata[:,3]
ydata = exampledata[:,4]
errors = exampledata[:,7] #the standard deviations are not used in the regression routine but only in the plot.

#litesped value for the speed of light
litesped = 299793

#Hu represents the Hubble constant and O_m represents the normalized matter density. Here Hu is the initial guess, to be replaced later in this program by a computer estimated value. 
Hu=70.0
#The long equation on the right-hand side is the FLRW model for a Universe with the Hubble expansion, matter density and spacetime - no dark energy.

#We first have to write the first two functions, integr and func2 dealing with the numerical integration
def integr(x,O_m):
    return intg.quad(lambda t: 1/(t*np.sqrt((O_m/t)+(1-O_m)**2)), x, 1)[0]
    
def func2(x, O_m):
    return np.asarray([integr(xx,O_m) for xx in x])

def func1(params, x, y):
    Hu = params[0] #to identify parameter #1
    O_m = params[1] #to identify parameter #2
    residual = ydata-(litesped/(x*Hu))*np.sinh(func2(x,O_m))
    return residual

params=[70,0.35] #Initial guesses: Hubble constant value and normalized matter density: the least_squares regression routine used; x0, the parameter guesses; jac, the math routine used; bounds, allowed lower, then upper bounds for params; loss, treatment of outlier data pairs; args, x and y data.

result2 = least_squares(func1, x0=params, jac='3-point',bounds=((50,0.001),(80,0.99)),loss='cauchy',args=(x, ydata))

#extracting the Hu and O_m parameters and rounding the values
Hu,O_m = result2.x
Hubble = np.round(Hu,2)
mattdens = np.round(O_m,3)

#the yfit1 function used to estimate goodness of fit using the Hu and O_m guesstimates.
yfit1 = (litesped/(x*Hu))*np.sinh(func2(x,O_m))

#calculation of residuals
residuals = ydata - yfit1
ss_res_lsq = np.sum(residuals**2)
ss_tot_lsq = np.sum((ydata-np.mean(ydata))**2)

# here P is the number of free parameters
P=2
#r_squared = 1 - (ss_res/ss_tot)
r_squared_lsq = 1 - (ss_res_lsq/ss_tot_lsq)
#r2 = round(r_squared,3)
r2_lsq = round(r_squared_lsq,3)
r2adjusted = round(1-(((1-r2_lsq)*(len(ydata)-1))/(len(ydata)-P-1)),3)

#This shall be used to estimate the covariance matrix of the parameters using the following formula: Sigma = (J'J)^-1.
J = result2.jac
cov = np.linalg.inv(J.T.dot(J))
#To find the variance of the parameters one can then use:

var = np.sqrt(np.diagonal(cov))
StndDev,O_mStndDev = var
normSD = round(StndDev,2)
normO_mStndDev = round(O_mStndDev,3)

#calculate the statistical fitness, using 157 as the number of data pairs and 2 as the degree of freedom (paramater count)

chisq = sum((ydata[1:-1]-yfit1[1:-1])**2/yfit1[1:-1])
chisquar = np.round(chisq,2)
#normalised chisquar is calculated from the number of data pairs (60) minus the number of free parameters (2). 
normchisquar = np.round((chisquar/(157-P)),2)
"""
#The BIC value is calculated as; BIC from Bayesian Information Criteria
BIC = 157 * np.log10(chisq/157) + P*np.log10(157)
normBIC = round(BIC,2)
"""
#Plot of data and fitted curve
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
plt.plot(x, yfit1, color="orange", label="$\Lambda$CDM model")
plt.xlabel('Expansion factor, \u03BE', fontsize = 18)
plt.ylabel('SNe Ia distance (Mpc)', fontsize = 18)
plt.title("\u039BCDM Model, Distance vs. Exp. fact.", fontsize = 18)
plt.legend(loc='best', fancybox=True, shadow=False)
plt.show()

print()
print("The estimated Hubble constant and S.D. are:", Hubble, ","  , normSD)
print("The estimated \u03A9m and S.D. are:", mattdens, ",", normO_mStndDev)
print ('The calculated r\u00b2_adj is:'+str(r2adjusted))
print("The goodness of fit, \u03C7\u00b2, guesstimate is:", chisquar)
print("The reduced goodness of fit, \u03C7\u00b2, estimate is:", normchisquar)
#print("Reduced \u03C7\u00b2 = \u03C7\u00b2/(N-P), where N are the number of data pairs and P is the paramter count.")
#print("The guesstimate for BIC is: ", normBIC)
#print("BIC is shorthand for Bayesian Information Criteria")

#Routines to save figues in eps and pdf formats
fig.savefig("LCDM_least_squares_D_L_data.eps", format="eps", dpi=2000, bbox_inches="tight", transparent=True)
fig.savefig("LCDM_least_squares_D_L_data.pdf", format="pdf", dpi=2000, bbox_inches="tight", transparent=True)

 