#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 17:41:24 2022

@author: mike
"""
print()
print("This curve_fit regression routine of Python scipy, using the SNe Ia data, as mag vs redshift (z), from the Gold data set from Riess, A.G. et al. Type Ia Supernova Discoveries at z> 1 from the Hubble Space Telescope: Evidence for Past Deceleration and Constraints on Dark Energy Evolution. Astrophys. J. vol. 607(2), 665-687 (2004). The model selected is the Einstein-DeSitter (E-DS) solution, not the Freidmann-Lemaitre-Robertson-Walker (FLRW) model. The E-DS model presents only one parameter, the Hubble constant. No estimation is possible for either the normalized matter density, which is presumed to be about 1, nor dark energy.")

# import the data and library files
import numpy as np
import csv
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

#open data file
with open("Gold_Riess_mag_2004.csv","r") as i:
    rawdata = list(csv.reader(i, delimiter = ","))
# ignore the top row    
exampledata = np.array(rawdata[1:],dtype=float)

xdata = exampledata[:,1]
ydata = exampledata[:,2]
error = exampledata[:,3]

#define the function, where Hu is the Hubble constant
def func(x,Hu):
    return (litesped*(1+x)/Hu)*np.sinh(x/(1+x))

def func2(x,Hu):
    return 5*np.log10(func(x,Hu)) + 25

#define the constants
litesped = 299793
#The intial guess for the Hubble constant
p0 = [70]
#evaluate and plot function where bnds are teh upper and lower limit allowed for Hu
funcdata = func2(xdata,p0)
bnds = (50.0, 80.0)

#curve fit data to model
params, pcov = curve_fit(func2,xdata,ydata,bounds = bnds, sigma = error, absolute_sigma = False)
perr = np.sqrt(np.diag(pcov))
#unpacking the Hubble parameter and the estimated fit error
Hubble, = params
Error, = perr
#Rounding the above two values to 2 decimal places
normHubble = round(Hubble,2)
normError = round(Error,2)

#calculate the statistical fitness, using 157 as the number of data pairs and 1 as the degree of freedom (parameter count)
chisq = sum((ydata - func2(xdata,Hubble))**2/func2(xdata,Hubble))
chisquar = round(chisq,2)
# here P is the number of parameters in the function
P=1
#normalised chisquar is calculated as 
normchisquar = round((chisquar/(157-P)),2)

#calculation of residuals
residuals = ydata - func2(xdata,Hubble)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((ydata-np.mean(ydata))**2)

#r squared calculation
r_squared = 1 - (ss_res/ss_tot)
r2 = round(r_squared,3)
r2adjusted = round(1-(((1-r2)*(len(ydata)-1))/(len(ydata)-P-1)),3)

#plot of imported data
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['lines.linewidth'] = 3
plt.xlim(0.0,1.5)
plt.ylim(32.0, 46.0)
plt.xscale("linear")
plt.yscale("linear")
fig, ax = plt.subplots()
ax.tick_params(axis="y", direction='in', length=14)
ax.tick_params(axis="x", direction='in', length=10)
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(3)
    ax.tick_params(width=3)
plt.xlabel("Redshift z", fontsize=18)
plt.ylabel("mag (no units)", fontsize=18)
plt.rc('xtick', labelsize=14) 
plt.rc('ytick', labelsize=14)
plt.errorbar(xdata, ydata, yerr=error, fmt='.k', capsize = 5)
plt.title("E-DS model, mag vs. redshift data", fontsize = 18)
plt.plot(xdata, funcdata, color = "orange", label = "magE-DS model")
plt.legend(loc='best', fancybox=True, shadow=False)
plt.show()

#print results
print("\n")
print("The estimated Hubble constant is: ", normHubble)
print("The S.D. of the Hubble constant is ", normError)
print("The adjusted r\u00b2 is calculated to be: ",r2adjusted)
print("The r\u00b2 is calculated to be: ",r2)
print("The goodness of fit \u03C7\u00b2 is: ", chisquar)
print("And reduced goodness of fit \u03C7\u00b2 is: ", normchisquar)
print("Reduced \u03C7\u00b2 = \u03C7\u00b2/(N-P), where N are the number of data pairs and P is the parameter count.")

#Routines to save figues in eps and pdf formats
fig.savefig("EinsteinDeSitter_mag_data.eps", format="eps", dpi=2000, bbox_inches="tight", transparent=True)
fig.savefig("EinsteinDeSitter_mag_data.pdf", format="pdf", dpi=2000, bbox_inches="tight", transparent=True)

