#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 17:41:24 2022

@author: mike
"""
print("This curve_fit regression routine uses the SNe Ia data, as D_L vs expansion factor, calculated using the 
      Gold data set from Riess, A.G. et al. Type Ia Supernova Discoveries at z> 1 from the Hubble Space Telescope: 
      Evidence for Past Deceleration and Constraints on Dark Energy Evolution. Astrophys. J. vol. 607(2), 665-687 (2004). 
      The model selected is the Einstein-DeSitter analytical solution with only one parameter, the Hubble constant. 
      No estimation is possible for the matter density nor dark energy")

# import data and library files
import numpy as np
import csv
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

#open data file
with open("Gold_Riess_D_L_2004.csv","r") as i:
    rawdata = list(csv.reader(i, delimiter = ","))
    
exampledata = np.array(rawdata[1:],dtype=float)

xdata = exampledata[:,3]
ydata = exampledata[:,4]
error = exampledata[:,7]

#define the function, where b is the Hubble constant
def func(x,b):
    return (litesped/(x*b))*np.sinh(1-x)

#define the constants
litesped = 299793

#evaluate and plot function
funcdata = func(xdata,60) #The initial guess for the Hubble constant is the value just after xdata.
bnds = (50.0, 80.0)

#curve_fit the model to the data
params, pcov = curve_fit(func,xdata,ydata,bounds = bnds, sigma = error, absolute_sigma = False)
perr = np.sqrt(np.diag(pcov))
      
#unpacking the Hubble parameter and the estimated fit error
Hubble, = params
Error, = perr
#Rounding the above two values to 2 decimal places
normHubble = round(Hubble,2)
normError = round(Error,2)

#calculate the statistical fitness, using 158 as the number of data pairs and 1 as the degree of freedom (paramater count)
chisq = sum((ydata - func(xdata,Hubble))**2/(error**2))
chisquar = round(chisq,2)
#normalised chisquar is calculated as 
normchisquar = round((chisquar/(158-1)),2)

#calculation of residuals
residuals = ydata - func(xdata,Hubble)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((ydata-np.mean(ydata))**2)

#r squared calculation
r_squared = 1 - (ss_res/ss_tot)
r2 = round(r_squared,4)
r2adjusted = round(1-(((1-r2)*(len(ydata)-1))/(len(ydata)-1-1)),4)

#plot of data and results
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['lines.linewidth'] = 3
plt.xlim(0.5,1.0)
plt.ylim(0.0,15000)
plt.xscale("linear")
plt.yscale("linear")
fig, ax = plt.subplots()
ax.tick_params(axis="y", direction='in', length=8)
ax.tick_params(axis="x", direction='in', length=10)
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(3)
    ax.tick_params(width=3)
plt.xlabel("Expansion factor, \u03BE ", fontsize=18)
plt.ylabel("Luminosity distance (Mpc)", fontsize=18)
plt.rc('xtick', labelsize=14) 
plt.rc('ytick', labelsize=14)
plt.errorbar(xdata, ydata, yerr=error, fmt='.k', capsize = 5)
plt.title("E-DS Model, $D_L$ vs. Exp. fact.", fontsize = 18)
plt.plot(xdata, funcdata, color = "orange", label = "E-DS model")
plt.legend(loc='best', fancybox=True, shadow=False)
plt.show()

#print results
print()
print("The estimated Hubble constant is: ", normHubble)
print("The S.D. of the Hubble constant is ", normError)
print("The adjusted r\u00b2 is calculated to be: ",r2adjusted)
print("The calculated r\u00b2 is: ",r2)
print("The goodness of fit \u03C7\u00b2 guesstimate is: ", chisquar)
print("And reduced goodness of fit \u03C7\u00b2 guesstimate is: ", normchisquar)
print("Reduced \u03C7\u00b2 = \u03C7\u00b2/(N-P), where N are the number of data pairs and P is the parameter count.")
#print(f"D_L is {func(1,normHubble)} when expansion factor is 1")

#Routines to save figues in eps and pdf formats
fig.savefig("EinsteinDeSitter_D_L_data.eps", format="eps", dpi=2000, bbox_inches="tight", transparent=True)
fig.savefig("EinsteinDeSitter_D_L_data.pdf", format="pdf", dpi=2000, bbox_inches="tight", transparent=True)

