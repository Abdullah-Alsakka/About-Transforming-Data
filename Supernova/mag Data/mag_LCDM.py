#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 17:53:20 2022

@author: mike
"""
print("A test of the Friedmann-Lemaitre-Robertson-Walker (FLRW) model using the curve_fit regression routine of Python scipy uses the SNe Ia data, as mag vs redshift (z), from the Gold data set of Riess, A.G. et al. 'Type Ia Supernova Discoveries at z> 1 from the Hubble Space Telescope: Evidence for Past Deceleration and Constraints on Dark Energy Evolution' Astrophys. J. vol. 607(2), 665-687 (2004). This variation of the LCDM model used here has two parameters: Hubble constant, Hu, normalised matter density, O_m; the cosmological constant is the remainder of information in a universe with flat geometry.")
print("This is the magLCDM model, typically knwon as the standard model of coslmology.")

# import the data file and the Python 3 libraries
import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import integrate as intg

#open data file
with open("Gold_Riess_mag_2004.csv","r") as i:
    rawdata = list(csv.reader(i, delimiter = ","))
# ignore the top row    
exampledata = np.array(rawdata[1:],dtype=float)

xdata = exampledata[:,1]
ydata = exampledata[:,2]
error = exampledata[:,3]

# initial guess for the normailized matter density, O_m
O_m = 0.30

# where t is the "dummy" variable during integration
def integr(x,O_m):
    return intg.quad(lambda t: (1/(np.sqrt(((1+t)**2)*(1+O_m*t) - t*(2+t)*(1-O_m)))), 0, x)[0]
    
def func2(x, O_m):
    return np.asarray([integr(xx,O_m) for xx in x]) 

litesped = 299793

def func3(x,Hu,O_m):
    return 5*(np.log10((litesped*(1+x)/Hu)*np.sinh(func2(x,O_m)))) + 25

# guesses for the Hubble constant, Hu, and the normlaized matter density, O_m
init_guess = np.array([70,0.30])

# allowed range for the two parameters
bnds=([50,0.01],[80,1.0])

params, pcov = curve_fit(func3, xdata, ydata,p0 = init_guess, bounds = bnds, sigma = error, absolute_sigma = False)

#Extracting and rounding the parameter values
ans_Hu, ans_O_m = params
rans_Hu = round(ans_Hu,2)
rans_O_m = round(ans_O_m,3)

#Extracting and rounding the estimated oarameter standard deviations.
perr = np.sqrt(np.diag(pcov))
SD_Hu, SD_O_m = perr
rSD_Hu = round(SD_Hu,2)
rSD_O_m = round(SD_O_m,3)

#estimating the goodness of fit
chisq = sum((ydata - func3(xdata,ans_Hu,ans_O_m))**2/func3(xdata,ans_Hu,ans_O_m))
chisquar = round(chisq,2)
#normalised chisquar is calculated for 157 data pairs with P the parameter count as
P=2
normchisquar = round((chisquar/(157-P)),2)

#calculation of residuals,again
residuals = ydata - func3(xdata,ans_Hu,ans_O_m)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((ydata-np.mean(ydata))**2)

#r squared calculation
r_squared = 1 - (ss_res/ss_tot)
r2 = round(r_squared,3)
r2adjusted = round(1-(((1-r2)*(len(ydata)-1))/(len(ydata)-P-1)),3)

#plot of imported data and model
plt.figure(1,dpi=240)
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['lines.linewidth'] = 3
plt.xlim(0.0,1.8)
plt.ylim(32.0,46.0)
plt.xscale("linear")
plt.yscale("linear")
fig, ax = plt.subplots()
ax.tick_params(axis="y", direction='in', length=14)
ax.tick_params(axis="x", direction='in', length=10)
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(3)
    ax.tick_params(width=3)
plt.errorbar(xdata, ydata, yerr=error, fmt='.k', capsize = 4)
plt.plot(xdata, func3(xdata,ans_Hu,ans_O_m), color = "green", label = "standard model")
plt.xlabel("Redshift z", fontsize = 18)
plt.ylabel("mag (no units)", fontsize = 18)
plt.title("standard model, mag vs. redshift z", fontsize = 18)
plt.legend(loc='best', fancybox=True, shadow=False)

#print results
print()
print("The calculated Hubble constant and S.D. are: ", rans_Hu, ",",rSD_Hu)
print("The calculated normalised matter density and S.D. are: ",rans_O_m, ",",rSD_O_m )
print("The adjusted r\u00b2 is calculated to be: ",r2adjusted)
print("The goodness of fit \u03C7\u00b2 is: ", chisquar)
print("The reduced goodness of fit \u03C7\u00b2 is: ", normchisquar)
print("Reduced \u03C7\u00b2 = \u03C7\u00b2/(N-P), where N are the number of data pairs and P is the parameter count.")

#commands to save plots in two different formats
fig.savefig("flatLCDM_mag_data.eps", format="eps", dpi=2000, bbox_inches="tight", transparent=True)
fig.savefig("flatLCDM_mag_data.pdf", format="pdf", dpi=2000, bbox_inches="tight", transparent=True)

