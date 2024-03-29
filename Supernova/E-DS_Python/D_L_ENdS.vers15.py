#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 17:53:20 2022
This code solves equation A28 of Appendix, section A3.2, E-DS universe with \Omega_m=0 and \Omega_{\Lambda}>0
@author: Mike
"""
print("This curve_fit regression routine of Python scipy, uses the mag vs redshift (z) data  after calculations of the D_L and recession velocities from Brout et al. 2022, 'The Pantheon+ Analysis: Cosmological Constraints' Astrophys. J. vol. 938, 110. The measured distances, D_L, are plotted vs. the recession velocity (1/1+z). This variation of the Einstein-Newcomb-Desitter model has two parameters: Hubble constant, Hu, normalised matter density, O_m; in a universe with flat geometry. This model is reviewed in Yershov, V.N. 2023, Universe, vol. 9(5), 204. The corrected version of his model is used here with the Python curve_fit regression routine.")
print()
print("This is our D_L_ENdS model, a version of the Einstein-DeSitter model of cosmology.")

# import the data file and the Python 3 libraries
import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import integrate as intg
import math

# open data file
with open("DATA2B.csv","r") as i:
    rawdata = list(csv.reader(i, delimiter = ","))
    
# ignore the first row   
exampledata = np.array(rawdata[1:],dtype=float)

xdata = exampledata[:,1]
ydata = exampledata[:,4]
error = exampledata[:,7]

# initial guess for the normalized matter density, O_m
O_m = 0.05

# where t is the "dummy" variable during integration
def integr(x,O_m):
    return intg.quad(lambda t: (1/(np.sqrt((O_m/(t**3))-O_m+1))), x, 1)[0]
    
def func2(x, O_m):
    return np.asarray([integr(xx,O_m) for xx in x]) 

# specify the speed of light
litesped = 299793

# guesses for the Hubble constant, Hu, and the normalized matter density, O_m
init_guess = np.array([70,0.05])

def func3(x,Hu,O_m):
    return (litesped/(x*Hu)*(func2(x,O_m)))

# allowed range for the two parameters
bnds=([50,0.001],[80,1.0])

# fitting the model to the data, note that when choosing absolute_sigma = False the standard deviations (error) are normalized for use in weighted regression.
params, pcov = curve_fit(func3, xdata, ydata,p0 = init_guess, bounds = bnds, sigma = error, absolute_sigma = False)

# extracting and rounding the parameter, Hu and O_m, values
ans_Hu, ans_O_m = params
rans_Hu = round(ans_Hu,2)
rans_O_m = round(ans_O_m,3)

# extracting and rounding the estimated standard deviations.
perr = np.sqrt(np.diag(pcov))
SD_Hu, SD_O_m = perr
rSD_Hu = round(SD_Hu,2)
rSD_O_m = round(SD_O_m,3)

# normalised chisquar is calculated for 1702 data pairs with P the parameter count (2) as
P=2
N=1702
e=2.71828183

#Calculate the chi^2 according to astronomers
newxsqrd = sum(((ydata[1:-1] - func3(xdata,ans_Hu,ans_O_m)[1:-1])**2)/(error[1:-1]**2))
newxsqrded = round(newxsqrd/(N-P),2)

# estimating the goodness of fit in the more common manner
chisq = sum((ydata[1:-1] - func3(xdata,ans_Hu,ans_O_m)[1:-1])**2/func3(xdata,ans_Hu,ans_O_m)[1:-1])
normchisquar = round((chisq/(N-P)),2) #rounded to 2 digits

#The BIC value is calculated as 
SSE = sum((ydata - func3(xdata,ans_Hu,ans_O_m))**2)
alt_BIC = N*math.log(e,SSE/N) + P*math.log(e,N)
normalt_BIC = round(alt_BIC,2)

# calculation of residuals
residuals = ydata - func3(xdata,ans_Hu,ans_O_m)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((ydata-np.mean(ydata))**2)

#r squared calculation
r_squared = 1 - (ss_res/ss_tot)
r2 = round(r_squared,4)
r2adjusted = round(1-(((1-r2)*(len(ydata)-1))/(len(ydata)-P-1)),4)

#plot of imported data and model
plt.figure(1,dpi=240)
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['lines.linewidth'] = 3
plt.xlim(0.3,1.0)
plt.ylim(0.0,30000)
plt.xscale("linear")
plt.yscale("linear")
fig, ax = plt.subplots()
ax.tick_params(axis="y", direction='in', length=14)
ax.tick_params(axis="x", direction='in', length=10)
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(3)
    ax.tick_params(width=3)
plt.errorbar(xdata, ydata, yerr=error, fmt='.k', capsize = 4)
plt.plot(xdata, func3(xdata,ans_Hu,ans_O_m), color = "orange", label = "$D_L$ENdS model")
plt.xlabel("Expansion factor, \u03BE", fontsize = 18)
plt.ylabel("Luminosity distance (Mpc)", fontsize = 18)
plt.legend(loc='best', fancybox=True, shadow=False)

#print results
print()
print("The calculated Hubble constant and S.D. are: ", rans_Hu, ",",rSD_Hu)
print("The calculated normalised matter density and S.D. are: ",rans_O_m, ",",rSD_O_m )
print("Note that values for \u03A9k and \u03A9\u039B cannot be calculated with this model.")
print()
print("The adjusted r\u00b2 is calculated to be: ",r2adjusted)
print("The reduced goodness of fit, as per astronomers, \u03C7\u00b2 is: ", newxsqrded)
print("The reduced goodness of fit, as commonly done, \u03C7\u00b2 is: ", normchisquar)
print("The estimate for BIC is: ", normalt_BIC)
print()

#commands to save plots in two different formats
fig.savefig("DL_ENdS.eps", format="eps", dpi=2000, bbox_inches="tight", transparent=True)
fig.savefig("DL_ENdS.pdf", format="pdf", dpi=2000, bbox_inches="tight", transparent=True)
