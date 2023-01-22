#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 17:41:24 2022

@author: mike
"""
print()
print("This Python 3, scipy curve_fit regression routine uses the SNe Ia data, as mag vs redshift (z), of the gold data set from Riess, A.G. et al. 'Type Ia Supernova Discoveries at z>1 from the Hubble Space Telescope: Evidence for Past Deceleration and Constraints on Dark Energy Evolution' Astrophys. J. 607(2), 665-687 (2004). This is the arctanh, analytical solution to the Friedmann-Lemaitre-Roberston-Walker (FLRW) model with three parameters, the Hubble constant, Hu, the normalised matter density, O_m and the spacetime parameter, O_k. No estimation is possible for dark energy.")
print()

# import data and Python 3 library files
import numpy as np
import csv
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import math

# open data file
with open("Gold_Riess_mag_2004.csv","r") as i:
    rawdata = list(csv.reader(i, delimiter = ","))
    
# ignore the first row which are strings    
exampledata = np.array(rawdata[1:],dtype=float)

xdata = exampledata[:,1]
ydata = exampledata[:,2]
error = exampledata[:,3]

# define the function - the model to be examined, where x represents the independent variable; b (Hubble constant), c (matter density), d spacetime density are the parameters to be estimated
def func(x,b,c,d):
    return (litesped*(1+x)/(b*np.sqrt(abs(d))))*np.sinh(2*(np.arctanh(np.sqrt(abs(d)))-np.arctanh(np.sqrt(abs(d))/np.sqrt((c/(1/(1+x)))+ (d)))))

def func2(x,b,c,d):
    return 5*np.log10(func(x,b,c,d)) + 25

# the initial guesses of the model parameters
p0=[70.0,0.01, 0.99]

# specify the constant speed of light
litesped = 299793

# curve fit the model to the data, the bnds are the lower and upper bounds for the two parameters
bnds = ([50.0, 0.0001, 0.0001],[80.0,1.0,1.0])
params,pcov = curve_fit(func2,xdata,ydata, p0, bounds = bnds, sigma = error, absolute_sigma = False)

# extracting the two parameter values and rounding the values
ans_b, ans_c, ans_d = params
rans_b = round(ans_b, 2)
rans_c = round(ans_c, 4)
rans_d = round(ans_d, 4)

# specify the parameters for the plot of the model
b=ans_b
c=ans_c
d=ans_d

# evaluate the function
func2data = func2(xdata,b,c,d)

# extracting the three standard deviations and rounding the values
perr = np.sqrt(np.diag(pcov))
ans_bSD, ans_cSD, ans_dSD = perr
rans_bSD = round(ans_bSD,2)
rans_cSD = round(ans_cSD,4)
rans_dSD = round(ans_dSD,4)

# normalised chisquar where P the parameter number (3) and N the number of data pairs as 
P=3
N=156
e = 2.718281

# estimating the goodness of fit
chisq = sum((ydata - func2(xdata,ans_b,ans_c,ans_d))**2/func2(xdata,ans_b,ans_c,ans_d))
chisquar = round(chisq,5)

#Calculate the chi^2 according to astronomers
newxsqrd = sum(((ydata - func2(xdata,ans_b,ans_c,ans_d))**2)/(error**2))
newxsqrded = np.round(newxsqrd/(N-P),2)

normchisquar = round((chisquar/(N-P)),4)

#The BIC value is calculated as
BIC = N * math.log(e,(newxsqrd/N)) + P*math.log(e,N)
normBIC = round(BIC,2)

# calculation of residuals
residuals = ydata - func2(xdata,ans_b,ans_c,ans_d)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((ydata-np.mean(ydata))**2)

# r squared calculation
r_squared = 1 - (ss_res/ss_tot)
r2 = round(r_squared,3)
r2adjusted = round(1-(((1-r2)*(len(ydata)-1))/(len(ydata)-P-1)),3)

#plt.plot(xdata,ydata)
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['lines.linewidth'] = 3
plt.xlim(0.0,1.8)
plt.ylim(32.0,46.0)
#plt.xscale("linear")
#plt.yscale("linear")
fig, ax = plt.subplots()
ax.tick_params(axis="y", direction='in', length=14)
ax.tick_params(axis="x", direction='in', length=10)
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(3)
    ax.tick_params(width=3)
plt.errorbar(xdata, ydata, yerr=error, fmt='.k', capsize = 4)
plt.xlabel("Redshift z", fontsize = 18)
plt.ylabel("mag (no units)", fontsize = 18)
#plt.title("magarctanh model, mag vs. redshift z", fontsize = 18)
plt.plot(xdata, func2(xdata,ans_b,ans_c,ans_d), color = "green", label = "3Pmagarctanh model")
plt.legend(loc='best', fancybox=True, shadow=False)
plt.show()

#print results
print()
print("The calculated Hubble constant with S.D. is: ", rans_b, ",",rans_bSD )
print("The normalised matter density with S.D. is: ", rans_c,"," , rans_cSD)
print("The spacetime density with S.D. is: ", rans_d,"," ,rans_dSD)
print()
print("The adjusted r\u00b2 is calculated to be: ",r2adjusted)
print("The reduced goodness of fit \u03C7\u00b2 is: ", newxsqrded)
print("The estimate for BIC is: ", normBIC)

#Routines to save figues in eps and pdf formats
fig.savefig("3P_Arctanh_mag_data.eps", format="eps", dpi=2000, bbox_inches="tight", transparent=True)
fig.savefig("3P_Arctanh_mag_data.pdf", format="pdf", dpi=2000, bbox_inches="tight", transparent=True)


















