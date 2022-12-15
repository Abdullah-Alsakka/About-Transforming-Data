#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 17:53:20 2022

@author: mike
"""
print("This curve_fit regression routine uses the SNe Ia data, as D_L vs expansion factor, calculated using the Gold data set from Riess, A.G. et al. 'Type Ia Supernova Discoveries at z> 1 from the Hubble Space Telescope: Evidence for Past Deceleration and Constraints on Dark Energy Evolution' Astrophys. J. vol. 607(2), 665-687 (2004). The LCDM model used here requires numerical integration with two parameters, the Hubble constant, Hu and the normalised matter density, O_m in a flat geometry. An estimate of the normalized cosmological constant (dark energy) is possible presuming a Universe with flat geometry.")

import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import integrate as intg

# open data file
with open("Gold_Riess_D_L_2004.csv","r") as i:
    rawdata = list(csv.reader(i, delimiter = ","))
    
# ignore the first row   
exampledata = np.array(rawdata[1:],dtype=float)

xdata = exampledata[:,3]
ydata = exampledata[:,4]
error = exampledata[:,7]

#Model function
O_m = 0.30 #initial guess for matter density

def integr(x,O_m):
    return intg.quad(lambda t: 1/(t*(np.sqrt((O_m/t)+(1-O_m)*t**2))), x, 1)[0]
    
def func2(x, O_m):
    return np.asarray([integr(xx,O_m) for xx in x])

litesped = 299793

# Hu is the Hubble constant
def func3(x,Hu,O_m):
    return (litesped/(x*Hu))*np.sinh(func2(x,O_m))

init_guess = np.array([65,0.30])
bnds=([50,0.001],[80,1.0])

# the bnds are the two lower and two higher bounds for the unknowns (parameters), when absolute_sigma = False the errors are "normalized"
params, pcov = curve_fit(func3, xdata, ydata, p0 = init_guess, bounds = bnds, sigma = error, absolute_sigma = False)

#extracting the two parameters from the solution and rounding the values
ans_Hu, ans_O_m = params
Rans_Hu = round(ans_Hu,2)
Rans_O_m = round(ans_O_m,3)

# extracting the S.D. and rounding the values
perr = np.sqrt(np.diag(pcov))
ans_Hu_SD, ans_O_m_SD = np.sqrt(np.diag(pcov))
Rans_Hu_SD = round(ans_Hu_SD,2)
Rans_O_m_SD = round(ans_O_m_SD,3)

# estimating the goodness of fit, here the first row must be ignored since the error of the origin is 0.
chisq = sum((ydata[1:-1] - func3(xdata,ans_Hu,ans_O_m)[1:-1])**2/func3(xdata,ans_Hu,ans_O_m)[1:-1])
chisquar = round(chisq,2)

# normalised chisquar where P is the number of parameters (2) and is calculated as 
P=2
normchisquar = round((chisquar/(157-P)),2)

#calculation of residuals
residuals = ydata - func3(xdata,ans_Hu,ans_O_m)
#residuals_lsq = data - data_fit_lsq
ss_res = np.sum(residuals**2)
ss_tot = np.sum((ydata-np.mean(ydata))**2)

# r squared adjusted
r_squared = 1 - (ss_res/ss_tot)
r2 = round(r_squared,3)
r2adjusted = round(1-(((1-r2)*(len(ydata)-1))/(len(ydata)-P-1)),3)

#plot of imported data and model
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['lines.linewidth'] = 3
plt.figure(1,dpi=240)
plt.xlabel("Expansion factor")
plt.ylabel("Distance (Mpc)")
plt.xlim(0.3,1)
plt.ylim(0.0,16000)
plt.xscale("linear")
plt.yscale("linear")
fig, ax = plt.subplots()
plt.xlabel("Expansion factor, \u03BE", fontsize=18)
plt.ylabel("Luminosity distance (Mpc)", fontsize=18)
#plt.title("Flat $\Lambda$CDM model, $D_L$ vs. Exp. fact.", fontsize = 18)
ax.tick_params(axis="y", direction='in', length=8)
ax.tick_params(axis="x", direction='in', length=8)
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(3)
    ax.tick_params(width=3)
plt.errorbar(xdata, ydata, yerr=error, fmt='.k', capsize = 4)
plt.plot(xdata, func3(xdata,ans_Hu,ans_O_m), color = "orange", label = "Flat $\Lambda$CDM model")
plt.legend(loc='best', fancybox=True, shadow=False)

#print results
print()
print("The calculated Hubble constant with S.D. is: ", Rans_Hu,",", Rans_Hu_SD)
print("The calculated matter density with S.D. is: ", Rans_O_m,",",Rans_O_m_SD)
print("The adjusted r\u00b2 is calculated to be: ",r2adjusted)
#print("The calculated r\u00b2 is: ",r2)
#print("The goodness of fit, \u03C7\u00b2, estimate: ", chisquar)
print("The reduced goodness of fit, \u03C7\u00b2, is: ", normchisquar)

#Saving the plots in two different formats
fig.savefig("flatLCDM_D_L_data.eps", format="eps", dpi=2000, bbox_inches="tight", transparent=True)
fig.savefig("flatLCDM_D_L_data.pdf", format="pdf", dpi=2000, bbox_inches="tight", transparent=True)