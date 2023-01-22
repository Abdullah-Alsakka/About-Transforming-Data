# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 11:57:07 2022

@author: abd__
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 17:53:20 2022

@author: mike
"""
print("A test of the Friedmann-Lemaitre-Robertson-Walker (FLRW) model using the Python 3, scipy curve_fit regression routine with the SNe Ia data, as mag vs redshift (z), from the gold data set of Riess, A.G. et al. 'Type Ia Supernova Discoveries at z> 1 from the Hubble Space Telescope: Evidence for Past Deceleration and Constraints on Dark Energy Evolution' Astrophys. J. vol. 607(2), 665-687 (2004). This variation of the LCDM model used here has three parameters: Hubble constant, Hu, normalised matter density, O_m; the cosmological constant, O_L, with 1=O_m+O_L+O_k as the remainder of information.")
print()
print("This is the 3PmagLCDM model, typically known as the standard model of cosmology.")

# import the data file and the Python 3 libraries
import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import integrate as intg

# open data file
with open("Gold_Riess_mag_2004.csv","r") as i:
    rawdata = list(csv.reader(i, delimiter = ","))
    
# ignore the top row    
exampledata = np.array(rawdata[1:],dtype=float)

xdata = exampledata[:,1]
ydata = exampledata[:,2]
error = exampledata[:,3]

# initial guesses for the normalized matter density, O_m and dark energy, O_L
O_m = 0.3 #initial guess for matter density
O_L = 0.6 #initial guess for Omega_L

# where t is the "dummy" variable for integration

def integr(x,O_m,O_L):
    return intg.quad(lambda t: 1/((np.sqrt((1+t)**3*O_m + (1+t)**2*(1-O_m-O_L) + O_L))), 0, x)[0]
    
def func2(x, O_m, O_L):
    return np.asarray([integr(xx,O_m,O_L) for xx in x])

# specify the speed of light
litesped = 299793

def func3(x,Hu,O_m,O_L):
    return 5*(np.log10((litesped*(1+x)/(np.sqrt(np.abs(1-O_m-O_L))*Hu))*np.sinh(np.sqrt(np.abs(1-O_m-O_L))*func2(x,O_m,O_L)))) + 25

# guess for the Hubble constant, Hu. No need to guess the normalized matter density, O_m, and dark energy, O_L.
init_guess = np.array([65,O_m,O_L])

# allowed range for the two parameters
bnds=([50,0.001,0.001],[80,1.0,1.0])

# fitting the model to the data, note that when choosing absolute_sigma = False the standard deviations (error) are normalized
params, pcov = curve_fit(func3, xdata, ydata,p0 = init_guess, bounds = bnds, sigma = error, absolute_sigma = False)

# extracting and rounding the parameter, Hu, O_m, O_L, O_k values
ans_Hu, ans_O_m, ans_O_L = params
rans_Hu = round(ans_Hu,2)
rans_O_m = round(ans_O_m,4)
rans_O_L = round(ans_O_L,4)
rans_O_k = round(1 - rans_O_m - rans_O_L,4)

# extracting and rounding the estimated standard deviations.
perr = np.sqrt(np.diag(pcov))
SD_Hu, SD_O_m, SD_O_L = perr
rSD_Hu = round(SD_Hu,2)
rSD_O_m = round(SD_O_m,3)
rSD_O_L = round(SD_O_L,3)

# calculating the value of O_k S.D. and rounding off 
O_k_SD = np.sqrt((SD_O_m)**2 + (SD_O_L)**2)
rSD_O_k = round(O_k_SD,3)

# normalised chisquar is calculated for N=156 data pairs with P the parameter count (3) as
P=3
N=156

#Calculate the chi^2 according to astronomers
newxsqrd = sum(((ydata - func3(xdata,ans_Hu,ans_O_m,ans_O_L))**2)/(error**2))
newxsqrded = round(newxsqrd/(N-P),2)

#The BIC value is calculated as
BIC = N * np.log10(newxsqrd/N) + P*np.log10(N)
normBIC = round(BIC,2)

# calculation of residuals
residuals = ydata - func3(xdata,ans_Hu,ans_O_m,ans_O_L)
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
#plt.xscale("linear")
#plt.yscale("linear")
fig, ax = plt.subplots()
ax.tick_params(axis="y", direction='in', length=14)
ax.tick_params(axis="x", direction='in', length=10)
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(3)
    ax.tick_params(width=3)
plt.errorbar(xdata, ydata, yerr=error, fmt='.k', capsize = 4)
plt.plot(xdata, func3(xdata,ans_Hu,ans_O_m,ans_O_L), color = "green", label = "3Pstandard model")
plt.xlabel("Redshift z", fontsize = 18)
plt.ylabel("mag (no units)", fontsize = 18)
#plt.title("standard model, mag vs. redshift z", fontsize = 18)
plt.legend(loc='best', fancybox=True, shadow=False)

#print results
print()
print("The calculated Hubble constant and S.D. are: ", rans_Hu, ",",rSD_Hu)
print("The calculated Omega_m and S.D. are: ",rans_O_m, ",",rSD_O_m )
print("The calculated Omega_L with S.D. is: ", rans_O_L,",",rSD_O_L)
print("The calculated Omega_k with S.D. is: ", rans_O_k,",",rSD_O_k)
print()
print("The adjusted r\u00b2 is calculated to be: ",r2adjusted)
#print("The calculated r\u00b2 is: ",r2)
#print("The goodness of fit, \u03C7\u00b2, estimate: ", chisquar)
print("The reduced goodness of fit, \u03C7\u00b2, is: ", newxsqrded)
print("The estimate for BIC is: ", normBIC)


#commands to save plots in two different formats
fig.savefig("3Param_LCDM_mag_data.eps", format="eps", dpi=2000, bbox_inches="tight", transparent=True)
fig.savefig("3Param_LCDM_mag_data.pdf", format="pdf", dpi=2000, bbox_inches="tight", transparent=True)
