# -*- coding: utf-8 -*-
"""
Created on Sun Dec 18 23:05:42 2022
@author: abd__
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 17:53:20 2022

@author: mike
@co-author: Abdullah Alsakka

This curve_fit regression routine of Python scipy, uses the 'distance mag' data, 
converted to luminosity distances, D_L, vs expansion factor, from Brout et al. 2022, 
'The Pantheon+ Analysis: Cosmological Constraints' Astrophys. J. vol. 938, 110. 
The LCDM model used here requires numerical integration with two parameters the 
normalised matter density, O_m, and O_L, dark energy. The third parameter, Hubble constant, 
is then used during the regression. An estimate of \Omega_k is possible though the sinn(x) = x
condition is used presuming Euclidean space geometry.
"""
print()
print("This is the 3PD_L_standard model. The correlation is luminosity distance, D_L vs. expansion factor")
print("with sinn(x) = x for Euclidean geometry.")
print()

import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import integrate as intg
import math
from sklearn.metrics import r2_score
from astropy.stats.info_theory import bayesian_info_criterion

# open data file
with open("TRGB_D_L_DATA.csv","r") as i:
    rawdata = list(csv.reader(i, delimiter = ","))
    
# ignore the first row   
exampledata = np.array(rawdata[1:],dtype=float)

xdata = exampledata[:,1]
ydata = exampledata[:,4]
error = exampledata[:,7]

#Model function where t is the "dummy" variable for integration
def integr(x,O_m,O_L):
    return intg.quad(lambda t: 1/(t*(np.sqrt((O_m/t)+((O_L)*(t**2))+(1-O_m-O_L)))), x, 1)[0]
    
def func2(x, O_m,O_L):
    return np.asarray([integr(xx,O_m,O_L) for xx in x])

litesped = 299793

# Hu is the Hubble constant
def func3(x,Hu,O_m,O_L):
    return (litesped/(x*Hu*np.sqrt(np.abs(1-O_m-O_L))))*(np.sqrt(np.abs(1-O_m-O_L))*func2(x,O_m,O_L))

init_guess = np.array([72,0.30,0.69])
bnds=([60,0.001,0.001],[80,1.0,1.0])

# the bnds are the lower and upper bounds for the unknowns (parameters), when absolute_sigma = False the errors are "normalized"
params, pcov = curve_fit(func3, xdata, ydata, p0 = init_guess, bounds = bnds, sigma = error, absolute_sigma = False)

#extracting the two parameters from the solution and rounding the values
ans_Hu, ans_O_m, ans_O_L = params
Rans_Hu = round(ans_Hu,2)
Rans_O_m = round(ans_O_m,5)
Rans_O_L = round(ans_O_L,5)
Rans_O_k = round(1 - ans_O_m - ans_O_L,3)

# extracting the S.D. and rounding the values
perr = np.sqrt(np.diag(pcov))
ans_Hu_SD, ans_O_m_SD, ans_O_L_SD = np.sqrt(np.diag(pcov))
Rans_Hu_SD = round(ans_Hu_SD,2)
Rans_O_m_SD = round(ans_O_m_SD,5)
Rans_O_L_SD = round(ans_O_L_SD,5)
Rans_O_k_SD = round(np.sqrt(ans_O_m_SD**2 + ans_O_L_SD**2),3)

# where P is the number of parameters (3), N is the number of data pairs (1702) 
P=3
N=1717
e = 2.71828183

#Calculate the chi^2 according to astronomers
newxsqrd = sum(((ydata[1:-1] - func3(xdata,ans_Hu,ans_O_m,ans_O_L)[1:-1])**2)/(error[1:-1]**2))
newxsqrded = round(newxsqrd/(N-P),2) #rounded to 2 digits

#The usual method for BIC calculation is
SSE = sum((ydata - func3(xdata,ans_Hu,ans_O_m,ans_O_L))**2)
log_SSE = math.log(e,SSE)
small_bic = bayesian_info_criterion(log_SSE, P, N)
rBIC = round(small_bic,2)

#calculation of residuals
residuals = ydata - func3(xdata,ans_Hu,ans_O_m,ans_O_L)
#residuals_lsq = data - data_fit_lsq
ss_res = np.sum(residuals**2)
ss_tot = np.sum((ydata-np.mean(ydata))**2)

#routine for calculating r squared
ycalc = func3(xdata,ans_Hu,ans_O_m,ans_O_L)
R_sqrd = r2_score(ydata, ycalc)
R_square = round(R_sqrd,4)

#Calculation of the weighted F-statistic
SSEw = sum((1/error)*(residuals)**2)
SSM = sum((1/error)*(ydata - np.mean(ydata))**2) 
MSR = (SSM - SSEw)/(P)
MSE = SSEw/(N-P)
Fstat = MSR/MSE
rFstat = round(Fstat,1)

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
ax.tick_params(axis="y", direction='in', length=8)
ax.tick_params(axis="x", direction='in', length=8)
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(3)
    ax.tick_params(width=3)
plt.errorbar(xdata, ydata, yerr=error, fmt='.k', capsize = 4)
plt.plot(xdata, func3(xdata,ans_Hu,ans_O_m,ans_O_L), color = "orange", label = "3P$D_L$standard model")
plt.legend(loc='best', fancybox=True, shadow=False)

#print results
print()
print("The calculated Hubble constant with S.D. is: ", Rans_Hu,",", Rans_Hu_SD)
print("The calculated Omega_m with S.D. is: ", Rans_O_m,",",Rans_O_m_SD)
print("The calculated Omega_L with S.D. is: ", Rans_O_L,",",Rans_O_L_SD)
print("The calculated Omega_k with S.D. is: ", Rans_O_k,",",Rans_O_k_SD)
print()
print('The r\u00b2 is:', R_square)
print('The weighted F-statistic is:', rFstat)
print("The reduced goodness of fit, according to astronomers, \u03C7\u00b2, is: ", newxsqrded)
print("The BIC estimate is: ",rBIC)
print()

#Saving the plots in two different formats
fig.savefig("3PDLstandard.eps", format="eps", dpi=2000, bbox_inches="tight", transparent=True)
fig.savefig("3PDLstandard.pdf", format="pdf", dpi=2000, bbox_inches="tight", transparent=True)
