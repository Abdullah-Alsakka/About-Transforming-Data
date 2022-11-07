#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 17:41:24 2022

@author: Mike Best to save this code on your own computer/file including the links for the two generated figues.
"""
print()
print("The exponential decay model for capacitor discharge in an RC (resistor-capasitor) circuit.")

# import files
import numpy as np
import csv
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

#open data file
with open("Noisy_Volt_mod_vers2.csv","r") as i:
    rawdata = list(csv.reader(i, delimiter = ","))
    
exampledata = np.array(rawdata[1:],dtype=float)

xdata = exampledata[:,1]
ydata = exampledata[:,2]
error = exampledata[:,3]

#define the function - 
#the model to be examined, where x represents the independent variable values and V (volts) and RC (resistivity) are the parameters to be estimated. This is the nonlinear model with two parameters.
def func(x,V,RC):
    return V*np.exp(-x/RC)

#The initial guesses of the model parameters
p0=[30.0,50.0] #The first value is the voltage guess, the second value for the guess for resistivity.


#Nonlinear fit of the model to the data. The bnds are the two minima and then the two maxima search ranges for V and RC parameters. The sigma uses the error data from the selected column of the .csv file.
bnds = ([25.0,0.1],[35.0,70.0])
params, pcov = curve_fit(func,xdata,ydata, p0, bounds = bnds, sigma = error, absolute_sigma = False)
#Above - the error are the values from line 25, absolute_sigma = False means that the standard deviation is not selected by the program but we let the errors be the file values. 

#Extracting the two parameters and S.D. from the curve_fit.
ans_b, ans_c = params #ans_b is V and ans_c is RC
perr = np.sqrt(np.diag(pcov))
ans_b_SD, ans_c_SD = perr #the standard deviations of ans_b and ans_c

#Rounding to 2 or 3 significant figures
rans_b = round(ans_b,2)
rans_c = round(ans_c,3)
rans_b_SD = round(ans_b_SD,2)
rans_c_SD = round(ans_c_SD,3)

#define the parameters for the display plot
b=ans_b
c=ans_c

#solve, again, the function to be used for the nonlinear plot
funcdata = func(xdata,b,c)

#Estimating the goodness of fit from the difference between the observed distance data (ydata) and the calculated distances using the term on the right-hand side, below.
chisq = sum((ydata - func(xdata,ans_b,ans_c))**2/(error**2))
chisquar = round(chisq,2)
#normalised chisquar is calculated with the number of observations (23) and 2 the number of parameters 
normchisquar = round((chisquar/(23-2)),2)
#The BIC value is calculated as
BIC = 23 * np.log10(chisq/23) + 2*np.log10(23)
normBIC = round(BIC,2)
#BIC is the acronym for Bayesian Information Criteria

#To calculate the r**2 value
resids = ydata - func(xdata,ans_b,ans_c)
ss_res = np.sum(resids**2)
ss_tot = np.sum((ydata-np.mean(ydata))**2)
r_squared = 1 - (ss_res/ss_tot)
r2 = round(r_squared,4)

#plot of the imported data and function results
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['lines.linewidth'] = 3
plt.xlim(-10.0,140.0)
plt.ylim(0.0,35)
plt.xscale("linear")
plt.yscale("linear")
fig, ax = plt.subplots()
ax.tick_params(axis="y", direction='in', length=8)
ax.tick_params(axis="x", direction='in', length=10)
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(3)
    ax.tick_params(width=3)
plt.xlabel("Time", fontsize=18)
plt.ylabel("Volts", fontsize=18)
plt.rc('xtick', labelsize=14) 
plt.rc('ytick', labelsize=14)
plt.errorbar(xdata, ydata, yerr=error, fmt='.k', capsize = 5)
plt.title("RC decay over time", fontsize = 16)
plt.plot(xdata, funcdata, label = "Exponential model")
plt.legend(loc='best', fancybox=True, shadow=False)

#print result values
print()
print("The calculated initial voltage and S.D. are: ", rans_b, ",",rans_b_SD)
print("The calculated RC and S.D are: ", rans_c, ",",rans_c_SD)
print("The r\u00b2 value is calculated as: ",r2)
print("The goodness of fit, \u03C7\u00b2, is: ", chisquar)
print("The reduced goodness of fit, \u03C7\u00b2, is: ", normchisquar)
print("Reduced \u03C7\u00b2 = \u03C7\u00b2/(N-P), where N are the number of data pairs and P is the parameter count.")
print("The calculated BIC is: ", normBIC)
print("BIC represents the Bayesian Information Criteria")

#Routines to save figues in eps and pdf formats
fig.savefig("Noisy_Volt_time.eps", format="eps", dpi=2000, bbox_inches="tight", transparent=True)
fig.savefig("Noisy_Volt_time.pdf", format="pdf", dpi=2000, bbox_inches="tight", transparent=True)










