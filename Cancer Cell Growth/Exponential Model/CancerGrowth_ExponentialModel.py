#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 17:41:24 2022

@author: Mike Best to save this code on your own computer/file including the links for the two generated figures.
"""
print()
print("The exponential increase model for cell replication of dilute cancer cells in a media.")

# import data and library files
import numpy as np
import csv
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

#open data file
with open("CancerGrowth-Time.csv","r") as i:
    rawdata = list(csv.reader(i, delimiter = ","))
    
exdata = np.array(rawdata[1:],dtype=float)

xdata = exdata[:,0]
ydata = exdata[:,1]
error = exdata[:,2]

#define the function - 
#the model to be examined, where B represents the number of cancer cells in the medium at time = 0, 
#C represents the growth/cell division constant. This is the nonlinear model with two parameters.
def func(x,B,C):
    return B*np.exp(C*x)

#The initial guesses of the model parameters
p0=[400,0.01] 
#The first value is the guess of the initial number of cancer cells, the second value for the guess for growth rate per periond of time.

#Nonlinear fit of the model to the data. The bnds are the two minima and then the two maxima search ranges for the B and C parameters. 
#The sigma uses the error data from the selected column, 2, of the .csv file.
bnds = ([1.0,0.001],[500,1.0])
params, pcov = curve_fit(func,xdata,ydata, p0, bounds = bnds, sigma = error, absolute_sigma = False)
#Above - the error values are from line 25, absolute_sigma = False means that the standard deviations are not normalized by the program 
#but the individual values from column 2 of the data file. 

#Extracting the two parameters and S.D. from the curve_fit.
ans_b, ans_c = params #ans_b is the estimate for B and ans_c is the estimate for C
perr = np.sqrt(np.diag(pcov))
ans_b_SD, ans_c_SD = perr #the standard deviations of ans_b and ans_c

#Rounding to 2, 3 or 5 significant figures
rans_b = round(ans_b,2)
rans_c = round(ans_c,3)
rans_b_SD = round(ans_b_SD,2)
rans_c_SD = round(ans_c_SD,5)

#define the parameters for the display plot
bvalue=ans_b
cvalue=ans_c

#solve the function to be used for the plot
funcdata = func(xdata,bvalue,cvalue)

#Estimating the goodness of fit from the difference between the observed distance data (ydata) and 
#the calculated distances using the term on the right-hand side, below.
chisq = sum((ydata - func(xdata,ans_b,ans_c))**2/(error**2))
chisquar = round(chisq,2)
#normalised chisquar is calculated with the number of observation times (60) and 2 the number of parameters 
normchisquar = round((chisquar/(60-2)),2)

#To calculate the r**2 value
resids = ydata - func(xdata,ans_b,ans_c)
ss_res = np.sum(resids**2)
ss_tot = np.sum((ydata-np.mean(ydata))**2)
r_squared = 1 - (ss_res/ss_tot)
r2 = round(r_squared,4)
r2adjusted = round(1-(((1-r2)*(len(ydata)-1))/(len(ydata)-len(params)-1)),4)

#plot of the imported data and function results
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['lines.linewidth'] = 3
plt.xlim(0.0,240.0)
plt.ylim(0.0,7500)
plt.xscale("linear")
plt.yscale("linear")
fig, ax = plt.subplots()
ax.tick_params(axis="y", direction='in', length=8)
ax.tick_params(axis="x", direction='in', length=10)
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(3)
    ax.tick_params(width=3)
plt.xlabel("Time (hours)", fontsize=18)
plt.ylabel("Cell count", fontsize=18)
plt.rc('xtick', labelsize=14) 
plt.rc('ytick', labelsize=14)
plt.errorbar(xdata, ydata, yerr=error, fmt='.k', capsize = 5)
plt.title("Cancer cell number increase over time", fontsize = 16)
plt.plot(xdata, funcdata, label = "Exponential model")
plt.legend(loc='best', fancybox=True, shadow=False)

#print result values
print()
print("The calculated initial number and S.D. are: ", rans_b, ",",rans_b_SD)
print("The calculated growth constant and S.D are: ", rans_c, ",",rans_c_SD)
print("The adjusted r\u00b2 is calculated to be: ",r2adjusted)
print("The r\u00b2 value is calculated as: ",r2)
print("The goodness of fit, \u03C7\u00b2, is: ", chisquar)
print("The reduced goodness of fit, \u03C7\u00b2, is: ", normchisquar)
print("Reduced \u03C7\u00b2 = \u03C7\u00b2/(N-P), where N are the number of data pairs and P is the parameter count.")

#Routines to save figues in eps and pdf formats
fig.savefig("CellGrowth_vs_time.eps", format="eps", dpi=2000, bbox_inches="tight", transparent=True)
fig.savefig("CellGrowth_vs_time.pdf", format="pdf", dpi=2000, bbox_inches="tight", transparent=True)










