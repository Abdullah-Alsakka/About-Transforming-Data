#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 17:41:24 2022

@author: Mike - Best to save this code on your own computer/file including the links for the two generated figues.
"""
print()
print("The semi-log (Log) model for cancer cell growth/replication in culture media.")

# import files and libraries
import numpy as np
import math
import csv
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# open data file
with open("LogCancerGrowth-Time.csv","r") as i:
    rawdata = list(csv.reader(i, delimiter = ","))

# exclude the first row from the analysis
exampledata = np.array(rawdata[1:],dtype=float)

xdata = exampledata[:,0]
ydata = exampledata[:,1]
error = exampledata[:,3]

# define the function - 
# The model to be examined, where x represents the independent variable time, B is the initial cell count and C is a measure of replication per hour. 
#This is the linear model with two parameters.
def func(x,B,C):
    return B + (x*C)

# The initial guesses of the model parameters
p0=[5.0,0.01] #The first value is the guess of the log(initial count), the second value for the guess of replication with time.

# Nonlinear fit of the model to the data, even though the model is linear. The bnds are the two minima and then the two maxima search ranges for B and C parameters. 
#The sigma uses the error data from the selected column of the .csv file.
bnds = ([1.0,0.001],[10.0,1.0])
params, pcov = curve_fit(func,xdata,ydata, p0, bounds = bnds, sigma = error, absolute_sigma = False)
# Above - the error are the values from line 25, absolute_sigma = False means that the standard deviations are "normalized" by the program.

# Extracting the two parameters and S.D. from the curve_fit.
ans_b, ans_c = params #ans_b is B and ans_c is C
perr = np.sqrt(np.diag(pcov))
ans_b_SD, ans_c_SD = perr #the standard deviations of ans_b and ans_c

# Rounding from 2 to 5 significant digits
rans_b = round(ans_b,2)
rans_c = round(ans_c,3)
rans_b_SD = round(ans_b_SD,4)
rans_c_SD = round(ans_c_SD,5)

# define the parameter values for the display plot derived from the regression routine
b=ans_b
c=ans_c

# calculate the initial cell number from the intercept
Yinter = math.exp(ans_b)
Ysd = math.exp(ans_b_SD)

# solve the function to be used for the nonlinear plot
funcdata = func(xdata,b,c)

# Estimating the goodness of fit from the difference between the observed distance data (ydata) and the calculated distances 
# using the term on the right-hand side, below.
chisq = sum((ydata - func(xdata,ans_b,ans_c))**2/(func(xdata,ans_b,ans_c)))
chisquar = round(chisq,4)

# normalised chisquar is calculated with the number of observations (60) and 2 the number of parameters, P
P=2
normchisquar = round((chisquar/(60-P)),5)

# To calculate the r**2 value
resids = ydata - func(xdata,ans_b,ans_c)
ss_res = np.sum(resids**2)
ss_tot = np.sum((ydata-np.mean(ydata))**2)
r_squared = 1 - (ss_res/ss_tot)
r2 = round(r_squared,4)
r2adjusted = round(1-(((1-r2)*(len(ydata)-1))/(len(ydata)-P-1)),4)

# plot of the imported data and function results
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['lines.linewidth'] = 3
plt.xlim(0.0,500.0)
plt.ylim(-10.0,3.5)
plt.xscale("linear")
plt.yscale("linear")
fig, ax = plt.subplots()
ax.tick_params(axis="y", direction='in', length=8)
ax.tick_params(axis="x", direction='in', length=10)
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(3)
    ax.tick_params(width=3)
plt.xlabel("Time (hours)", fontsize=18)
plt.ylabel("$Log_{e}(cells)$", fontsize=18)
plt.rc('xtick', labelsize=14) 
plt.rc('ytick', labelsize=14)
plt.errorbar(xdata, ydata, yerr=error, fmt='.k', capsize = 5)
plt.title("Cancer cell replication over time", fontsize = 16)
plt.plot(xdata, funcdata, label = "$Log_{e}$ model")
#plt.xlim(0.0,500.0)
#plt.ylim(-10.0,3.5)
plt.legend(loc='best', fancybox=True, shadow=False)

#print result values
print()
print("The calculated log initial cell count and S.D. are: ", rans_b, ",",rans_b_SD)
print ("Calculated initial cell count and S.D. are: "     ", ",       )
print("The calculated growth constant and S.D are: ", rans_c, ",",rans_c_SD)
print("The r\u00b2 adjusted value is calculated as: ",r2adjusted)
print("The goodness of fit, \u03C7\u00b2, is: ", chisquar)
print("The reduced goodness of fit, \u03C7\u00b2, is: ", normchisquar)
print("Reduced \u03C7\u00b2 = \u03C7\u00b2/(N-P), where N are the number of data pairs and P is the parameter count.")

#Routines to save figues in eps and pdf formats
fig.savefig("LogCancerGrowth_vs_time.eps", format="eps", dpi=2000, bbox_inches="tight", transparent=True)
fig.savefig("LogCancerGrowth_vs_time.pdf", format="pdf", dpi=2000, bbox_inches="tight", transparent=True)










