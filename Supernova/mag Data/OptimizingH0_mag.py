# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 13:42:32 2023

@author: Abdullah Alsakka

In this program we run through a desired Hubble range to find at which value we
get the minimum chi squared and reduced chi squared. We first run it through a
large range and then narrow the range to find a more accurate value.

We also massage the ydata above z=0.1 from 0.97 to 1.03 of the original value just to see
if there is any clear differences we can report.
"""

## Import libraries ##
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from scipy import integrate as intg

## Read Data and save as numpy array ##
data_without_headerclutter = pd.read_csv("Gold_Riess_mag_2004.csv")
DATA = data_without_headerclutter[['z', 'mag', 'sigma']]
exampledata = np.array([DATA.iloc[:,0],DATA.iloc[:,1],DATA.iloc[:,2]])
xdata = exampledata[0]
ydata = exampledata[1]
error = exampledata[2]

## Define some general functions ##
def integr(x,O_m):
    return intg.quad(lambda t: (1/(np.sqrt(((1+t)**2)*(1+O_m*t) - t*(2+t)*(1-O_m)))), 0, x)[0]
    
def func2(x, O_m):
    return np.asarray([integr(xx,O_m) for xx in x]) 

## Define the constants and the lists where data will be saved ##
litesped = 299793
N = 50    # Determines the steps that we will run through the free range
all_chisq_mod = []    # Modified Chi Squared
all_chisq_og = []    # Original Chi Squared
all_astrochisq_mod = []    # Modified Astronomer's Chi Squared
all_astrochisq_og = []    # Original Astronomer's Chi Squared
all_Om_mod = []    # Modified O_m
all_astro_Om_mod = []    # Modified O_m with astronomers chi squared
all_Om_og = []    # Original O_m

## The Hubble constants we will run through ##
Hubble_range = np.linspace(63,65,20)

## Where we will save all our modified y data ##
all_y_mod = np.empty([len(Hubble_range),len(ydata)])
all_astro_y_mod = np.empty([len(Hubble_range),len(ydata)])

## START THE RUN THROUGH ##
for des_Hu in Hubble_range:
    
    print(f'Step #{np.where(Hubble_range == des_Hu)[0][0] + 1}: \n==========')
    
    ## Curve Fitting ##
    def func3(x,O_m):
        return 5*(np.log10((litesped*(1+x)/des_Hu)*np.sinh(func2(x,O_m)))) + 25

    def func4(x,O_m):
        return 5*(np.log10((litesped*(1+x)/des_Hu)*np.sinh(integr(x,O_m)))) + 25

    init_guess = np.array([0.25])
    bnds=([0.01],[1.0])

    params, pcov = curve_fit(func3, xdata, ydata,p0 = init_guess, bounds = bnds, sigma = error, absolute_sigma = False)

    ans_O_m = params[0]

    print(f'Hubble Constant: {des_Hu}')
    print(f'Matter Density: {ans_O_m}')
    
    ## Checking for all the minimum chi squares in the defined range ##
    min_z = 0.1
    possible_y = np.empty([len(xdata[np.where(xdata > min_z)]),2*N+1])
    chisq_new = np.empty([len(xdata),2*N+1])
    astrochisq_new = np.empty([len(xdata),2*N+1])

    ## Calculating all y values and their chi squared from the lowest to highest in the determined range ##
    for j in range(len(xdata[np.where(xdata > min_z)])):
        a = len(xdata)-len(xdata[np.where(xdata > min_z)])+j # Just to start the data from index 0
        for i in range(-N, N+1):
            y = ydata[a] + i * error[a] * 0.03/N
            possible_y[j,i+N] = y
            chisq_new[j,i+N] = (y - func4(xdata[a],ans_O_m))**2/func4(xdata[a],ans_O_m)
            astrochisq_new[j,i+N] = (y - func4(xdata[a],ans_O_m))**2/(error[a]**2)

    ## Finding the minimum Chi squared for each y value and adding it to the mod y database
    new_ydata = []
    new_astroydata = []
    for i in range(len(xdata[np.where(xdata > min_z)])):
        new_ydata.append(possible_y[i, np.where(chisq_new[i,:] == min(chisq_new[i,:]))[0][0]])
        new_astroydata.append(possible_y[i, np.where(astrochisq_new[i,:] == min(astrochisq_new[i,:]))[0][0]])

    mod_ydata = np.array(ydata[0:len(xdata)-len(xdata[np.where(xdata > min_z)])].tolist()+new_ydata)
    mod_astroydata = np.array(ydata[0:len(xdata)-len(xdata[np.where(xdata > min_z)])].tolist()+new_astroydata)
    
    all_y_mod[np.where(Hubble_range == des_Hu)[0][0],:] = mod_ydata
    all_astro_y_mod[np.where(Hubble_range == des_Hu)[0][0],:] = mod_astroydata
    
    ## Finding the parameters with curve_fit for the modified y data ##
    mod_params, mod_pcov = curve_fit(func3, xdata, mod_ydata, p0 = init_guess, bounds = bnds, sigma = error, absolute_sigma = False)
    mod_ans_O_m = mod_params[0]
    
    all_Om_og.append(ans_O_m)
    all_Om_mod.append(mod_ans_O_m)
    
    mod_astro_params, mod_astro_pcov = curve_fit(func3, xdata, mod_astroydata, p0 = init_guess, bounds = bnds, sigma = error, absolute_sigma = False)
    mod_astro_ans_O_m = mod_params[0]
    
    all_astro_Om_mod.append(mod_astro_ans_O_m)
    
    print(f'Modified Matter Density: {mod_ans_O_m}')
    print(f'Modified Matter Density according to astronomers chisq: {mod_astro_ans_O_m}')
    
    
    ## Calculating the final chi squared for the normal and modified y data ##
    P=len(params)
    N=len(xdata)
    
    chisq = sum((ydata - func3(xdata,ans_O_m))**2/func3(xdata,ans_O_m))
    astrochisq = (sum((ydata - func3(xdata,ans_O_m))**2/(error**2)))/(N-P)
    new_chisq = sum((mod_ydata - func3(xdata,ans_O_m))**2/func3(xdata,ans_O_m))
    new_astrochisq = (sum((mod_ydata - func3(xdata,ans_O_m))**2/(error**2)))/(N-P)
    all_chisq_mod.append(new_chisq)
    all_astrochisq_mod.append(new_astrochisq)
    all_chisq_og.append(chisq)
    all_astrochisq_og.append(astrochisq)

    print(f'Our normal Chi squared is {chisq}\nOur reduced Chi squared is {astrochisq}\nOur optimized Chi squared is {new_chisq}\nOur optimized reduced Chi squared is {new_astrochisq}')

    print("-"*60,'\n')
    
print(f'THE BEST HUBBLE CONSTANT FOR NORMAL CHISQ MODIFIED DATA IS THE ONE IN STEP {all_chisq_mod.index(min(all_chisq_mod))+1}')
print(f'THE BEST HUBBLE CONSTANT FOR REDUCED CHISQ MODIFIED DATA IS THE ONE IN STEP {all_astrochisq_mod.index(min(all_astrochisq_mod))+1}')
print(f'THE BEST HUBBLE CONSTANT FOR NORMAL CHISQ ORIGINAL DATA IS THE ONE IN STEP {all_chisq_og.index(min(all_chisq_og))+1}')
print(f'THE BEST HUBBLE CONSTANT FOR REDUCED CHISQ ORIGINAL DATA IS THE ONE IN STEP {all_astrochisq_og.index(min(all_astrochisq_og))+1}')
print('*'*60, '\n')

print(f'H0 = {Hubble_range[all_chisq_mod.index(min(all_chisq_mod))]}\n')
print(f'Om = {all_Om_og[all_chisq_mod.index(min(all_chisq_mod))]}\n')
print(f'Om mod = {all_Om_mod[all_chisq_mod.index(min(all_chisq_mod))]}\n')
print(f'Om reduced mod = {all_astro_Om_mod[all_astrochisq_mod.index(min(all_astrochisq_mod))]}\n')
print(f'Chi square = {min(all_chisq_og)}\n')
print(f'Reduced Chi square = {min(all_astrochisq_og)}\n')
print(f'Chi square mod = {min(all_chisq_mod)}\n')
print(f'Chi square reduced mod = {min(all_astrochisq_mod)}')

## Plot the data with minimum chi squared ##
plt.plot(xdata, func3(xdata,ans_O_m), "b-", label = 'Fitted Curve')
plt.errorbar(xdata, ydata, yerr=error, fmt='.k', capsize = 5, label = 'Original Data with Error')
plt.plot(xdata, all_y_mod[all_chisq_mod.index(min(all_chisq_mod))], "r.", label = 'Modified Data')
plt.plot(xdata, func3(xdata, mod_ans_O_m), 'r--', label = 'Modified Curve')
plt.plot(xdata, all_y_mod[all_chisq_mod.index(min(all_chisq_mod))], "g.", label = 'Red_Chisq Modified Data')
plt.plot(xdata, func3(xdata, mod_astro_ans_O_m), 'g--', label = 'Red_Chisq Modified Curve')
plt.legend()
plt.show()