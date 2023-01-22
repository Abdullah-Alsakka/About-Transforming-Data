# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 00:46:47 2023

@author: Abdullah Alsakka

In this program we run through different constants around the accepted value of 25
in the equation 

mag = 5*log(D_L) + 25

to find at which value we get the minimum reduced chi squared. We first run it through
a large range and then narrow the range to find a more accurate value.
"""

## Import libraries ##
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

## Read Data and save as numpy array ##
data_without_headerclutter = pd.read_csv("Gold_Riess_mag_2004.csv")
DATA = data_without_headerclutter[['z', 'mag', 'sigma']] 
exampledata = np.array([DATA.iloc[:,0],DATA.iloc[:,1],DATA.iloc[:,2]])
xdata = exampledata[0]
ydata = exampledata[1]
error = exampledata[2]

## Define some general functions ##
def func(x,b,c):
    return (litesped*(1+x)/(b*np.sqrt(abs(1-c))))*np.sinh(2*(np.arctanh(np.sqrt(abs(1-c)))-np.arctanh(np.sqrt(abs(1-c))/np.sqrt((c/(1/(1+x)))+ (1-c)))))



## Define the constants and the lists where data will be saved ##
litesped = 299793
N = 200 # Determines the amount of increase from constant 24 and up to 26
chisq = np.empty(N)
Hubble = np.empty(N)
Hubble_Err = np.empty(N)
O_m = np.empty(N)
O_m_Err = np.empty(N)

for i in range(N):
    
    ## Curve Fit ##
    def func2(x,b,c):
        return 5*np.log10(func(x,b,c)) + 24  + 2*i/N
    
    print(f' Constant is {24 + 2*i/N}')
    
    init_guess = np.array([70,0.25])
    bnds=([50,0.01],[80,1.0])
    
    params, pcov = curve_fit(func2, xdata, ydata,p0 = init_guess, bounds = bnds, sigma = error, absolute_sigma = False)
    
    ans_Hu, ans_O_m = params
    SD_Hu, SD_O_m = np.sqrt(np.diag(pcov))
    
    ## Add the value of chi squared and H0 to a database ##
    P=len(params)
    n=len(xdata)
    chisq[i] = (sum((ydata - func2(xdata,ans_Hu,ans_O_m))**2/(error**2)))/(n-P)
    Hubble[i] = ans_Hu
    O_m[i] = ans_O_m
    Hubble_Err[i] = SD_Hu
    O_m_Err[i] = SD_O_m  
    
    print(f'H0 = {ans_Hu}')
    print(f'Reduced Chi Squared = {chisq[i]}')
    print('-'*50)
    
opt = np.where(chisq == min(chisq))[0][0] # The index where the optimal chi squared value is

print(f'When we have + {24 + 2*opt/N} we get the minimum chi squared as {chisq[opt]}, the hubble constant as {Hubble[opt]}+-{Hubble_Err[opt]} and the matter density as {O_m[opt]}+-{O_m_Err[opt]}')