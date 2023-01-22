# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 00:24:56 2023

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
N = 200 # Determines the amount of increase from constant 24 and up to 26
chisq = np.empty(N)
Hubble = np.empty(N)
Hubble_Err = np.empty(N)
O_m = np.empty(N)
O_m_Err = np.empty(N)

for i in range(N):
    
    ## Curve Fit ##
    def func3(x,Hu,O_m):
        return 5*(np.log10((litesped*(1+x)/Hu)*np.sinh(func2(x,O_m)))) + 24  + 2*i/N
    
    print(f' Constant is {24 + 2*i/N}')
    
    init_guess = np.array([70,0.25])
    bnds=([50,0.01],[80,1.0])
    
    params, pcov = curve_fit(func3, xdata, ydata,p0 = init_guess, bounds = bnds, sigma = error, absolute_sigma = False)
    
    ans_Hu, ans_O_m = params
    SD_Hu, SD_O_m = np.sqrt(np.diag(pcov))
    
    ## Add the value of chi squared and H0 to a database ##
    P=len(params)
    n=len(xdata)
    chisq[i] = (sum((ydata - func3(xdata,ans_Hu,ans_O_m))**2/(error**2)))/(n-P)
    Hubble[i] = ans_Hu
    O_m[i] = ans_O_m
    Hubble_Err[i] = SD_Hu
    O_m_Err[i] = SD_O_m  
    
    print(f'H0 = {ans_Hu}')
    print(f'Reduced Chi Squared = {chisq[i]}')
    print('-'*50)
    
opt = np.where(chisq == min(chisq))[0][0] # The index where the optimal chi squared value is

print(f'When we have + {24 + 2*opt/N} we get the minimum reduced chi squared as {chisq[opt]}, the hubble constant as {Hubble[opt]}+-{Hubble_Err[opt]} and the matter density as {O_m[opt]}+-{O_m_Err[opt]}')