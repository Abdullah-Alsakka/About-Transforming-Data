# Modeling Signals from Distant Emissions

Presenting data and programs (codes) used to evaluate astronomical data, object emissions, supernova type Ia, and associated redshifts published year 2022. These are the data from 1701 SNe Ia in the article by Brout et al. (2022) vol.938:110, the Pantheon+ data. We also use 18 TRGB (Tip of the Red Giant Branch stars) observations, from the Freedman group (2019) "The Carnegie-Chicago Hubble Program. VIII. An Independent Determination of the Hubble Constant Based on the Tip of the Red Giant Branch" vol.882:34. These data are included in some data sets and also as a stand-alone for Hubble constant determinations. Finally 16 measurements of TRGB emissions from Anand et al. (2022) "Comparing Tip of the Red Giant Branch Distance Scales: An Independent Reduction of the Carnegie-Chicago Hubble Program and the Value of the Hubble Constant" vol.932:15 are also included in some data ensembles and as stand-alone for Hubble constant determinations.

'TRGB' stars, Tip of the Red Giant Branch stars ---> Data and programs for modeling combining both SNe Ia signals (above) and 16 signals from nearby TRGB stars as reported by Anand et al. 2022 Astrophysical Journal, Vol. 932:15.

Every evaluation presents several statistic calculations. First, is the typical r^2 correlation which is of only middling usefulness with SNe Ia and TRGB data, since many analyses report r^2 of approximately 0.99. Second, is the F-test which is a rarely used, statistical test which is very sensitive to correlation without an upper limit and useful across models with different number of free parametters using large data ensembles. Third, is a version of the chi^2 calculation as commonly performed by astronomers (see Riess et al. 1998 AJ Vol. 116 p. 1009-1038) but is an uncommon varient. Fourth, are calculations of the values of the Bayesian Information Criteria (BIC).  It is observed that the BIC test is not really very useful for comparing unrelated models with the SNe Ia data and ends up only indicating the number of free parameters in the model. Specifically, the low BIC values calculated for the Einstein-DeSitter model variants do not reflect the relative failure of these models compared to models based on the Friedmann-Lemaitre-Robertson-Walker (FLRW) approximation, most which appear when plotted to better fit the SNe Ia data. The F-statistic values increase dramatically with better correlation and more data, this statistic seems best for model evaluation of SNe Ia data and TRGB star emissions, across models with differeing numbers of free parameters.

For convenience a few data ensembles are provided in the same folders as the evaluation codes. The user must be careful to adjust the data number (N) to acurately reflect the real number of data pairs. This value is required for accurate calculations of statistics.
