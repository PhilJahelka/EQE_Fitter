# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 14:29:13 2024

@author: phill
"""
import numpy as np
import tkinter as tk
from pathlib import Path
import matplotlib.pyplot as plt
import scipy
import math

#limits for interpolating reflectance and alpha spectrum. 
#Must be wider than experimental data for wavelength error determination
LAMBDA_MIN = 0.3
LAMBDA_MAX = 1.0
LAMBDA_NUM = 200

#wvl error, surf diff len, bulk diff len, depletion width, emitter width, surface recomb param, fudge factor
INIT_GUESS = [0.0, 0.1, 1.0, 0.5, 0.5, 10.0, 1.0] 
BOUNDS = [(-0.01, 0.001, 0.001, 0, 0, 0, 0.95), (0.01, 20, 20, 1, 10, np.inf, 1.05)]
#How many times fitter can evaluate function
MAX_CALL = 100000


root = tk.Tk()
root.withdraw()

#Load spectra files
cwd = Path.cwd()
spectra_path = cwd.parents[1]/'Spectra/'
ref_file = spectra_path/'GaAs_R.csv' #reflectivity
ref_raw = np.genfromtxt(ref_file, delimiter = ',')
ref_raw[:,0] = ref_raw[:,0]/1E3 #convert to um
alpha_file = spectra_path/'GaAs_alpha.csv'
alpha_raw = np.genfromtxt(alpha_file, delimiter = ',')
alpha_raw[:,0] = alpha_raw[:,0]/1E3
alpha_raw[:,1] = alpha_raw[:,1]/1E4 #convert to 1/cm
eqe_file = tk.filedialog.askopenfilename()
#eqe_file = cwd.parents[1]/'2024-10-14/1cm2 cell 1 eqe.asc'
eqe_raw = np.genfromtxt(eqe_file, skip_header = 3)
eqe_raw[:,0] = eqe_raw[:,0]/1E3
exp_min_lambda = eqe_raw[0,0]
exp_max_lambda = eqe_raw[-1,0]

lambda_list = np.linspace(LAMBDA_MIN, LAMBDA_MAX, LAMBDA_NUM)
exp_lambda_list = np.array([x for x in lambda_list if x > exp_min_lambda and x < exp_max_lambda ])
ref_resamp = np.interp(exp_lambda_list, ref_raw[:,0], ref_raw[:,1])
alpha_resamp = np.interp(lambda_list, alpha_raw[:,0], alpha_raw[:,1])
eqe_resamp = np.interp(exp_lambda_list, eqe_raw[:,0], eqe_raw[:,1])
iqe = eqe_resamp/(1-ref_resamp)

#define the model. Ln is for surface, Lp for bulk
def iqe_model(wvl, dwvl, Ln, Lp, Wd, We, K, c):
    wvl_real = wvl - dwvl
    alpha = np.interp(wvl_real, lambda_list, alpha_resamp)
    iqe_dep = np.exp(-We * alpha) - np.exp(-(Wd + Wd) * alpha)
    iqe_bulk = (np.exp(-(Wd + Wd) * alpha) * Lp * alpha) / (1 + Lp * alpha)
    iqe_surf = -((np.exp(-We * alpha) * Ln * alpha) * (-np.exp(We * alpha) * (K + Ln * alpha) + (K + Ln * alpha) * math.cosh(We/Ln) + (1 + K * Ln * alpha) * math.sinh(We/Ln)))/((-1 + np.power(Ln * alpha, 2)) * (math.cosh(We/Ln) + K * math.sinh(We/Ln)))
    return c * (iqe_dep + iqe_bulk + iqe_surf)

fit, cov = scipy.optimize.curve_fit(iqe_model,
                                exp_lambda_list,
                                iqe,
                                p0 = INIT_GUESS,
                                maxfev = MAX_CALL,
                                bounds = BOUNDS,
                                )

plt.plot(exp_lambda_list, iqe)
plt.plot(exp_lambda_list, iqe_model(exp_lambda_list, *fit))
plt.show()
print('''
      Wavelength Error         = {0:.4f} um, \n
      Surface Diffusion Length = {1:.4f} um, \n
      Bulk Diffusion Length    = {2:.4f} um, \n
      Depletion Width          = {3:.4f} um, \n
      Emitter Width            = {4:.4f} um, \n
      Surface Recomb.          = {5:.4f}, \n
      Fudge Factor             = {6:.4f}   
      '''.format(*fit))