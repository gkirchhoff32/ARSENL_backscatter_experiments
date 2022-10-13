import numpy as np
import pandas as pd
import time
import pickle
import matplotlib.pyplot as plt
import torch
from scipy import stats
import os
import xarray as xr
import sys

cwd = os.getcwd()
dirLib = cwd + r'/library'
if dirLib not in sys.path:
    sys.path.append(dirLib)

import fit_polynomial_methods as fit
import data_organize as dorg

# Adjust parameters here to customize run

### CONSTANTS ####
c = 2.99792458e8                      # [m/s] Speed of light
dt = 25e-12                   # [s] TCSPC resolution

### PARAMETERS ###
window_bnd = [30e-9, 33e-9]       # [s] Set boundaries for binning to exclude outliers
exclude_shots = True                     # Set TRUE to exclude data to work with smaller dataset
# set_max_det = True  # Set TRUE if want to use detection events as the maximum value
max_num = 10                   # Include up to certain number of laser shots
deadtime = 25e-9                  # [s] Acquisition deadtime

load_dir = r'C:\Users\jason\OneDrive - UCB-O365\ARSENL\Experiments\Deadtime_Experiments\Data\Deadtime_Experiments_HiFi'
files = os.listdir(load_dir)

OD_list = np.zeros(len(files))
for i in range(len(files)):
    OD_list[i] = float(files[i][2:4]) / 10

fname_ref = r'\OD20_Dev_0_-_2022-04-15_11.17.49.ARSENL.OD20.ARSENL.nc'
# max_num = int(max_num * 10**OD_list[-3])
flight_time_ref, n_shots_ref, t_det_lst_ref = dorg.data_organize(dt, load_dir, fname_ref, window_bnd, max_num, exclude_shots)
print('\n{}:'.format(fname_ref[1:5]))
print('Number of detections: {}'.format(len(flight_time_ref)))
print('Number of laser shots: {}'.format(n_shots_ref))

for i in range(3):
    print(OD_list[i])
    fname = r'/' + files[i]
    flight_time, n_shots, t_det_lst = dorg.data_organize(dt, load_dir, fname, window_bnd, max_num, exclude_shots)
    print('\n{}:'.format(fname_ref[1:5]))
    print('Number of detections: {}'.format(len(flight_time)))
    print('Number of laser shots: {}'.format(n_shots))