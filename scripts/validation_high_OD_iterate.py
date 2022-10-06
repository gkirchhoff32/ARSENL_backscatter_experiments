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
set_max_det = True
max_num = 100                   # Include up to certain number of laser shots
deadtime = 25e-9                  # [s] Acquisition deadtime

# load_dir = r'C:\Users\Grant\OneDrive - UCB-O365\ARSENL\Experiments\Deadtime_Experiments\Data\Deadtime_Experiments_HiFi'
# files = os.listdir(load_dir)

# # Make sure file location is accurate
cwd = os.getcwd()
data_dir = r'C:\Users\Grant\OneDrive - UCB-O365\ARSENL\Experiments\Deadtime_Experiments\Data\Deadtime_Experiments_HiFi'
fname = r'/OD02_Dev_0_-_2022-04-15_12.01.32.ARSENL.OD02.ARSENL.nc'
flight_time_ref, n_shots_ref, t_det_lst_ref = dorg.data_organize(dt, data_dir, fname, window_bnd, max_num, set_max_det, exclude_shots)
print('\n{}:'.format(fname[1:5]))
print('Number of detections: {}'.format(len(flight_time_ref)))
print('Number of laser shots: {}'.format(n_shots_ref))

# OD = np.zeros(len(files))
# for i in range(len(files)):
#     OD[i] = float(files[i][2:4]) / 10



# for i in range(len(OD)):
#
#
