# plot_processed_data.py
#
# Grant Kirchhoff
# Last Updated: 12/19/2022
"""
After processing data from 'evaluation_high_OD_iterate.py', plot post-processed data.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import pandas as pd
import pickle

cwd = os.getcwd()
dirLib = cwd + r'/library'
if dirLib not in sys.path:
    sys.path.append(dirLib)

from load_ARSENL_data import set_binwidth

### NOTE!!! Make sure to check filepaths for appropriate and MATCHING files for plotting.

load_dir = r'C:\Users\jason\OneDrive - UCB-O365\ARSENL\Experiments\SPCM\evaluation_loss'
load_file = r'\eval_loss_dtimeTrue_OD3.4E+00-3.4E+00_order7-19_shots2.50E+02_use_final_True_best_fit_run#1.csv'
param_filename = r'\eval_loss_dtimeTrue_OD3.4-3.4_order7-19_ref_shots1.00E+07_lsr_shots2.50E+02_use_final_True_best_fit_run#1.pkl'

df = pd.read_csv(load_dir + load_file)
with open(load_dir+r'\fit_figures'+param_filename, 'rb') as f:
   params = pickle.load(f)
flight_time_lst = params[0]
flight_time_ref = params[1]
t_min = params[2]
t_max = params[3]
dt = params[4]
n_shots = params[5]

headers = list(df.columns.values)
t_fine = df['time vector']
fit_rate_seg_lst = df.loc[:, headers[2]:headers[-1]]

num_OD = len(headers[2:])

for i in range(num_OD):
    header = headers[2+i]
    fit_rate_seg = fit_rate_seg_lst[header]
    flight_time = flight_time_lst[i]

    fig = plt.figure()
    ax = fig.add_subplot(111)

    bin_array = set_binwidth(t_min, t_max, dt)
    n, bins = np.histogram(flight_time, bins=bin_array)
    binwidth = np.diff(bins)[0]
    N = n / binwidth / n_shots  # [Hz] Scaling counts to arrival rate
    center = 0.5 * (bins[:-1] + bins[1:])
    ax.bar(center, N, align='center', width=binwidth, color='b', alpha=0.5)

    # Arrival rate fit
    t_fine = np.arange(t_min, t_max, dt)
    ax.plot(t_fine, fit_rate_seg, 'r--')
    ax.set_title('Arrival Rate Fit: OD{}'.format(header))
    ax.set_xlabel('time [s]')
    ax.set_ylabel('Photon Arrival Rate [Hz]')
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # ax.text(0.1, 0.90, 'Polynomial order: {}'.format(min_order), transform=ax.transAxes, fontsize=14,
    #         verticalalignment='top', bbox=props)
    plt.tight_layout()

plt.show()

