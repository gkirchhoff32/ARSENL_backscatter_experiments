# validation_high_OD_iterate.py
#
# Grant Kirchhoff
# Last Updated: 10/21/2022
"""
Automation script to loop through different OD datasets and evaluate fit performance against an evaluation dataset
(i.e., high OD setting).

Note to user: Make sure to edit the parameters in the Parameters section before running.
TODO: Make a guide that describes the parameter functions.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import sys
import time
import pandas as pd
import xarray as xr
import pickle

start = time.time()

cwd = os.getcwd()
dirLib = cwd + r'/library'
if dirLib not in sys.path:
    sys.path.append(dirLib)

import fit_polynomial_methods as fit
import data_organize as dorg

########################################################################################################################

### CONSTANTS ####
c = 2.99792458e8                      # [m/s] Speed of light
dt = 25e-12                   # [s] TCSPC resolution

# EDIT THESE PARAMETERS BEFORE RUNNING!
### PARAMETERS ###
window_bnd = [30e-9, 33e-9]       # [s] Set boundaries for binning to exclude outliers
exclude_shots = True                     # Set TRUE to exclude data to work with smaller dataset
max_lsr_num_ref = int(1e6)                   # If set_max_det set to FALSE, include up to certain number of laser shots
max_det_num_ref = 2000                       # If set_max_det set to TRUE, include up to a certain number of detections
set_max_det = False                          # Set TRUE if data limiter is number of detections instead of laser shots.
deadtime = 25e-9                  # [s] Acquisition deadtime
use_stop_idx = True               # Set TRUE if you want to use up to the OD value preceding the reference OD
run_full = True                   # Set TRUE if you want to run the fits against all ODs. Otherwise, it will just load the reference data.
include_deadtime = False  # Set True to include deadtime in noise model
use_poisson_eval = True  # Set TRUE if you want to use the Poisson model for the evaluation loss
use_sim_data = True  # Set TRUE if you want to use simulated data for this run

# Optimization parameters
rel_step_lim = 1e-8  # termination criterion based on step size
max_epochs = 400  # maximum number of iterations/epochs
learning_rate = 1e-1  # ADAM learning rate
term_persist = 20  # relative step size averaging interval in iterations
intgrl_N = 10000  # Set number of steps in numerical integration

# Set iterate to True if you want to iterate through increasing complexity.
# Otherwise set to False if you want to check a single polynomial order.
single_step_iter = False
M_max = 21  # Max polynomial complexity to test if iterating
M_lst = np.arange(4, 15, 1)

########################################################################################################################

# Load / generate important data structures from simulated / real datsets
# NOTE: If user wants to test different parameters with simulated data, look into "generate_sim_data.py"

# Only need to change this file location if using simulated data generated from "generate_sim_data.py"
load_sim_pkl_dir = r'C:\Users\Grant\OneDrive - UCB-O365\ARSENL\Experiments\Deadtime_Experiments\Data\Simulated\Select'
fname_sim_pkl_ref = r'C:\Users\Grant\OneDrive - UCB-O365\ARSENL\Experiments\Deadtime_Experiments\Data\Simulated\sim_amp1.0E+06_nshot1.0E+06.pkl'
infile_ref = open(fname_sim_pkl_ref, 'rb')
df = pickle.load(infile_ref)
infile_ref.close()

flight_time_ref = xr.DataArray(df.flight_time.data)
true_flight_time_ref = xr.DataArray(df.true_flight_time.data)
n_shots_ref = df.n_shots.data
t_det_lst_ref = df.t_det_lst.data
t_phot_lst_ref = df.t_phot_lst.data
target_amplitude_ref = df.target_amplitude.data
target_time_ref = df.target_time.data
laser_pulse_width_ref = df.laser_pulse_width.data
window_bnd_ref = df.window_bnd.data
background_ref = df.background.data

# load_dir = r'C:\Users\Grant\OneDrive - UCB-O365\ARSENL\Experiments\Deadtime_Experiments\Data\Deadtime_Experiments_HiFi'
save_dir = load_sim_pkl_dir + r'/../../../Figures/evaluation_loss'
files = os.listdir(load_sim_pkl_dir)
#
# OD_list = np.zeros(len(files))
# for i in range(len(files)):
#     OD_list[i] = float(files[i][2:4]) / 10
#
# fname_ref = r'\OD30_Dev_0_-_2022-04-15_11.24.55.ARSENL.OD30.ARSENL.nc'
# OD_ref = int(fname_ref[3:5]) / 10
# flight_time_ref, n_shots_ref, t_det_lst_ref = dorg.data_organize(dt, load_dir, fname_ref, window_bnd,
#                                                                  max_lsr_num_ref, max_det_num_ref, set_max_det,
#                                                                  exclude_shots)
# print('\n{}:'.format(fname_ref[1:5]))
# print('Number of detections (reference): {}'.format(len(flight_time_ref)))
# print('Number of laser shots (reference): {}'.format(n_shots_ref))

# I define the max/min times as fixed values. They are the upper/lower bounds of the fit.
# Time vector per shot
t_min = window_bnd_ref[0]
t_max = window_bnd_ref[1]
dt = dt
t_fine = np.arange(t_min, t_max, dt)

# Generate "active-ratio histogram" that adjusts the histogram proportionally according to how many bins the detector was "active vs dead"
bin_edges = np.linspace(t_min, t_max, intgrl_N+1, endpoint=False)
if not include_deadtime:
    active_ratio_hst_ref = torch.ones(len(bin_edges-1))
else:
    active_ratio_hst_ref = fit.deadtime_noise_hist(t_min, t_max, intgrl_N, deadtime, t_det_lst_ref, n_shots_ref)

# Fitting routine
if run_full:
    val_final_loss_lst = []
    eval_final_loss_lst = []
    rmse_final_lst = []
    C_scale_final = []
    percent_active_lst = []
    target_amplitude_lst = []
    if not use_stop_idx:
        stop_idx = 1
    else:
        # stop_idx = int(np.where(OD_list == OD_ref)[0])
        stop_idx = len(files)
    for k in range(stop_idx):
        fname_sim_pkl = r'/' + files[k]
        # OD_fit = int(fname[3:5]) / 10
        max_lsr_num = max_lsr_num_ref
        max_det_num = max_det_num_ref

        infile = open(load_sim_pkl_dir + fname_sim_pkl, 'rb')
        df = pickle.load(infile)
        infile.close()

        flight_time = xr.DataArray(df.flight_time.data)
        true_flight_time = xr.DataArray(df.true_flight_time.data)
        n_shots = df.n_shots.data
        t_det_lst = df.t_det_lst.data
        t_phot_lst = df.t_phot_lst.data
        target_amplitude = df.target_amplitude.data
        target_time = df.target_time.data
        laser_pulse_width = df.laser_pulse_width.data
        window_bnd = df.window_bnd.data
        background = df.background.data

        target_amplitude_lst.append(target_amplitude)

        # flight_time, n_shots, t_det_lst = dorg.data_organize(dt, load_dir, fname, window_bnd, max_lsr_num, max_det_num,
        #                                                      set_max_det, exclude_shots)
        print('\n{}:'.format(fname_sim_pkl))
        print('Number of detections: {}'.format(len(flight_time)))
        print('Number of laser shots: {}'.format(n_shots))

        try:
            t_phot_fit_tnsr, t_phot_val_tnsr, t_phot_eval_tnsr,\
                t_det_lst_fit, t_det_lst_val, \
                    n_shots_fit, n_shots_val, n_shots_eval = fit.generate_fit_val_eval(flight_time, flight_time_ref, t_det_lst, n_shots, n_shots_ref)
        except:
            ZeroDivisionError
            print('ERROR: Insufficient laser shots... increase the "max_lsr_num" parameter.')
            exit()

        # Generate "active-ratio histogram" that adjusts the histogram proportionally according to how many bins the detector was "active vs dead"
        if not include_deadtime:
            active_ratio_hst_fit = torch.ones(len(bin_edges-1))
            active_ratio_hst_val = torch.ones(len(bin_edges-1))
        else:
            active_ratio_hst_fit = fit.deadtime_noise_hist(t_min, t_max, intgrl_N, deadtime, t_det_lst_fit, n_shots_fit)
            active_ratio_hst_val = fit.deadtime_noise_hist(t_min, t_max, intgrl_N, deadtime, t_det_lst_val, n_shots_val)
        percent_active = torch.sum(active_ratio_hst_fit).item()/len(active_ratio_hst_fit)
        percent_active_lst.append(percent_active)

        # Optimization process
        if single_step_iter:
            M_lst = np.arange(1, M_max, 1)
        else:
            M_lst = M_lst
            M_max = max(M_lst)

        # Run fit optimizer
        ax, val_loss_arr, eval_loss_arr, \
            fit_rate_fine, coeffs, C_scale_arr = fit.optimize_fit(M_max, M_lst, t_fine, t_phot_fit_tnsr, t_phot_val_tnsr,
                                                    t_phot_eval_tnsr, active_ratio_hst_fit, active_ratio_hst_val,
                                                    active_ratio_hst_ref, n_shots_fit, n_shots_val, n_shots_eval,
                                                    learning_rate, rel_step_lim, intgrl_N,
                                                    max_epochs, term_persist)

        ax.set_ylabel('Loss')
        ax.set_xlabel('Iterations')
        ax.set_title('file: {}'.format(fname_sim_pkl[k]))
        plt.suptitle('Fit loss')
        plt.tight_layout()
        ax.legend()

        print('Validation loss for\n')
        for i in range(len(M_lst)):
            print('Order {}: {:.5f}'.format(M_lst[i], val_loss_arr[M_lst[i]]))

        print('Evaluation loss for\n')
        for i in range(len(M_lst)):
            print('Order {}: {:.5f}'.format(M_lst[i], eval_loss_arr[M_lst[i]]))

        # Choose order to investigate
        minx, miny = np.nanargmin(val_loss_arr), np.nanmin(val_loss_arr)
        min_order = minx
        try:
            model = coeffs[min_order, 0:min_order + 1]
            for i in range(min_order + 1):
                print('Final C{}: {:.4f}'.format(i, model[i]))
        except:
            print("\nERROR: Order exceeds maximum complexity iteration value.\n")

        val_final_loss_lst.append(val_loss_arr[min_order])
        eval_final_loss_lst.append(eval_loss_arr[min_order])
        C_scale_final.append(C_scale_arr[min_order])

        fig = plt.figure()
        ax = fig.add_subplot(111)

        def gen_rho(A, x, mu, sig, bg):
            return A * np.exp(-1 * (x - mu) ** 2 / 2 / sig ** 2) + bg

        n, bins = np.histogram(true_flight_time, bins=60)
        binwidth = np.diff(bins)[0]
        N = n / binwidth / n_shots  # [Hz] Scaling counts to arrival rate
        center = 0.5 * (bins[:-1] + bins[1:])
        ax.bar(center, N, align='center', width=binwidth, color='r', alpha=0.75, label='incident photons')

        t_fine = np.arange(t_min, t_max, dt)
        true_rho = gen_rho(target_amplitude, t_fine, target_time, laser_pulse_width, background)
        ax.plot(t_fine, true_rho, 'r--', label='true arrival rate')

        n, bins = np.histogram(flight_time, bins=34)
        binwidth = np.diff(bins)[0]
        N = n / binwidth / n_shots  # [Hz] Scaling counts to arrival rate
        center = 0.5 * (bins[:-1] + bins[1:])
        ax.bar(center, N, align='center', width=binwidth, color='b', alpha=0.75, label='detected photons')

        # Arrival rate fit
        t_fine = np.arange(t_min, t_max, dt)
        fit_rate_seg = fit_rate_fine[min_order, :]
        ax.plot(t_fine, fit_rate_seg, 'b--', label='fitted arrival rate')
        ax.set_title('Arrival Rate Fit: {}'.format(files[k]))
        ax.set_xlabel('time [s]')
        ax.set_ylabel('Photon Arrival Rate [Hz]')
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.1, 0.90, 'Polynomial order: {}'.format(min_order), transform=ax.transAxes, fontsize=14,
                verticalalignment='top', bbox=props)
        plt.legend()
        plt.tight_layout()

        rmse = np.sqrt(np.sum(((fit_rate_seg-true_rho)/true_rho)**2)/len(fit_rate_seg))
        # rmse = np.sqrt(np.sum((fit_rate_seg - true_rho) ** 2) / np.sum(true_rho**2) / len(fit_rate_seg))
        rmse_final_lst.append(rmse)

    # hypothetical = 0.1**(OD_ref-np.array(OD_list))
    # print('\nScale factor for OD:')
    # for k in range(stop_idx):
    #     print('{}: Scale Factor {:.3}, Hypothetical {:.3}'.format(OD_list[k], C_scale_final[k], hypothetical[k]))

    # Save to csv file
    if not set_max_det:
        save_csv_file = r'\eval_loss_dtime{}_order{}-{}_ref_shots{:.0E}_use_sim_True.csv'.format(include_deadtime,
                                                                                M_lst[0], M_lst[-1],
                                                                                max_lsr_num_ref)
    else:
        save_csv_file = r'\eval_loss_dtime{}_order{}-{}_detections{:.0E}_use_sim_True.csv'.format(include_deadtime, M_lst[0],
                                                                                M_lst[-1], max_lsr_num_ref)
    headers = ['Target Amplitude', 'RMSE', 'Evaluation Loss', 'Optimal Scaling Factor', 'Average %-age where Detector was Active']
    df_out = pd.concat([pd.DataFrame(target_amplitude_lst), pd.DataFrame(rmse_final_lst),
                        pd.DataFrame(eval_final_loss_lst), pd.DataFrame(C_scale_final),
                        pd.DataFrame(percent_active_lst)], axis=1)
    df_out = df_out.to_csv(save_dir + save_csv_file, header=headers)

    print('Total run time: {} seconds'.format(time.time()-start))

    if not set_max_det:
        save_plt_file = r'\eval_loss_dtime{}_order{}-{}_ref_shots{:.2E}_use_sim_True.png'.format(include_deadtime, M_lst[0], M_lst[-1],
                                                                                max_lsr_num_ref)
    else:
        save_plt_file = r'\eval_loss_dtime{}_order{}-{}_detections{:.2E}_use_sim_True.png'.format(include_deadtime, M_lst[0],
                                                                                     M_lst[-1], max_lsr_num_ref)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(target_amplitude_lst[:stop_idx], eval_final_loss_lst, 'r.')
    ax.set_xlabel('Amplitude Arrival Rate [Hz]')
    ax.set_ylabel('Evaluation loss')
    ax.set_title('Evaluation Loss vs OD')
    ax.set_xscale('log')
    fig.savefig(save_dir + save_plt_file)
    time.sleep(2)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(target_amplitude_lst[:stop_idx], rmse_final_lst, 'r.')
    ax.set_xlabel('Amplitude Arrival Rate [Hz]')
    ax.set_ylabel('RMSE')
    ax.set_title('RMSE')
    ax.set_xscale('log')
    # fig.savefig(save_dir + save_plt_file)
    time.sleep(2)
    plt.show()



