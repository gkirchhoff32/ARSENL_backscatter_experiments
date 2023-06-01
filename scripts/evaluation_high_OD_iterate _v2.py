# validation_high_OD_iterate.py
#
# Grant Kirchhoff
# Last Updated: 03/09/2023 (actually last time I bothered to edit this haha)
"""
Automation script to loop through different OD datasets and evaluate fit performance against an evaluation dataset
(i.e., high OD setting).

IMPORTANT: Make sure to edit the parameters in the "PARAMETERS" section before running.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import sys
import time
import pandas as pd
import pickle

start = time.time()

cwd = os.getcwd()
dirLib = cwd + r'/library'
if dirLib not in sys.path:
    sys.path.append(dirLib)
from pathlib import Path

import fit_polynomial_utils as fit
import data_organize as dorg
from load_ARSENL_data import set_binwidth

########################################################################################################################

### CONSTANTS ####
c = 2.99792458e8  # [m/s] Speed of light

# EDIT THESE PARAMETERS BEFORE RUNNING!
### PARAMETERS ###
exclude_shots = True  # Set TRUE to exclude data to work with smaller dataset (enables 'max_lsr_num_fit_ref' variables)
max_lsr_num_ref = int(9.999e6)  # Maximum number of laser shots for the reference dataset
max_lsr_num_fit = int(9.999e4)  # Maximum number of laser shots for the fit dataset
use_final_idx = False  # Set TRUE if you want to use up to the OD value preceding the reference OD
start_idx = 9  # If 'use_final_idx' FALSE, set the min idx value to this value (for troubleshooting purposes)
stop_idx = 10  # If 'use_final_idx' FALSE, set the max+1 idx value to this value (for troubleshooting purposes)
run_full = True  # Set TRUE if you want to run the fits against all ODs. Otherwise, it will just load the reference data
include_deadtime = True  # Set TRUE to include deadtime in noise model
use_sim = True  # Set TRUE if using simulated data
repeat_run = False  # Set TRUE if repeating processing with same parameters but with different data subsets (e.g., fit number is 1e3 and processing first 1e3 dataset, then next 1e3 dataset, etc.)
repeat_range = np.arange(1, 6)  # If 'repeat_run' is TRUE, these are the indices of the repeat segments (e.g., 'np.arange(1,3)' and 'max_lsr_num_fit=1e2' --> run on 1st-set of 100, then 2nd-set of 100 shots.

window_bnd = [28e-9, 34e-9]  # [s] Set boundaries for binning to exclude outliers
if use_sim:
    deadtime = 25e-9  # [s] simulated deadtime
else:
    deadtime = 29.1e-9  # [s] Acquisition deadtime (25ns for PicoQuant boards, 29.1ns for Excelitas SPCM)
dt = 25e-12  # [s] TCSPC resolution

# Optimization parameters
# rel_step_lim = 1e-8  # termination criteria based on step size
rel_step_lim = 1e-8  # termination criteria based on step size
# max_epochs = 10000  # maximum number of iterations/epochs
max_epochs = 10000  # maximum number of iterations/epochs
learning_rate = 1e-1  # ADAM learning rate
term_persist = 20  # relative step size averaging interval in iterations

# Polynomial orders (min and max) to be iterated over in specified step size in the optimizer
M_min = 5
# M_max = 22
M_max = 22
step = 1
M_lst = np.arange(M_min, M_max, step)

if not repeat_run:
    repeat_range = np.array([1])

### PATH VARIABLES ###
home = str(Path.home())
load_dir = home + r'\OneDrive - UCB-O365\ARSENL\Experiments\SPCM\Data\Simulated\subset'  # Where the data is loaded from
save_dir = load_dir + r'\..\..\..\evaluation_loss'  # Where the evaluation loss outputs will be saved
fname_ref = r'\sim_amp1.0E+06_nshot1.0E+07.nc'  # The dataset that will serve as the high-fidelity reference when evaluating

# Generate list of ODs used in the file directory
if use_sim:
    rho_ref = float(fname_ref[8:15])
else:
    OD_ref = int(fname_ref[3:5]) / 10
files = os.listdir(load_dir)
if use_sim:
    rho_list = np.zeros(len(files))
    for i in range(len(files)):
        rho_val = float(files[i][7:14])
        rho_list[i] = rho_val
    min_idx = np.where(rho_list == min(rho_list))[0][0]
    max_idx = np.where(rho_list == max(rho_list))[0][0]
else:
    OD_list = np.zeros(len(files))
    for i in range(len(files)):
        OD_list[i] = float(files[i][2:4]) / 10
    min_idx = np.where(OD_list == min(OD_list))[0][0]
    max_idx = np.where(OD_list == np.unique(OD_list)[-2])[0][0]

if run_full and use_final_idx:
    if use_sim:
        start_idx = 1
        stop_idx = len(files)
    else:
        start_idx = 0
        stop_idx = len(files) - 1

# Save file name for important outputs (to csv and pickle object). These are used by scripts like "plot_eval_loss.ipynb"
save_csv_file = r'\eval_loss_dtime{}_{}{:.1E}-{:.1E}_order{}-{}_shots{:.2E}_use_final_{}.csv'.format(include_deadtime, 'Rho' if use_sim else 'OD',
                                                                                        rho_list[min_idx] if use_sim else OD_list[min_idx],
                                                                                        rho_list[max_idx] if use_sim else OD_list[max_idx],
                                                                                        M_min, M_max-1, max_lsr_num_fit, use_final_idx)
save_csv_file_fit = r'\eval_loss_dtime{}_{}{:.1E}-{:.1E}_order{}-{}_shots{:.2E}_use_final_{}_best_fit.csv'.format(include_deadtime, 'Rho' if use_sim else 'OD',
                                                                                                     rho_list[min_idx] if use_sim else OD_list[min_idx],
                                                                                                     rho_list[max_idx] if use_sim else OD_list[max_idx],
                                                                                                     M_min, M_max-1,
                                                                                                     max_lsr_num_fit, use_final_idx)
save_dframe_fname = r'\fit_figures\eval_loss_dtime{}_{}{:.1E}-{:.1E}_order{}-{}' \
                     '_ref_shots{:.2E}_lsr_shots{:.2E}_use_final_{}_best_fit.pkl'.format(include_deadtime, 'Rho' if use_sim else 'OD',
                                                                            rho_list[min_idx] if use_sim else OD_list[min_idx],
                                                                            rho_list[max_idx] if use_sim else OD_list[max_idx],
                                                                            M_min, M_max-1, max_lsr_num_ref,
                                                                            max_lsr_num_fit, use_final_idx)

########################################################################################################################

# I define the max/min times as fixed values. They are the upper/lower bounds of the fit.
t_min = window_bnd[0]
t_max = window_bnd[1]
t_fine = np.arange(t_min, t_max, dt)

intgrl_N = len(t_fine)

flight_time_ref, n_shots_ref, t_det_lst_ref = dorg.data_organize(dt, load_dir, fname_ref, window_bnd, max_lsr_num_ref,
                                                                 exclude_shots)
if use_sim:
    print('\n{}'.format(fname_ref[1:15]))
else:
    print('\n{}:'.format(fname_ref[1:5]))
print('Number of detections (reference): {}'.format(len(flight_time_ref)))
print('Number of laser shots (reference): {}'.format(n_shots_ref))

# Generate "active-ratio histogram" that adjusts the histogram proportionally according to how many bins the detector
# was "active vs dead"
bin_edges = np.linspace(t_min, t_max, intgrl_N+1, endpoint=False)
if not include_deadtime:
    active_ratio_hst_ref = torch.ones(len(bin_edges)-1)
else:
    active_ratio_hst_ref = fit.deadtime_noise_hist(t_min, t_max, intgrl_N, deadtime, t_det_lst_ref, n_shots_ref)

# Fitting routine
if run_full:
    for j in range(len(repeat_range)):
        val_final_loss_lst = []
        eval_final_loss_lst = []
        C_scale_final = []
        percent_active_lst = []
        fit_rate_seg_lst = []
        flight_time_lst = []
        active_ratio_hst_lst = []

        for k in np.arange(start_idx, stop_idx):
            # if k == skip_idx:
            #     continue
            fname = r'/' + files[k]
            # Obtain the OD value from the file name. Follow the README guide to ascertain the file naming convention
            flight_time, n_shots, t_det_lst = dorg.data_organize(dt, load_dir, fname, window_bnd, max_lsr_num_fit, exclude_shots, repeat_range[j])
            if use_sim:
                print('\n{}'.format(fname[1:15]))
            else:
                print('\n{}:'.format(fname[1:5]))
            print('Number of detections: {}'.format(len(flight_time)))
            print('Number of laser shots: {}'.format(n_shots))

            try:
                t_phot_fit_tnsr, t_phot_val_tnsr, t_phot_eval_tnsr, \
                t_det_lst_fit, t_det_lst_val, n_shots_fit, \
                n_shots_val, n_shots_eval = fit.generate_fit_val_eval(flight_time, flight_time_ref, t_det_lst, n_shots, n_shots_ref)
            except:
                ZeroDivisionError
                print('ERROR: Insufficient laser shots... increase the "max_lsr_num_fit" parameter.')
                exit()

            # Generate "active-ratio histogram" that adjusts the histogram proportionally according to how many bins the detector was "active vs dead"
            if not include_deadtime:
                active_ratio_hst_fit = torch.ones(len(bin_edges)-1)
                active_ratio_hst_val = torch.ones(len(bin_edges)-1)
            else:
                active_ratio_hst_fit = fit.deadtime_noise_hist(t_min, t_max, intgrl_N, deadtime, t_det_lst_fit, n_shots_fit)
                active_ratio_hst_val = fit.deadtime_noise_hist(t_min, t_max, intgrl_N, deadtime, t_det_lst_val, n_shots_val)
            percent_active = torch.sum(active_ratio_hst_fit).item()/len(active_ratio_hst_fit)
            percent_active_lst.append(percent_active)

            # Run fit optimizer
            ax, val_loss_arr, eval_loss_arr, \
            fit_rate_fine, coeffs, \
            C_scale_arr = fit.optimize_fit(M_max, M_lst, t_fine, t_phot_fit_tnsr, t_phot_val_tnsr, t_phot_eval_tnsr,
                                           active_ratio_hst_fit, active_ratio_hst_val, active_ratio_hst_ref, n_shots_fit,
                                           n_shots_val, n_shots_eval, learning_rate, rel_step_lim, intgrl_N, max_epochs,
                                           term_persist)

            ax.set_ylabel('Loss')
            ax.set_xlabel('Iterations')
            ax.set_title('{}{:.2E}'.format('True Rho = ' if use_sim else 'OD = ', rho_list[k] if use_sim else +OD_list[k]))
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
                model = coeffs[min_order, 0:min_order+1]
                for i in range(min_order+1):
                    print('Final C{}: {:.4f}'.format(i, model[i]))
            except:
                print("\nERROR: Order exceeds maximum complexity iteration value.\n")

            val_final_loss_lst.append(val_loss_arr[min_order])
            eval_final_loss_lst.append(eval_loss_arr[min_order])
            C_scale_final.append(C_scale_arr[min_order])

            # Arrival rate fit
            fit_rate_seg = fit_rate_fine[min_order, :]

            if not repeat_run:
                fig = plt.figure()
                ax = fig.add_subplot(111)

                bin_array = set_binwidth(t_min, t_max, dt)
                n, bins = np.histogram(flight_time, bins=bin_array)
                binwidth = np.diff(bins)[0]
                N = n / binwidth / n_shots  # [Hz] Scaling counts to arrival rate
                center = 0.5 * (bins[:-1] + bins[1:])
                ax.bar(center, N, align='center', width=binwidth, color='b', alpha=0.5)
                ax.plot(t_fine, fit_rate_seg, 'r--')
                ax.set_title('Arrival Rate Fit: {}{:.2E}'.format('True Rho = ' if use_sim else 'OD = ', rho_list[k] if use_sim else +OD_list[k]))
                ax.set_xlabel('time [s]')
                ax.set_ylabel('Photon Arrival Rate [Hz]')
                props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                ax.text(0.1, 0.90, 'Polynomial order: {}'.format(min_order), transform=ax.transAxes, fontsize=14,
                        verticalalignment='top', bbox=props)
                plt.tight_layout()

            fit_rate_seg_lst.append(fit_rate_seg)
            flight_time_lst.append(flight_time)
            active_ratio_hst_lst.append(active_ratio_hst_fit)

        # Save to csv file
        headers = ['{}'.format('Rho' if use_sim else 'OD'), 'Evaluation Loss', 'Optimal Scaling Factor', 'Average %-age where Detector was Active']
        if use_sim:
            df_out = pd.concat([pd.DataFrame(rho_list), pd.DataFrame(eval_final_loss_lst), pd.DataFrame(C_scale_final),
                                pd.DataFrame(percent_active_lst)], axis=1)
        else:
            df_out = pd.concat([pd.DataFrame(OD_list), pd.DataFrame(eval_final_loss_lst), pd.DataFrame(C_scale_final),
                                pd.DataFrame(percent_active_lst)], axis=1)
        if repeat_run:
            save_csv_file_temp = save_csv_file[:-4] + '_run#{}.csv'.format(repeat_range[j])
            save_csv_file_fit_temp = save_csv_file_fit[:-4] + '_run#{}.csv'.format(repeat_range[j])
            save_dframe_fname_temp = save_dframe_fname[:-4] + '_run#{}.pkl'.format(repeat_range[j])
        else:
            save_csv_file_temp = save_csv_file
            save_csv_file_fit_temp = save_csv_file_fit
            save_dframe_fname_temp = save_dframe_fname

        df_out = df_out.to_csv(save_dir + save_csv_file_temp, header=headers)

        if use_sim:
            headers = ['Rho' + str(i) for i in rho_list[start_idx:stop_idx]]
        else:
            headers = ['OD'+str(i) for i in OD_list[start_idx:stop_idx]]
        headers.insert(0, 'time vector')
        df_out = pd.DataFrame(np.array(fit_rate_seg_lst).T.tolist())
        df_out = pd.concat([pd.DataFrame(t_fine), df_out], axis=1)
        df_out = df_out.to_csv(save_dir + save_csv_file_fit_temp, header=headers)

        dframe = [flight_time_lst, flight_time_ref, t_min, t_max, dt, n_shots, n_shots_ref, active_ratio_hst_lst]
        pickle.dump(dframe, open(save_dir+save_dframe_fname_temp, 'wb'))

        print('Total run time: {} seconds'.format(time.time()-start))

        if not repeat_run:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            if use_sim:
                ax.semilogx(rho_list[start_idx:stop_idx], eval_final_loss_lst, 'r.')
            else:
                ax.plot(OD_list[start_idx:stop_idx], eval_final_loss_lst, 'r.')
            ax.set_xlabel('{}'.format('Rho [Hz]' if use_sim else 'OD'))
            ax.set_ylabel('Evaluation loss')
            ax.set_title('Evaluation Loss vs OD')
            time.sleep(2)
            plt.show()



