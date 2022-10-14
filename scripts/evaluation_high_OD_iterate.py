# validation_high_OD_iterate.py
#
# Grant Kirchhoff
# Last Updated: 10/13/2022
"""
Automation script to loop through different OD datasets and evaluate fit performance against an evaluation dataset
(i.e., high OD setting).

Note to user: Make sure to edit the parameters in the Parameters section before running.
TODO: Make a guide that describes the parameter functions.
"""

import numpy as np
import time
import matplotlib.pyplot as plt
import torch
import os
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

# EDIT THESE PARAMETERS BEFORE RUNNING!
### PARAMETERS ###
window_bnd = [30e-9, 33e-9]       # [s] Set boundaries for binning to exclude outliers
exclude_shots = True                     # Set TRUE to exclude data to work with smaller dataset
max_num_ref = 10000                   # Include up to certain number of laser shots
deadtime = 25e-9                  # [s] Acquisition deadtime
use_stop_idx = True               # Set TRUE if you want to use the OD value preceding the reference OD
run_full = True                   # Set TRUE if you want to run the fits against all ODs. Otherwise, it will just load the reference data.

load_dir = r'C:\Users\jason\OneDrive - UCB-O365\ARSENL\Experiments\Deadtime_Experiments\Data\Deadtime_Experiments_HiFi'
files = os.listdir(load_dir)

OD_list = np.zeros(len(files))
for i in range(len(files)):
    OD_list[i] = float(files[i][2:4]) / 10

fname_ref = r'\OD30_Dev_0_-_2022-04-15_11.24.55.ARSENL.OD30.ARSENL.nc'
OD_ref = int(fname_ref[3:5]) / 10
flight_time_ref, n_shots_ref, t_det_lst_ref = dorg.data_organize(dt, load_dir, fname_ref, window_bnd, max_num_ref, exclude_shots)
print('\n{}:'.format(fname_ref[1:5]))
print('Number of detections (reference): {}'.format(len(flight_time_ref)))
print('Number of laser shots (reference): {}'.format(n_shots_ref))

if run_full:
    # Optimization parameters
    rel_step_lim = 1e-8  # termination criteria based on step size
    max_epochs = 400  # maximum number of iterations/epochs
    learning_rate = 1e-1  # ADAM learning rate
    term_persist = 20  # relative step size averaging interval in iterations
    intgrl_N = 10000  # Set number of steps in numerical integration

    # I define the max/min times as fixed values. They are the upper/lower bounds of the fit.
    # Time vector per shot
    t_min = window_bnd[0]
    t_max = window_bnd[1]
    dt = dt
    t_fine = np.arange(t_min, t_max, dt)

    # Set iterate to True if you want to iterate through increasing complexity.
    # Otherwise set to False if you want to check a single polynomial order.
    single_step_iter = False
    M_max = 21  # Max polynomial complexity to test if iterating
    M_lst = np.arange(6, 9, 1)

    # Set True to include deadtime in noise model
    include_deadtime = True

    # Generate "active-ratio histogram" that adjusts the histogram proportionally according to how many bins the detector was "active vs dead"
    active_ratio_hst_ref = fit.deadtime_noise_hist(t_min, t_max, intgrl_N, deadtime, t_det_lst_ref)
    if not include_deadtime:
        active_ratio_hst_ref = torch.ones(len(active_ratio_hst_ref))

    stop_idx = int(np.where(OD_list == OD_ref)[0])
    if not use_stop_idx:
        stop_idx = 3
    for k in range(stop_idx):
        fname = r'/' + files[k]
        OD_fit = int(fname[3:5]) / 10
        max_num = np.floor(max_num_ref / 10**(OD_ref-OD_fit)).astype(int)
        flight_time, n_shots, t_det_lst = dorg.data_organize(dt, load_dir, fname, window_bnd, max_num, exclude_shots)
        print('\n{}:'.format(fname[1:5]))
        print('Number of detections: {}'.format(len(flight_time)))
        print('Number of laser shots: {}'.format(n_shots))

        try:
            t_phot_fit_tnsr, t_phot_val_tnsr, t_phot_eval_tnsr,\
                n_shots_fit, n_shots_val, n_shots_eval = fit.generate_fit_val_eval(flight_time, flight_time_ref, n_shots, n_shots_ref)
        except:
            ZeroDivisionError
            print('ERROR: Insufficient laser shots... increase the "max_num" parameter.')
            exit()
        # print('Fit n shots: {}'.format(n_shots_fit))
        # print('Fit data: {}'.format(t_phot_fit_tnsr))

        # Generate "active-ratio histogram" that adjusts the histogram proportionally according to how many bins the detector was "active vs dead"
        active_ratio_hst = fit.deadtime_noise_hist(t_min, t_max, intgrl_N, deadtime, t_det_lst)
        if not include_deadtime:
            active_ratio_hst = torch.ones(len(active_ratio_hst))

        # Optimization process
        if single_step_iter:
            M_lst = np.arange(1, M_max, 1)
        else:
            M_lst = M_lst
            M_max = max(M_lst)

        ax, val_loss_arr, eval_loss_arr, \
            fit_rate_fine, coeffs = fit.optimize_fit(M_max, M_lst, t_fine, t_phot_fit_tnsr, t_phot_val_tnsr,
                                                    t_phot_eval_tnsr, active_ratio_hst,
                                                    active_ratio_hst_ref, n_shots_fit, n_shots_val, n_shots_eval,
                                                    learning_rate, rel_step_lim, intgrl_N,
                                                    max_epochs, term_persist)

        ax.set_ylabel('Loss')
        ax.set_xlabel('Iterations')
        ax.set_title('OD{}'.format(OD_list[k]))
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
        minx, miny = np.argmin(val_loss_arr), min(val_loss_arr)
        order = minx
        try:
            model = coeffs[order, 0:order + 1]
            for i in range(order + 1):
                print('Final C{}: {:.4f}'.format(i, model[i]))
        except:
            print("\nERROR: Order exceeds maximum complexity iteration value.\n")

        fig = plt.figure()
        ax = fig.add_subplot(111)

        n, bins = np.histogram(flight_time, bins=34)
        binwidth = np.diff(bins)[0]
        N = n / binwidth / n_shots  # [Hz] Scaling counts to arrival rate
        center = 0.5 * (bins[:-1] + bins[1:])
        ax.bar(center, N, align='center', width=binwidth, color='b', alpha=0.5)

        # Arrival rate fit
        t_fine = np.arange(t_min, t_max, dt)
        fit_rate_seg = fit_rate_fine[order, :]
        ax.plot(t_fine, fit_rate_seg, 'r--')
        ax.set_title('Arrival Rate Fit')
        ax.set_xlabel('time [s]')
        ax.set_ylabel('Photon Arrival Rate [Hz]')
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.1, 0.90, 'Polynomial order: {}'.format(order), transform=ax.transAxes, fontsize=14,
                verticalalignment='top', bbox=props)
        plt.tight_layout()

    plt.show()

    #     iter_len = len(M_lst)
    #     val_loss_arr = np.zeros(M_max + 1)
    #     eval_loss_arr = np.zeros(M_max + 1)
    #     coeffs = np.zeros((M_max + 1, M_max + 1))
    #     fit_rate_fine = np.zeros((M_max + 1, len(t_fine)))
    #     print('Time elapsed:\n')
    #
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111)
    #     # Iterate through increasing polynomial complexity.
    #     # Compare fit w/ validation set and use minimum loss find optimal polynomial order.
    #     for i in range(len(M_lst)):
    #         # initialize for fit loop
    #         M = M_lst[i]  # Polynomial order  (e.g., x^2 --> M=2)
    #         fit_model = fit.Fit_Pulse(M, t_min, t_max)
    #         optimizer = torch.optim.Adam(fit_model.parameters(), lr=learning_rate)
    #         epoch = 0
    #         rel_step = 1e3 * rel_step_lim
    #         fit_loss_lst = []
    #         val_loss_lst = []
    #         rel_step_lst = []
    #
    #         init_C = np.zeros(M + 1)
    #         for j in range(M + 1):
    #             init_C[j] = fit_model.C[j].item()
    #
    #         # set the loss function to use a Poisson point process likelihood function
    #         loss_fn = fit.pois_loss
    #
    #         # perform fit
    #         start = time.time()
    #         t_fit_norm = fit_model.tstamp_condition(t_phot_fit_tnsr, t_min, t_max)
    #         t_val_norm = fit_model.tstamp_condition(t_phot_val_tnsr, t_min, t_max)
    #         t_eval_norm = fit_model.tstamp_condition(t_phot_eval_tnsr, t_min, t_max)
    #         t_intgrl = fit.cheby_poly(torch.linspace(0, 1, intgrl_N), M)
    #         while rel_step > rel_step_lim and epoch < max_epochs:
    #             fit_model.train()
    #             pred_fit, integral_fit = fit_model(intgrl_N, active_ratio_hst, t_fit_norm, t_intgrl, cheby=True)
    #             loss_fit = loss_fn(pred_fit, integral_fit * n_shots_fit)  # add regularization here
    #             fit_loss_lst += [loss_fit.item()]
    #
    #             # calculate relative step as an average over the last term_persist iterations
    #             if epoch == 0:
    #                 rel_step_lst += [1e3 * rel_step_lim]
    #                 rel_step = 1e3 * rel_step_lim
    #             else:
    #                 rel_step_lst += [(fit_loss_lst[-2] - fit_loss_lst[-1]) / np.abs(fit_loss_lst[-2])]
    #                 rel_step = np.abs(np.array(rel_step_lst)[-term_persist:].mean())
    #
    #             # update estimated parameters
    #             loss_fit.backward()
    #             optimizer.step()
    #
    #             # zero out the gradient for the next step
    #             optimizer.zero_grad()
    #
    #             epoch += 1
    #
    #         pred_mod_seg, __ = fit_model(intgrl_N, active_ratio_hst, torch.tensor(t_fine), t_intgrl, cheby=False)
    #         fit_rate_fine[M, :] = pred_mod_seg.detach().numpy().T
    #         coeffs[M, 0:M + 1] = fit_model.C.detach().numpy().T
    #
    #         # Calculate validation loss
    #         # Using fit generated from fit set, calculate loss when applied to validation set
    #         pred_val, integral_val = fit_model(intgrl_N, active_ratio_hst, t_val_norm, t_intgrl, cheby=True)
    #         loss_val = loss_fn(pred_val, integral_val * n_shots_fit)
    #         val_loss_arr[M] = loss_val
    #
    #         # Now use the generated fit and validate against evaluation set (e.g., no deadtime, high-OD data)
    #         pred_eval, integral_eval = fit_model(intgrl_N, active_ratio_hst_ref, t_eval_norm, t_intgrl, cheby=True)
    #
    #         # If the number of shots between evaluation set and validation set differ, then arrival rate needs to be scaled accordingly.
    #         n_det_eval = len(pred_eval)
    #         C_scale = n_det_eval / n_shots_eval / integral_eval
    #         loss_eval = loss_fn(C_scale * pred_eval, C_scale * integral_eval * n_shots_eval)
    #         eval_loss_arr[M] = loss_eval
    #
    #         end = time.time()
    #         print('Order={}: {:.2f} sec'.format(M, end - start))
    #
    #         ax.plot(fit_loss_lst, label='Order {}'.format(M))
    #
    #     ax.set_ylabel('Loss')
    #     ax.set_xlabel('Iterations')
    #     ax.set_title('OD{}'.format(OD_list[k]))
    #     plt.suptitle('Fit loss')
    #     plt.tight_layout()
    #     ax.legend()
    #
    #     print('\nValidation loss for')
    #     for i in range(len(M_lst)):
    #         print('Order {}: {:.5f}'.format(M_lst[i], val_loss_arr[M_lst[i]]))
    #
    #     print('\nEvaluation loss for')
    #     for i in range(len(M_lst)):
    #         print('Order {}: {:.5f}'.format(M_lst[i], eval_loss_arr[M_lst[i]]))
    #
    #     minx, miny = np.argmin(val_loss_arr), min(val_loss_arr)
    #
    #     # Choose order to investigate
    #     order = minx
    #     try:
    #         model = coeffs[order, 0:order + 1]
    #         for i in range(order + 1):
    #             print('Final C{}: {:.4f}'.format(i, model[i]))
    #     except:
    #         print("\nERROR: Order exceeds maximum complexity iteration value.\n")
    #
    #     # Overlay fit & histogram
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111)
    #
    #     n, bins = np.histogram(flight_time, bins=34)
    #     binwidth = np.diff(bins)[0]
    #     N = n / binwidth / n_shots  # [Hz] Scaling counts to arrival rate
    #     center = 0.5 * (bins[:-1] + bins[1:])
    #     ax.bar(center, N, align='center', width=binwidth, color='b', alpha=0.5)
    #
    #     # Arrival rate fit
    #     t_fine = np.arange(t_min, t_max, dt)
    #     fit_rate_seg = fit_rate_fine[order, :]
    #     ax.plot(t_fine, fit_rate_seg, 'r--')
    #     plt.suptitle('Arrival Rate Fit')
    #     ax.set_title('OD{}'.format(OD_list[k]))
    #     ax.set_xlabel('time [s]')
    #     ax.set_ylabel('Photon Arrival Rate [Hz]')
    #     props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    #     ax.text(0.1, 0.90, 'Polynomial order: {}'.format(order), transform=ax.transAxes, fontsize=14,
    #             verticalalignment='top', bbox=props)
    #     plt.tight_layout()
    #
    # plt.show()



