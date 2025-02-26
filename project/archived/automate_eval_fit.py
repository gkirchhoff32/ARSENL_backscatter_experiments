def automate_eval_fit(max_lsr_num_ref, include_deadtime=True):
    # validation_high_OD_iterate.py
    #
    # Grant Kirchhoff
    # Last Updated: 10/24/2022
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
    max_det_num_ref = 2000                       # If set_max_det set to TRUE, include up to a certain number of detections
    set_max_det = False                          # Set TRUE if data limiter is number of detections instead of laser shots.
    deadtime = 25e-9                  # [s] Acquisition deadtime
    use_stop_idx = True               # Set TRUE if you want to use up to the OD value preceding the reference OD
    run_full = True                   # Set TRUE if you want to run the fits against all ODs. Otherwise, it will just load the reference data.
    use_poisson_eval = True           # Set TRUE if you want to use the Poisson model for the evaluation loss

    # Optimization parameters
    rel_step_lim = 1e-8  # termination criteria based on step size
    max_epochs = 300  # maximum number of iterations/epochs
    learning_rate = 1e-1  # ADAM learning rate
    term_persist = 20  # relative step size averaging interval in iterations
    intgrl_N = 10000  # Set number of steps in numerical integration

    # Set iterate to True if you want to iterate through increasing complexity.
    # Otherwise set to False if you want to check a single polynomial order.
    single_step_iter = False
    M_max = 21  # Max polynomial complexity to test if iterating
    M_lst = np.arange(6, 12, 1)

    ########################################################################################################################

    # I define the max/min times as fixed values. They are the upper/lower bounds of the fit.
    # Time vector per shot
    t_min = window_bnd[0]
    t_max = window_bnd[1]
    dt = dt
    t_fine = np.arange(t_min, t_max, dt)

    load_dir = r'C:\Users\Grant\OneDrive - UCB-O365\ARSENL\Experiments\Deadtime_Experiments\Data\Deadtime_Experiments_HiFi'
    save_dir = load_dir + r'/../../Figures/evaluation_loss'
    files = os.listdir(load_dir)

    OD_list = np.zeros(len(files))
    for i in range(len(files)):
        OD_list[i] = float(files[i][2:4]) / 10

    fname_ref = r'\OD30_Dev_0_-_2022-04-15_11.24.55.ARSENL.OD30.ARSENL.nc'
    OD_ref = int(fname_ref[3:5]) / 10
    flight_time_ref, n_shots_ref, t_det_lst_ref = dorg.data_organize(dt, load_dir, fname_ref, window_bnd,
                                                                     max_lsr_num_ref, max_det_num_ref, set_max_det,
                                                                     exclude_shots)
    print('\n{}:'.format(fname_ref[1:5]))
    print('Number of detections (reference): {}'.format(len(flight_time_ref)))
    print('Number of laser shots (reference): {}'.format(n_shots_ref))

    # Generate "active-ratio histogram" that adjusts the histogram proportionally according to how many bins the detector was "active vs dead"
    bin_edges = np.linspace(t_min, t_max, intgrl_N + 1, endpoint=False)
    if use_poisson_eval:
        active_ratio_hst_ref = torch.ones(len(bin_edges - 1))
    else:
        active_ratio_hst_ref = fit.deadtime_noise_hist(t_min, t_max, intgrl_N, deadtime, t_det_lst_ref, n_shots_ref)

    # Fitting routine
    if run_full:
        val_final_loss_lst = []
        eval_final_loss_lst = []
        C_scale_final = []
        percent_active_lst = []
        stop_idx = int(np.where(OD_list == OD_ref)[0])
        if not use_stop_idx:
            stop_idx = 3
        for k in range(stop_idx):
            fname = r'/' + files[k]
            OD_fit = int(fname[3:5]) / 10
            # max_lsr_num = np.floor(max_lsr_num_ref / 10**(OD_ref-OD_fit)).astype(int)
            max_lsr_num = max_lsr_num_ref
            max_det_num = max_det_num_ref
            flight_time, n_shots, t_det_lst = dorg.data_organize(dt, load_dir, fname, window_bnd, max_lsr_num,
                                                                 max_det_num, set_max_det, exclude_shots)
            print('\n{}:'.format(fname[1:5]))
            print('Number of detections: {}'.format(len(flight_time)))
            print('Number of laser shots: {}'.format(n_shots))

            try:
                t_phot_fit_tnsr, t_phot_val_tnsr, t_phot_eval_tnsr,\
                    n_shots_fit, n_shots_val, n_shots_eval = fit.generate_fit_val_eval(flight_time, flight_time_ref,
                                                                                       n_shots, n_shots_ref)
            except:
                ZeroDivisionError
                print('ERROR: Insufficient laser shots... increase the "max_lsr_num" parameter.')
                exit()

            # Generate "active-ratio histogram" that adjusts the histogram proportionally according to how many bins the detector was "active vs dead"
            if not include_deadtime:
                active_ratio_hst = torch.ones(len(bin_edges - 1))
            else:
                active_ratio_hst = fit.deadtime_noise_hist(t_min, t_max, intgrl_N, deadtime, t_det_lst, n_shots)
            percent_active = torch.sum(active_ratio_hst).item()/len(active_ratio_hst)
            percent_active_lst.append(percent_active)

            # Optimization process
            if single_step_iter:
                M_lst = np.arange(1, M_max, 1)
            else:
                M_lst = M_lst
                M_max = max(M_lst)

            # Run fit optimizer
            ax, val_loss_arr, eval_loss_arr, \
                fit_rate_fine, coeffs, C_scale_arr = fit.optimize_fit(M_max, M_lst, t_fine, t_phot_fit_tnsr,
                                                                      t_phot_val_tnsr, t_phot_eval_tnsr,
                                                                      active_ratio_hst, active_ratio_hst_ref,
                                                                      n_shots_fit, n_shots_val, n_shots_eval,
                                                                      learning_rate, rel_step_lim, intgrl_N, max_epochs,
                                                                      term_persist)

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

            n, bins = np.histogram(flight_time, bins=34)
            binwidth = np.diff(bins)[0]
            N = n / binwidth / n_shots  # [Hz] Scaling counts to arrival rate
            center = 0.5 * (bins[:-1] + bins[1:])
            ax.bar(center, N, align='center', width=binwidth, color='b', alpha=0.5)

            # Arrival rate fit
            t_fine = np.arange(t_min, t_max, dt)
            fit_rate_seg = fit_rate_fine[min_order, :]
            ax.plot(t_fine, fit_rate_seg, 'r--')
            ax.set_title('Arrival Rate Fit: OD{}'.format(OD_list[k]))
            ax.set_xlabel('time [s]')
            ax.set_ylabel('Photon Arrival Rate [Hz]')
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax.text(0.1, 0.90, 'Polynomial order: {}'.format(min_order), transform=ax.transAxes, fontsize=14,
                    verticalalignment='top', bbox=props)
            plt.tight_layout()

        hypothetical = 0.1**(OD_ref-np.array(OD_list))
        print('\nScale factor for OD:')
        for k in range(stop_idx):
            print('{}: Scale Factor {:.3}, Hypothetical {:.3}'.format(OD_list[k], C_scale_final[k], hypothetical[k]))

        # Save to csv file
        if not set_max_det:
            save_csv_file = r'\eval_loss_dtime{}_order{}-{}_shots{:.0E}.csv'.format(include_deadtime, M_lst[0],
                                                                                    M_lst[-1], max_lsr_num_ref)
        else:
            save_csv_file = r'\eval_loss_dtime{}_order{}-{}_shots{:.0E}.csv'.format(include_deadtime, M_lst[0],
                                                                                    M_lst[-1], max_lsr_num_ref)
        headers = ['OD', 'Evaluation Loss', 'Optimal Scaling Factor', 'Hypothetical Scaling Factor',
                   'Average %-age where Detector was Active']
        df_out = pd.concat([pd.DataFrame(OD_list), pd.DataFrame(eval_final_loss_lst), pd.DataFrame(C_scale_final),
                            pd.DataFrame(hypothetical), pd.DataFrame(percent_active_lst)], axis=1)
        df_out = df_out.to_csv(save_dir + save_csv_file, header=headers)

        print('Total run time: {} seconds'.format(time.time()-start))

        if not set_max_det:
            save_plt_file = r'\eval_loss_dtime{}_order{}-{}_shots{:.2E}.png'.format(include_deadtime, M_lst[0],
                                                                                    M_lst[-1], max_lsr_num_ref)
        else:
            save_plt_file = r'\eval_loss_dtime{}_order{}-{}_detections{:.2E}.png'.format(include_deadtime, M_lst[0],
                                                                                         M_lst[-1], max_lsr_num_ref)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(OD_list[:stop_idx], eval_final_loss_lst, 'r.')
        ax.set_xlabel('OD')
        ax.set_ylabel('Evaluation loss')
        ax.set_title('Evaluation Loss vs OD')
        fig.savefig(save_dir + save_plt_file)


if __name__ == '__main__':
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    max_lsr_num_lst = np.floor(np.logspace(3, 5, 5)).astype(int)
    include_deadtime = False
    for i in range(len(max_lsr_num_lst)):
        print('Number of laser shots: {}'.format(max_lsr_num_lst[i]))
        automate_eval_fit(max_lsr_num_lst[i], include_deadtime)


