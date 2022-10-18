# Methods for 'fit_histogram_polynomial_complexity_iteration.ipynb'.
#
# Grant Kirchhoff
# Last Updated: 09/20/2022

import torch
import numpy as np
import time
import matplotlib.pyplot as plt

# build the fit model as a NN module
class Fit_Pulse(torch.nn.Module):
    def __init__(self, M, t_min, t_max):
        """
        Instantiate and initialize the fit parameters.
        :param M: (int) Polynomial order
        :param t_min: (float) Lower bound for fit window [s]
        :param t_max: (float) Upper bound for fit window [s]
        """
        super().__init__()
        self.M = M  # Polynomial order
        self.C = torch.nn.Parameter(-1 * torch.ones(M+1, 1, dtype=float))  # Coefficients to be optimized
        self.t_max = t_max  # Fit upper bound
        self.t_min = t_min  # Fit lower bound

    # Helpers for numerical integration (Riemann and trapezoidal method)
    @staticmethod
    def trapezoid(vals, dx):
        trap_intgrl = 2*torch.sum(vals) - vals[0] - vals[-1]
        trap_intgrl *= dx / 2
        return trap_intgrl

    @staticmethod
    def riemann(vals, dx):
        riem_intgrl = torch.sum(vals) * dx
        return riem_intgrl

    def tstamp_condition(self, t, t_min, t_max):
        """
        Transform time tag array into chebyshev polynomial matrix form.
        :param t: (torch.tensor) Time tag array [s]
        :param t_min: (float) Lower bound for fit window [s]
        :param t_max: (float) Upper bound for fit window [s]
        :return:
        t_poly_cheb: (torch.tensor) Chebyshev polynomial matrix of time tags
        """
        t_norm = (t - t_min) / (t_max - t_min)  # Normalize timestamps along [0,1]
        t_poly_cheb = cheby_poly(t_norm, self.M)  # Generate chebyshev timestamp basis
        return t_poly_cheb

    def forward(self, intgrl_N, active_ratio_hst, t, t_intgrl, cheby=True):
        """
        Forward model the profile for input time t of polynomial order M (e.g., x^2 --> M=2).
        Also return the integral.
        Parameters:
        intgrl_N  (int): number of steps in numerical integration \\ []
        active_ratio_hst (torch array): Deadtime-adjusted array ("deadtime_adjust_vals output") \\ [Nx1]
        t (torch array): time stamps (unnormalized if cheby=False, cheby_poly output if cheby=True) \\ [Nx1]
        t_intgrl (torch array): time vector [0,1] as chebyshev polynomial (i.e., cheby_poly output) \\ [intgrl_Nx1]
        cheby (bool): Set true if t is normalized (i.e., output from self.tstamp_condition)
        Returns:
        model_out    (torch array): forward model                    \\ [Nx1]
        integral_out (torch array): finite numerical integral output \\ float
        """

        # orthonormalize by leveraging chebyshev polynomials, then calculate forward model
        if not cheby:
            t_poly_cheb = self.tstamp_condition(t, self.t_min, self.t_max)
        else:
            t_poly_cheb = t * 1
        poly = t_poly_cheb @ self.C
        model_out = torch.exp(poly)  # Forward model

        # calculate the integral
        t_poly_cheb = t_intgrl
        poly = t_poly_cheb @ self.C
        fine_res_model = torch.exp(poly)

        # dt = (self.t_max - self.t_min) / intgrl_N  # Step size
        _, dt = np.linspace(self.t_min, self.t_max, intgrl_N, endpoint=False, retstep=True)
        assert (len(fine_res_model) == len(active_ratio_hst))
        active_ratio_hst.resize_(fine_res_model.size())
        fine_res_model = fine_res_model * active_ratio_hst  # Generate deadtime noise model
        integral_out = self.trapezoid(fine_res_model, dt)  # Numerically integrate

        return model_out, integral_out


def pois_loss(prof, integral):
    """
    Non-homogenous Poisson point process loss function
    """
    return integral-torch.sum(torch.log(prof))

# Chebyshev polynomial matrix generator
def cheby_poly(x, M):
    """
    Parameters:
    x (torch array): Values to be evaluated on in chebyshev polynomial      \\ [Nx1]
    M (int)        : *Highest* order term of polynomial (e.g., x^2 --> M=2) \\ []
    Returns:
    chebyshev polynomial matrix (torch array): Evaluated polynomial \\ [NxM]
    """

    def cheby(x, m):
        """
        Helper to calculate value of specific chebyshev order
        """
        T0 = x ** 0
        T1 = x ** 1
        if m == 0:
            return T0
        elif m == 1:
            return T1
        else:
            return 2 * x * cheby(x, m - 1) - cheby(x, m - 2)

    N = len(x)
    model_out = torch.zeros((N, M+1), dtype=float)
    for i in range(M + 1):
        model_out[:, i] = cheby(x, i)

    return model_out

# Deadtime noise model
# Adjust bin ratios depending on reduced bin availability due to deadtime.
#
# This is done because deadtime reduces available bins following a detection event.
# To accomodate for this, in the loss function the impact each time bin has on the
# numerical integration is proportionally reduced to how long it was active (i.e.,
# unaffected by deadtime).

def deadtime_noise_hist(t_min, t_max, intgrl_N, deadtime, t_det_lst):
    """
    Deadtime adjustment for arrival rate estimate in optimizer.
    Parameters:
    t_min: Window lower bound \\ float
    t_max: Window upper bound \\ float
    intgrl_N (int): Number of bins in integral \\ int
    deadtime: Deadtime interval [sec] \\ float
    t_det_lst (list): Nested list of arrays, where each array contains the detections per laser shot
    Returns:
    active_ratio_hst (torch array): Histogram of deadtime-adjustment ratios for each time bin.
    """

    # Initialize
    bin_edges, dt = np.linspace(t_min, t_max, intgrl_N + 1, endpoint=False, retstep=True)
    active_ratio_hst = np.zeros(len(bin_edges)-1)
    deadtime_n_bins = np.floor(deadtime / dt).astype(int)  # Number of bins that deadtime occupies

    # Iterate through each shot. For each detection event, reduce the number of active bins according to deadtime length.
    for shot_num in range(len(t_det_lst)):
        active_ratio_hst += 1
        detections = t_det_lst[shot_num]

        if detections.size == 0:
            continue  # If no detection event for this shot, then skip
        else:
            for det in detections:
                det_time = det.item()  # Time tag of detection that occurred during laser shot

                # Only include detections that fall within fitting window
                if det_time >= (t_min - deadtime) and det_time <= t_max:
                    det_bin_idx = np.argmin(abs(det_time - bin_edges))  # Bin that detection falls into
                    final_dead_bin = det_bin_idx + deadtime_n_bins  # Final bin index that deadtime occupies

                    # Currently a crutch that assumes "dead" time >> window. Will need to include "wrap around" to be more accurate
                    # If final dead bin surpasses fit window, set it to the window upper bin
                    if final_dead_bin > len(active_ratio_hst):
                        final_dead_bin = len(active_ratio_hst)
                    # If initial dead bin (detection bin) precedes fit window, set it to the window lower bin
                    if det_time < t_min:
                        det_bin_idx = 0
                    active_ratio_hst[det_bin_idx:final_dead_bin+1] -= 1  # Remove "dead" region in active ratio

    active_ratio_hst /= len(t_det_lst)  # Normalize for ratio

    return torch.tensor(active_ratio_hst)

def generate_fit_val_eval(data, data_ref, n_shots, n_shots_ref):
    """
    Generates fit, validation, and evaluation data sets for the fitting routine. Recall (1) Fit set: Dataset used to
    generate the fit; (2) Validation set: Independent dataset used to calculate validation loss; and (3) Evaluation set:
    High fidelity set (e.g., unaffected by deadtime, high OD setting) that is used to calculate evaluation loss.
    :param data: (Nx1) Data used for calculating fit and validation loss
    :param data_ref: (Mx1) Reference data used for calculating evaluation loss
    :param n_shots: (int) Number of laser shots for "data"
    :param n_shots_ref: (int) Number of laser shots for "data_ref"
    :return: t_phot_fit_tnsr: (N/2 x 1) Fit set (torch tensor)
    :return: t_phot_val_tnsr: (N/2 x 1) Validation set (torch tensor)
    :return: t_phot_eval_tnsr: (Mx1) Evaluation set (torch tensor)
    :return: n_shots_fit: Number of laser shots for fit set
    :return: n_shots_val: Number of laser shots for validation set
    :return: n_shots_eval: Number of laser shots for evaluation set
    """

    # The target is assumed to be stationary, so I can split the data into halves
    split_value = int(len(data) // 2)
    t_phot_fit = data[:split_value]
    t_phot_val = data[split_value:]
    t_phot_eval = data_ref[:]

    # Adjust number of laser shots corresponding to fit and val sets
    ratio_fit_split = len(t_phot_fit) / len(data)
    ratio_val_split = len(t_phot_val) / len(data_ref)
    n_shots_fit = np.floor(n_shots * ratio_fit_split).astype(int)
    n_shots_val = np.floor(n_shots * ratio_val_split).astype(int)
    n_shots_eval = n_shots_ref

    t_phot_fit_tnsr = torch.tensor(t_phot_fit.to_numpy())
    t_phot_val_tnsr = torch.tensor(t_phot_val.to_numpy())
    t_phot_eval_tnsr = torch.tensor(t_phot_eval.to_numpy())

    return t_phot_fit_tnsr, t_phot_val_tnsr, t_phot_eval_tnsr, n_shots_fit, n_shots_val, n_shots_eval


def optimize_fit(M_max, M_lst, t_fine, t_phot_fit_tnsr, t_phot_val_tnsr, t_phot_eval_tnsr, active_ratio_hst,
                active_ratio_hst_ref, n_shots_fit, n_shots_val, n_shots_eval, learning_rate=1e-1, rel_step_lim=1e-8,
                intgrl_N=10000, max_epochs=400, term_persist=20):

    t_min, t_max = t_fine[0], t_fine[-1]

    iter_len = len(M_lst)
    val_loss_arr = np.zeros(M_max + 1)
    eval_loss_arr = np.zeros(M_max + 1)
    coeffs = np.zeros((M_max + 1, M_max + 1))
    fit_rate_fine = np.zeros((M_max + 1, len(t_fine)))
    C_scale_arr = np.zeros(M_max + 1)
    print('Time elapsed:\n')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    # Iterate through increasing polynomial complexity.
    # Compare fit w/ validation set and use minimum loss find optimal polynomial order.
    for i in range(len(M_lst)):
        # initialize for fit loop
        M = M_lst[i]  # Polynomial order  (e.g., x^2 --> M=2)
        fit_model = Fit_Pulse(M, t_min, t_max)
        optimizer = torch.optim.Adam(fit_model.parameters(), lr=learning_rate)
        epoch = 0
        rel_step = 1e3 * rel_step_lim
        fit_loss_lst = []
        rel_step_lst = []

        init_C = np.zeros(M + 1)
        for j in range(M + 1):
            init_C[j] = fit_model.C[j].item()

        # set the loss function to use a Poisson point process likelihood function
        loss_fn = pois_loss

        # perform fit
        start = time.time()
        t_fit_norm = fit_model.tstamp_condition(t_phot_fit_tnsr, t_min, t_max)
        t_val_norm = fit_model.tstamp_condition(t_phot_val_tnsr, t_min, t_max)
        t_eval_norm = fit_model.tstamp_condition(t_phot_eval_tnsr, t_min, t_max)
        t_intgrl = cheby_poly(torch.linspace(0, 1, intgrl_N), M)
        while rel_step > rel_step_lim and epoch < max_epochs:
            fit_model.train()
            pred_fit, integral_fit = fit_model(intgrl_N, active_ratio_hst, t_fit_norm, t_intgrl, cheby=True)
            loss_fit = loss_fn(pred_fit, integral_fit * n_shots_fit)  # add regularization here
            fit_loss_lst += [loss_fit.item()]

            # calculate relative step as an average over the last term_persist iterations
            if epoch == 0:
                rel_step_lst += [1e3 * rel_step_lim]
                rel_step = 1e3 * rel_step_lim
            else:
                rel_step_lst += [(fit_loss_lst[-2] - fit_loss_lst[-1]) / np.abs(fit_loss_lst[-2])]
                rel_step = np.abs(np.array(rel_step_lst)[-term_persist:].mean())

            # update estimated parameters
            loss_fit.backward()
            optimizer.step()

            # zero out the gradient for the next step
            optimizer.zero_grad()

            epoch += 1

        pred_mod_seg, __ = fit_model(intgrl_N, active_ratio_hst, torch.tensor(t_fine), t_intgrl, cheby=False)
        fit_rate_fine[M, :] = pred_mod_seg.detach().numpy().T
        coeffs[M, 0:M + 1] = fit_model.C.detach().numpy().T

        # Calculate validation loss
        # Using fit generated from fit set, calculate loss when applied to validation set
        pred_val, integral_val = fit_model(intgrl_N, active_ratio_hst, t_val_norm, t_intgrl, cheby=True)
        loss_val = loss_fn(pred_val, integral_val * n_shots_val)
        val_loss_arr[M] = loss_val

        # Now use the generated fit and calculate loss against evaluation set (e.g., no deadtime, high-OD data)
        # When evaluating, I don't want to use the deadtime model as my evaluation metric. So I will use the Poisson loss function.
        # To accommodate, I will remove the active_ratio_hst_ref which is what incorporates the deadtime.
        active_ratio_hst_ref = torch.ones(len(active_ratio_hst_ref))
        pred_eval, integral_eval = fit_model(intgrl_N, active_ratio_hst_ref, t_eval_norm, t_intgrl, cheby=True)

        # If the number of shots between evaluation set and validation set differ, then arrival rate needs to be scaled accordingly.
        n_det_eval = len(pred_eval)
        C_scale = n_det_eval / n_shots_eval / integral_eval
        loss_eval = loss_fn(C_scale * pred_eval, C_scale * integral_eval * n_shots_eval)
        eval_loss_arr[M] = loss_eval
        C_scale_arr[M] = C_scale

        end = time.time()
        print('Order={}: {:.2f} sec'.format(M, end - start))

        ax.plot(fit_loss_lst, label='Order {}'.format(M))

    return ax, val_loss_arr, eval_loss_arr, fit_rate_fine, coeffs, C_scale_arr







### Graveyard ###

# def loss_lse(f1, f2):
#     # LSE for 'C_optimize'
#     return 0.5*(f1 - f2)**2

# def C_optimize(loss1, loss_fn, ratio_step=0.99999, max_epochs=1000):
#     """
#     Calculate optimal scaling constant for arrival rate to calculate costs between two datasets with mismatching laser shots. For example, sets w/ different OD values are not comparable in the MLE loss function. There is a scaling that needs to happen to compare the two.
#     Parameters:
#     t_min: Window lower bound \\ float
#     t_max: Window upper bound \\ float
#     intgrl_N (int): Number of bins in integral \\ int
#     deadtime: Deadtime interval [sec] \\ float
#     t_det_lst (list): Nested list of arrays, where each array contains the detections per laser shot
#     Returns:
#     active_ratio_hst (torch array): Histogram of deadtime-adjustment ratios for each time bin.
#     """
#     epoch = 0
#     alpha = 0.00000000001
#     C = 1
#     while ratio_step<=0.99999 and epoch<max_epochs:
#         n_det_no_dtime = len(pred_no_dtime)
#         loss2 = loss_fn(C*pred_no_dtime, C*integral_no_dtime*n_shots_no_dtime)
#         cost = loss_lse(loss1, loss2)
#         step = -2*(loss1-loss2)*(n_shots_no_dtime*integral_no_dtime - n_det_no_dtime/C)
#         C = C - alpha*step

#         cost_lst.append(cost.item())
#         C_lst.append(C.item())

#         if epoch!=0:
#             ratio_step = C_lst[-1]/C_lst[-2]

#         epoch += 1

#     return C


