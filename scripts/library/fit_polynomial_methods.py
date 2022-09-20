# Methods for 'fit_histogram_polynomial_complexity_iteration.ipynb'.
#
# Grant Kirchhoff
# Last Updated: 09/20/2022

import torch
import numpy as np

# build the fit model as a NN module
class Fit_Pulse(torch.nn.Module):
    def __init__(self, M, t_min, t_max):
        """
        Instantiate and initialize the fit parameters.
        """
        super().__init__()
        self.M = M  # Polynomial order
        self.C = torch.nn.Parameter(-1 * torch.ones(M + 1, 1, dtype=float))  # Coefficients to be optimized
        self.t_max = t_max  # Fit upper bound
        self.t_min = t_min  # Fit lower bound

    # Helpers for numerical integration (Riemann and trapezoidal method)
    @staticmethod
    def trapezoid(vals, dx):
        trap_intgrl = 2 * torch.sum(vals) - vals[0] - vals[-1]
        trap_intgrl *= dx / 2
        return trap_intgrl

    @staticmethod
    def riemann(vals, dx):
        riem_intgrl = torch.sum(vals) * dx
        return riem_intgrl

    def tstamp_condition(self, t, t_min, t_max):
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
            t_norm = (t - self.t_min) / (self.t_max - self.t_min)  # Normalize timestamps along [0,1]
            t_poly_cheb = cheby_poly(t_norm, self.M)  # Generate chebyshev timestamp basis
        else:
            t_poly_cheb = t
        poly = t_poly_cheb @ self.C
        model_out = torch.exp(poly)  # Forward model

        # calculate the integral
        t_poly_cheb = t_intgrl
        poly = t_poly_cheb @ self.C
        eval_model = torch.exp(poly)

        dt = (self.t_max - self.t_min) / intgrl_N  # Step size
        assert (len(eval_model) == len(active_ratio_hst))
        active_ratio_hst.resize_(eval_model.size())
        eval_model = eval_model * active_ratio_hst  # Generate deadtime noise model
        integral_out = self.trapezoid(eval_model, dt)  # Numerically integrate

        return model_out, integral_out


def pois_loss(prof,integral):
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
    model_out = torch.zeros((N, M + 1), dtype=float)
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
    active_ratio_hst = np.zeros(len(bin_edges) - 1)
    deadtime_n_bins = np.floor(deadtime / dt).astype(int)  # Number of bins that deadtime occupies

    # Iterate through each shot. For each detection event, reduce the number of active bins according to deadtime length.
    for shot_num in range(len(t_det_lst)):
        active_ratio_hst += 1
        total_det = t_det_lst[shot_num]

        if total_det.size == 0:
            continue  # If no detection event for this shot, then skip
        else:
            for det in total_det:
                det_time = det.item()  # Time tag of detection that occurred during laser shot

                # Only include detections that fall within fitting window
                if det_time >= (t_min - deadtime) and det_time <= t_max:
                    det_bin_idx = np.argmin(abs(det_time - bin_edges))  # Bin that detection falls into
                    final_dead_bin = det_bin_idx + deadtime_n_bins  # Final bin index that deadtime occupies

                    # Currently a crutch that assumes "dead" time >> active time. Will need to include "wrap around" to be more accurate
                    # If final dead bin surpasses fit window, set it to the window upper bin
                    if final_dead_bin > len(active_ratio_hst):
                        final_dead_bin = len(active_ratio_hst)
                    # If initial dead bin (detection bin) precedes fit window, set it to the window lower bin
                    if det_time < t_min:
                        det_bin_idx = 0
                    active_ratio_hst[det_bin_idx:final_dead_bin + 1] -= 1  # Remove "dead" region in active ratio

    active_ratio_hst /= len(t_det_lst)  # Normalize for ratio

    return torch.tensor(active_ratio_hst)

