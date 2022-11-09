# generate_sim_data.py
#
# Grant Kirchhoff
# Last Updated: 11/03/2022
"""
Script to generate simulated retrievals and output data products that can be used in the deadtime noise model fitting
routine (e.g., evaluation_high_OD_iterate.py)
"""

import os,sys
import numpy as np
import time
import matplotlib.pyplot as plt
import xarray as xr
import pickle

# import library for simulating Poisson point processes
cwd = os.getcwd()
dirLib = cwd + r'/library'
if dirLib not in sys.path:
    sys.path.append(dirLib)

import sim_deadtime_utils as sim


def gen_sim_data(t_sim_max, dt_sim, tD, Nshot, wrap_deadtime, window_bnd, laser_pulse_width, target_time,
                 target_amplitude, background):
    ##### GENERATE SIMULATED DATA #####

    # simulation resolution settings
    t_sim_min = 0
    t_sim_max = t_sim_max
    dt_sim = dt_sim

    tD = tD  # deadtime
    if type(Nshot) != int:
        Nshot = int(Nshot)  # number of laser shots
    wrap_deadtime = wrap_deadtime  # wrap deadtime between shots

    window_bnd = window_bnd

    # Target parameters (Gaussian convolved w/ Gaussian)
    laser_pulse_width = laser_pulse_width # laser pulse width in seconds
    target_time = target_time
    target_amplitude = target_amplitude  # target peak count rate
    background = background  # background count rate

    # generate the simulated scene time axis
    t_sim = np.arange(t_sim_min, t_sim_max, dt_sim)           # simulation time
    t_sim_bins = np.concatenate((t_sim, t_sim[-1:]+dt_sim))  # simulation time histogram bins

    # generate the photon arrival rate of the profile
    # Gaussian target with constant background
    photon_rate_arr = target_amplitude*np.exp(-(t_sim - target_time)**2/(2*laser_pulse_width**2))+background

    # generate photon counts

    # lists of photon arrivals per laser shot
    start = time.time()
    t_det_lst = []  # detected photons (includes deadtime)
    t_phot_lst = []  # actual photons (no dead time)

    t_det_last = -100.0  # last photon detection event
    for n in range(Nshot):
        # simulate a laser shot
        ptime, ctime = sim.photon_count_generator(t_sim_bins,
                                                  photon_rate_arr,
                                                  tau_d_flt=tD,
                                                  last_photon_flt=t_det_last)
        if wrap_deadtime:
            if len(ctime) > 0:
                t_det_last = ctime[-1]
            t_det_last -= t_sim_bins[-1]

        t_det_lst += [ctime]  # detection time tags (including deadtime)
        t_phot_lst += [ptime]  # photon time tags

    print('time elapsed: {}'.format(time.time() - start))

    flight_time = xr.DataArray(np.concatenate(t_det_lst))
    flight_time = flight_time[np.where((flight_time >= window_bnd[0]) & (flight_time < window_bnd[1]))]  # Exclude specified t.o.f. bins
    n_shots = Nshot
    t_det_lst = t_det_lst

    true_flight_time = xr.DataArray(np.concatenate(t_phot_lst))
    true_flight_time = true_flight_time[np.where((true_flight_time >= window_bnd[0]) & (true_flight_time < window_bnd[1]))]

    return flight_time, true_flight_time, n_shots, t_det_lst, t_phot_lst

if __name__ == '__main__':

    ### PARAMETERS ###

    # simulation resolution settings
    t_sim_min = 0
    t_sim_max = 40e-9
    dt_sim = 1e-12

    tD = 25e-9  # deadtime
    Nshot = int(1e4)  # number of laser shots
    wrap_deadtime = True  # wrap deadtime between shots

    window_bnd = [26e-9, 34e-9]

    laser_pulse_width = 500e-12  # laser pulse width in seconds
    target_time = 31.2e-9
    target_amplitude = 1e10  # target peak count rate
    background = 1e4  # background count rate

    serialize = True


    ### GENERATE SIMULATED DATA ###

    flight_time, true_flight_time, n_shots, t_det_lst, t_phot_lst = gen_sim_data(t_sim_max, dt_sim, tD, Nshot,
                                                                                 wrap_deadtime, window_bnd,
                                                                                 laser_pulse_width, target_time,
                                                                                 target_amplitude, background)

    # Save simulated data to netCDF
    if serialize:
        processed_data = xr.Dataset(
            data_vars=dict(
                flight_time=xr.DataArray(flight_time, dims='flight time'),
                true_flight_time=xr.DataArray(true_flight_time, dims='true flight time'),
                n_shots=n_shots,
                t_det_lst=(['det1', 'det2'], t_det_lst),
                t_phot_lst=('true detections', t_phot_lst),
                target_amplitude=target_amplitude,
                target_time=target_time,
                laser_pulse_width=laser_pulse_width,
                window_bnd=window_bnd,
                background=background
            ),
            attrs=dict(
                description="'flight_time': time tagged data; \n'n_shots': number of laser shots; \n't_det_lst': detections per laser shot in each corresponding row")
        )

        save_dir = r'C:\Users\Grant\OneDrive - UCB-O365\ARSENL\Experiments\Deadtime_Experiments\Data\simulated'
        fname_pkl = r'\sim_amp{:.1E}_nshot{:.1E}.pkl'.format(target_amplitude, Nshot)
        outfile = open(save_dir+fname_pkl, 'wb')
        pickle.dump(processed_data, outfile)
        outfile.close()

    phot_arr = np.array(sorted(np.concatenate(t_phot_lst)))
    plt.figure()
    plt.stem(phot_arr, np.ones(phot_arr.size))
    plt.title('Photons')

    cnt_arr = np.array(sorted(np.concatenate(t_det_lst)))
    plt.figure()
    plt.stem(cnt_arr, np.ones(cnt_arr.size))
    plt.title('Detected Photons')

    # Scaled time-of-flight histogram
    fig = plt.figure()
    ax = fig.add_subplot(111)
    n, bins = np.histogram(flight_time*1e9, bins=30)
    binwidth = np.diff(bins)[0]
    N = n / binwidth / 1e-9 / n_shots  # [Hz] Scaling counts to arrival rate
    center = 0.5 * (bins[:-1] + bins[1:])
    ax.bar(center, N, align='center', width=binwidth, color='b', alpha=0.5, label='detected photons')
    n, bins = np.histogram(true_flight_time*1e9, bins=30)
    binwidth = np.diff(bins)[0]
    N = n / binwidth / 1e-9 / n_shots  # [Hz] Scaling counts to arrival rate
    center = 0.5 * (bins[:-1] + bins[1:])
    ax.bar(center, N, align='center', width=binwidth, color='r', alpha=0.5, label='true photons')
    ax.set_title('Arrival Rate Histogram')
    ax.set_xlabel('time [ns]')
    ax.set_ylabel('Photon Arrival Rate [Hz]')
    plt.legend()
    plt.show()

