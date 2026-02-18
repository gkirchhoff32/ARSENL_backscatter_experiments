#
# ARSENL Backscatter Experiments
# plot_histogram.py
#
# Grant Kirchhoff
# First Created: 02-25-2022
# University of Colorado Boulder
#
"""
Histogram photon arrival time data from ARSENL INPHAMIS lidar. IMPORTANT: Set data path settings in
'load_ARSENL_data.py' first.
"""

import sys
import os
import numpy as np
import time
import pickle
import matplotlib.pyplot as plt
import xarray as xr
from pathlib import Path

from utils.load_ARSENL_data_utils import load_INPHAMIS_data, set_binwidth, data_dir, fname, picklename

start = time.time()
# Constants
c = 299792458  # [m/s] Speed of light

# Parameters
create_csv = False  # Set TRUE to generate a .csv from .ARSENL data
load_data = True  # Set TRUE to load data into a DataFrame and serialize into a pickle object
load_netcdf = True  # Set TRUE if loading from netcdf file ('*.ARSENL.nc'). Set FALSE if loading from *.ARSENL file.
use_donovan = False  # Set TRUE if user wants to scale the histogram by using the Donovan correction
limit_shots = False  # Set TRUE to limit the number of shots used, which activates variable "use_shots"
native_window = False  # Set TRUE if use the window predefined in the simulation
bin_avg = 3  # set number of bins to integrate together during plotting
use_sim = False

# window_bnd = np.array([28e-9, 34e-9])  # [s] Set temporal boundaries for binning
new_window_bnd = np.array([28e-8, 34e-8])  # [s] Set temporal boundaries for binning
# window_bnd = np.array([900, 1200])  # [m] Set boundaries for binning to exclude outliers
# window_bnd = window_bnd / c * 2  # [s] Convert from range to tof
deadtime = 29.5e-9  # [s] Deadtime interval (25ns for sim, 29.1ns for SPCM)
# deadtime = 25e-9  # [s] Deadtime interval (25ns for sim, 29.1ns for SPCM)

if limit_shots:
    use_shots = 10000

print(sys.platform)

if load_netcdf:
    if 'win' in sys.platform:
        home = str(Path.home())
    elif sys.platform == 'linux':
        home = r'/mnt/c/Users/Grant'
    else:
        raise OSError('Check operating system is Windows or Linux')
    # data_dir = os.path.join(home, 'OneDrive - UCB-O365', 'ARSENL', 'Experiments', 'SPCM', 'Data', 'Simulated', 'manuscript_revise_distributed', 'ver1')
    data_dir = os.path.join(
        home,
        'OneDrive - UCB-O365',
        'ARSENL',
        'Experiments',
        'SPCM',
        'Data',
        'Experimental',
        'SPCM_Data_2023.03.06',
        'SPCM_Data_2023.03.06_Subset_Test'
    )
    fname = 'OD50_Dev_0_-_2023-03-06_16.56.00_OD5.0.ARSENL.nc'
    save_fname = 'histogram.jpg'

    open_file_path = os.path.join(data_dir, fname)
    ds = xr.open_dataset(open_file_path)

    cnts = ds.time_tag
    if limit_shots:
        lim_shot_idx = (abs(ds.time_tag_sync_index - use_shots)).argmin().values.item()
        cnts = cnts[:lim_shot_idx]

    if native_window:
        window_bnd = ds.window_bnd.to_numpy()
    else:
        window_bnd = new_window_bnd
    window_bnd = np.array([32e-9, 39e-9])  # [s]

    t_min = window_bnd[0]  # [s]
    t_max = window_bnd[1]  # [s]
    if use_sim:
        dt = ds.dt_sim  # [s]
    else:
        dt = 25e-12  # [s]

    flight_time = cnts * dt  # [s]
    # Exclude specified t.o.f. bins
    flight_time = flight_time[np.where((flight_time >= window_bnd[0]) & (flight_time < window_bnd[1]))]
    if limit_shots:
        n_shots = len(ds.sync_index[:use_shots])
    else:
        n_shots = len(ds.sync_index)

else:
    # Load INPHAMIS .ARSENL data if not yet serialized
    if load_data:
        load_INPHAMIS_data(data_dir, fname, picklename, create_csv)

    # Unpickle the data to DataFrame object
    infile = open('{}/{}'.format(data_dir, picklename), 'rb')
    df = pickle.load(infile)
    infile.close()

    df1 = df.loc[df['dtime'] != 0]
    detect = df1.loc[(df1['overflow'] == 0) & (df1['channel'] == 0)]  # Return data for detection event ("overflow","channel" = 0,0)
    sync = df1.loc[(df1['overflow'] == 1) & (df1['channel'] == 0)]
    n_shots = len(sync)

    sync_detect_idx = np.array(detect.index) - 1  # Extract index immediately prior to detection event to match with laser pulse
    sync_detect = df.loc[sync_detect_idx]  # Laser pulse event prior to detection event

    detect_time = detect['dtime'].to_numpy()
    sync_detect_time = sync_detect['dtime'].to_numpy()

    flight_time = (detect_time - sync_detect_time) * dt  # [s] Time is in segments of 25 ps
    flight_time = flight_time[np.where((flight_time >= t_min) & (flight_time < t_max))]  # window_bnd t.o.f. where bins ~= 0
    distance = flight_time * c / 2

### Histogram of time of flight ###
fig = plt.figure(
    figsize=(10,4),
    dpi=400
)
ax1 = fig.add_subplot(111)
res = dt * bin_avg
bin_array = set_binwidth(t_min, t_max, res)
n, bins = np.histogram(flight_time, bins=bin_array)
print('Histogram plot time elapsed: {:.3} sec'.format(time.time() - start))
binwidth = np.diff(bins)[0]
N = n / binwidth / n_shots
print('Number of shots: {}'.format(n_shots))
if use_donovan:
    N_dono = N / (1 - N*deadtime)
center = 0.5 * (bins[:-1]+bins[1:])
# ax1.bar(center*1e9, N/1e6, align='center', width=binwidth*1e9, color='b', alpha=0.75, label='Detections')
ax1.bar(center*1e9, n, align='center', width=binwidth*1e9, color='b', alpha=0.75, label='Detections')
# ax1.barh(center*c/2/1e3, N/1e6, align='center', height=binwidth*c/2/1e3, color='b', alpha=0.75)
if use_donovan:
    ax1.bar(center*c/2, N_dono, align='center', width=binwidth*c/2, color='r', alpha=0.5, label='Muller "Corrected" Profile')
    ax1.set_title('Inaccurate Muller Correction Demonstration')
    plt.legend()
# ax1.set_ylabel('Range [km]')
# ax1.set_xlabel('Arrival rate [MHz]')
# ax1.set_xscale('log')
# ax1.set_ylim(window_bnd*c/2/1e3)
ax1.set_xlabel('Time of flight [ns]')
# ax1.set_ylabel('Arrival rate [MHz]')
ax1.set_ylabel('Counts')
ax1.set_title('IRF Measurement')
# ax1.set_yscale('log')
ax1.set_xlim(window_bnd*1e9)
plt.tight_layout()
if sys.platform == 'win32':
    plt.show()
else:
    plt.savefig(os.path.join(data_dir, save_fname))

# --- Compute weighted mean and standard deviation ---
x = center  # [s]
w = N  # histogram values (weights)

# Weighted mean
mu = np.sum(w * x) / np.sum(w)

# Weighted variance
var = np.sum(w * (x - mu) ** 2) / np.sum(w)

# Standard deviation
sigma = np.sqrt(var)

print(f"Mean = {mu * 1e9:.3f} ns")
print(f"Std dev = {sigma * 1e9:.3f} ns")

# # Compute FWHM
# y = N
# x = center  # [s]
#
# # Peak and half-maximum
# peak = np.max(y)
# half_max = peak / 2
#
# # Find indices where the histogram crosses half-max
# sign = np.sign(y - half_max)
# crossings = np.where(np.diff(sign))[0]
#
# if len(crossings) >= 2:
#     # Left crossing (linear interpolation)
#     i1 = crossings[0]
#     x1 = x[i1] + (x[i1+1] - x[i1]) * (half_max - y[i1]) / (y[i1+1] - y[i1])
#
#     # Right crossing (linear interpolation)
#     i2 = crossings[-1]
#     x2 = x[i2] + (x[i2+1] - x[i2]) * (half_max - y[i2]) / (y[i2+1] - y[i2])
#
#     fwhm = x2 - x1  # [s]
#     print(f"FWHM = {fwhm*1e9:.3f} ns")
# else:
#     print("Could not determine FWHM — histogram may not be unimodal.")
#
#
