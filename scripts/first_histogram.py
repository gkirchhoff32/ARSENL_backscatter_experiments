#
# ARSENL Backscatter Experiments
# first_histogram.py
#
# Grant Kirchhoff
# 02-25-2022
# University of Colorado Boulder
#
"""
Histogram photon arrival time data from ARSENL INPHAMIS lidar. IMPORTANT: Set data path settings in
'load_ARSENL_data.py' first.
"""

import numpy as np
import pandas as pd
import time
import pickle
import matplotlib.pyplot as plt

from load_ARSENL_data import load_INPHAMIS_data, data_dir, fname, picklename

start = time.time()
# Constants
c = 299792458  # [m/s] Speed of light

# Parameters
create_csv = 0  # Set true to generate a .csv from .ARSENL data
load_data = True  # Set true to load data into a DataFrame and serialize into a pickle object
irregular_data = 1  # Set true if data has gaps (i.e., dtime is 0 for many clock cycles)
exclude = [20000, 40000]  # Set boundaries for binning

# Load INPHAMIS .ARSENL data if not yet serialized
if load_data:
    load_INPHAMIS_data(data_dir, fname, picklename, create_csv)

# Unpickle the data to DataFrame object
infile = open('{}/{}'.format(data_dir, picklename), 'rb')
df = pickle.load(infile)
infile.close()

df1 = df.loc[df['dtime'] != 0]
detect = df1.loc[(df1['overflow'] == 0) & (df1['channel'] == 0)]  # Return data for detection event ("overflow","channel" = 0,0)

sync_detect_idx = np.array(detect.index) - 1  # Extract index immediately prior to detection event to match with laser pulse
sync_detect = df.loc[sync_detect_idx]  # Laser pulse event prior to detection event
# If data has gaps: Ignore data where a detection event immediately follows a data gap
if irregular_data:
    sync_detect = sync_detect.loc[sync_detect['dtime'] != 0]
    detect = df1.loc[np.array(sync_detect.index + 1)]

detect_time = detect['dtime'].to_numpy()
sync_detect_time = sync_detect['dtime'].to_numpy()

flight_time = (detect_time - sync_detect_time) * 25  # [ps] Time is in segments of 25 ps
flight_time = flight_time[np.where((flight_time >= exclude[0]) & (flight_time < exclude[1]))]  # Exclude t.o.f. where bins ~= 0
distance = flight_time / 1e12 * c / 2

### Plot time of flight as a function of laser shot ###
# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# ax1.plot(flight_time, 'g.')
# ax1.set_xlabel('Laser pulse')
# ax1.set_ylabel('Time of flight [ps]')
# ax1.set_title('Time of flight per laser shot')
# plt.show()

### Histogram of time of flight ###
fig = plt.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
n1, bins1, patches1 = ax1.hist(flight_time, bins=200)
print('Histogram plot time elapsed: {:.3} sec'.format(time.time() - start))
n2, bins2, patches2 = ax2.hist(distance, bins=200)
ax1.set_xlabel('Time of flight [ps]')
ax1.set_title('Time of flight for INPHAMIS backscatter')
ax2.set_xlabel('Range [m]')
ax2.set_title('Detected range for INPHAMIS backscatter')
plt.tight_layout()
plt.show()

