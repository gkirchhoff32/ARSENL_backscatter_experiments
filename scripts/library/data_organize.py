# Method for loading and organizing data necessary for 'fit_histogram_polynomial_complexity_iteration.ipynb'.
#
# Grant Kirchhoff
# Last Updated: 09/21/2022


import numpy as np
import xarray as xr

def data_organize(dt, data_dir, fname, window_bnd, max_lsr_shot, exclude_shots=True):
    '''
    Some bookkeeping. Organizes data into structures and variables required for the fit routine.
    :param dt: (float) Temporal resolution [s]
    :param data_dir: (str) Data directory
    :param fname: (str) Data file name
    :param window_bnd: (2x1 list) Two time bounds to exclude outlying data [s]
    :param max_lsr_shot: (int) Number of maximum laser shots to include going forward (see "exclude_shots")
    :param exclude_shots: (bool) Set True if you want to exclude shots beyond the "max_lsr_shot" parameter
    :return:
    flight_time: (N_{tags}x1 xarray.DataArray) Time tags [s]
    n_shots: (int) Number of laser shots
    t_det_lst: (?xn_shots list) Nested list of xarray.DataArrays, where each DataAarray consists of the detections per laser shot [s]
    '''

    # Load and organize xarray dataset
    ds = xr.open_dataset(data_dir + fname)

    cnts = ds.time_tag
    flight_time = cnts * dt  # [s] Convert "time tags" from clock counts to actual timed tags
    flight_time = flight_time[
        np.where((flight_time >= window_bnd[0]) & (flight_time < window_bnd[1]))]  # Exclude specified t.o.f. bins

    tot_lsr_shots = ds.sync_index.to_numpy()
    ttag_sync_idx = ds.time_tag_sync_index.values

    if exclude_shots:
        excl_sync = ds.sync_index[max_lsr_shot].item()
        excl_ttag_idx = np.where(ttag_sync_idx == excl_sync)[0]
        if excl_ttag_idx.size == 0:
            nearest = ttag_sync_idx[np.argmin(ttag_sync_idx - excl_sync)] - tot_lsr_shots[
                0]  # Subtract first index value to start at 0
            print(
                "Last sync event doesn't correspond to a detection event. Please choose another. Please change 'max_lsr_shot' variable. Here is the closest sync-detection index: {}".format(
                    nearest))
            sys.exit()
        else:
            excl_ttag_idx = excl_ttag_idx[0]

        flight_time = flight_time[0:excl_ttag_idx]
        tot_lsr_shots = tot_lsr_shots[0:max_lsr_shot]
        n_shots = len(tot_lsr_shots)
    else:
        n_shots = len(ds.sync_index)

    # Generate nested list of xarray.DataArrays, where each DataAarray consists of the detections per laser shot
    t_det_lst = []
    for i in range(len(tot_lsr_shots)):
        total_det = ds.time_tag[np.where(ttag_sync_idx == tot_lsr_shots[i])[0]]
        total_det = total_det * dt  # Convert from 25ps intervals to seconds
        t_det_lst.append(total_det)

    return flight_time, n_shots, t_det_lst