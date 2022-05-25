#
# ARSENL Backscatter Experiments
# load_ARSENL_data.py
#
# Grant Kirchhoff
# 02-25-2022
# University of Colorado Boulder
#
"""

Load data from ARSENL INPHAMIS lidar as a DataFrame and serialize as a pickle file. The purpose is to load the large
dataset as a pickle object without having to reload the data every time main() is executed. This reduces execution time
and enables flexibility with the data handling.

"""

import os
import pandas as pd
import time
import pickle

cwd = os.getcwd()
data_dir = cwd + r'/../Data'
fname = r'/Dev_0_-_2022-02-18_10.48.52.ARSENL'
picklename = 'pickle'
create_csv = 0

def load_INPHAMIS_data(data_dir, fname, picklename, create_csv):
    """
    Loads data from INPHAMIS lidar acquisition system into DataFrame object and stores it as a serialized pickle
    object for fast data loading.
    :param fname: filename to be loaded [str]
    :param picklename: filename of pickle [str]
    :param create_csv: whether or not to create a csv file from the data [bool]
    """

    start = time.time()
    df = pd.read_csv(data_dir + fname, delimiter=',')
    print('Elapsed time (read pd): {} sec'.format(time.time()-start))

    if create_csv:
        start = time.time()
        df.to_csv('{}{}'.format(data_dir + fname, '.csv'), index=None)
        print('Elapsed time (create csv): {} sec'.format(time.time()-start))

    outfile = open('{}/{}'.format(data_dir, picklename), 'wb')
    pickle.dump(df, outfile)
    outfile.close()

    return df

if __name__ == "__main__":
    load_INPHAMIS_data(data_dir, fname, picklename, create_csv)

