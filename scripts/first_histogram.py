import numpy as np
import os

cwd = os.getcwd()
data_dir = cwd + r'/../Data'
fname1 = data_dir + r'/Dev_0_-_2022-02-18_10.48.52.ARSENL'

data = np.loadtxt(fname1, delimiter=',', skiprows=1)

# TODO: np.loadtxt very slow. use pandas