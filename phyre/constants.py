"""
constants.py: define constants to be used in model, independent of the filesystem
"""

import os
import numpy as np

DATA_EXT_DEFAULT = 'npy'

# machine eps threshold
EXTINCT_THRESH = 1e-10

# project root directory (i.e. src)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

KEYS_TO_POP = ['res_forcing_scale', 'phy_growth_rate_max_TEMP', 'zoo_grazing_rate_max_TEMP']

# Time constants, in days
NUM_DAYS_PER_YEAR = 360.0
ONE_MONTH = 30.0

ML_RAMP_TIMES_DEFAULT = np.array([2 * ONE_MONTH, 10 * ONE_MONTH])
ML_RAMP_LENGTHS_DEFAULT = np.array([30.0, 120.0])

PHOTIC_DEPTH_DEFAULT = 30
PHOTIC_PCT = 0.01

ML_RAMP_EXTREMA_DEFAULT = np.array([PHOTIC_DEPTH_DEFAULT, PHOTIC_DEPTH_DEFAULT])

NAN_VALUE = -99999.0  # use as NaN

COL_DELIMITER = ";"
ROW_DELIMITER = "\n"

PARAMS = 'params'
DEBUG = 'debug'
DATA = 'data'
HPC = 'hpc'

PARAMS_ROOT = os.path.join(PROJECT_ROOT, 'params')

SINGLE_PARAMS_DIR = os.path.join(PROJECT_ROOT, 'params/single/{}')
SINGLE_PARAMS_FILE = os.path.join(SINGLE_PARAMS_DIR, '{}.json')

SINGLE_DEBUG_DIR = os.path.join(PROJECT_ROOT, 'debug/single/{}')
SINGLE_DEBUG_FILE = os.path.join(SINGLE_DEBUG_DIR, '{}.json')

SWEEP_PARAMS_DIR = os.path.join(PROJECT_ROOT, 'params/sweep/{}/')
SWEEP_PARAMS_FILE = os.path.join(SWEEP_PARAMS_DIR, '{}.json')
