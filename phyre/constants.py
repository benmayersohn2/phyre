"""
constants.py: define constants to be used in model, independent of the filesystem
"""

import os
import numpy as np

DATA_EXT_DEFAULT = 'npy'

DEEP_NUTRIENTS = [0.15, 0, 0.14, 1e-2, 1e-5]  # Redfield ratio

# machine eps threshold
EXTINCT_THRESH = 1e-10

# project root directory (i.e. src)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

KEYS_TO_POP = ['noise_amp', 'noise_freq', 'noise_filter', 'zoo_high_pref',
               'zoo_low_pref', 'zoo_zoo_pref', 'res_phy_makeup_ratio', 'res_forcing_scale',
               'phy_large_indices', 'phy_growth_rate_max_TEMP', 'zoo_grazing_rate_max_TEMP']

# Time constants, in days
NUM_DAYS_PER_YEAR = 360.0
ONE_MONTH = 30.0

NO3_DECAY_DEFAULT = 4.6
RES_EXPORT_DEFAULT = 0.2

NAN_VALUE = -99999.0  # use as NaN

ML_RAMP_TIMES_DEFAULT = np.array([0 * ONE_MONTH + 1, 10 * ONE_MONTH])
ML_RAMP_LENGTHS_DEFAULT = np.array([1.0, 60.0])

PHOTIC_DEPTH_DEFAULT = 30
PHOTIC_PCT = 0.01

ML_RAMP_EXTREMA_DEFAULT = np.array([PHOTIC_DEPTH_DEFAULT, PHOTIC_DEPTH_DEFAULT])

NUM_RES = 5
NUM_ZOO = 2

# For res_phy(zoo)_stoich_ratio. Nitrate and ammonium in two separate compartments
NO3_INDEX = 0
NH4_INDEX = 1
SIL_INDEX = 2
PO4_INDEX = 3
IRN_INDEX = 4

# For res_phy(zoo)_makeup_ratio. Nitrate and ammonium in a single nitrogen compartment
NIT_INDEX_SHORT = 0
SIL_INDEX_SHORT = 1
PO4_INDEX_SHORT = 2
IRN_INDEX_SHORT = 3

NIT_COUNT = 2  # two nitrogen compartments (NO3 and NH4)
NIT_INDICES = [NO3_INDEX, NH4_INDEX]

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

# For dominant frequencies
DEAD_VALUE = 999999
