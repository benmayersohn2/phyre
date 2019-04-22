"""
single_example_noise.py: Same as single_example.py, but with Gaussian noise

Uses the function generate_noise from helpers.py to add noise to the dilution coefficient "turnover_rate"
"""

import numpy as np

from phyre import constants as c
from phyre import helpers
from phyre.model import single
import os

########################################################################

# Key switches we tend to change a bunch

params_name = os.path.basename(__file__)[os.path.basename(__file__).find('_')+1:-3]

turnover_rate = 0.04
res_export_frac = 1.
res_forcing_scale = 18.1  # in nitrogen units

include_zoo = True

# add noise with a bandpass filter to the turnover coefficient
noise_amp = 0.1  # the noise is small and not so noticeable
noise_freq = [1.0/360, 2.0/360]
noise_filter = 'bandpass'

########################################################################

# TIME & DATA STORAGE

# Time parameters
num_years = 50

########################################################################

# PHYTOPLANKTON + RESOURCES

# how many phytoplankton?
num_phy = 6

# Max growth rate
phy_growth_rate_max = [0.306, 0.306, 0.306, 0.612, 0.612, 0.612]

phy_small_indices = [0, 1, 2]
phy_large_indices = [3, 4, 5]

#############################################################

# STOICHIOMETRY
# N, Si, P, Fe

# Redfield ratio
makeup_base_small = np.array([0.15, 0.0, 1e-2, 1e-5])  # no silicate in small phytoplankton
makeup_base_large = np.array([0.15, 0.14, 1e-2, 1e-5])

# assign forcing amps such that nitrogen source has value of "res_forcing_scale"
# assign other sources in stoichiometric proportion
res_forcing_amps = res_forcing_scale * makeup_base_large[[0, 0, 1, 2, 3]] / makeup_base_large[0]
res_forcing_amps[1] = 0

# N, Si, P, Fe
# These are listed in terms of deviations from Redfield (a value of 1 is exactly Redfield value)
res_phy_makeup_small = np.array([[1.3, 1., 0.9],
                                 [np.nan, np.nan, np.nan],  # no silicate
                                 [0.9, 1.3, 1.],
                                 [1., 0.9, 1.3]])

res_phy_makeup_large = np.array(res_phy_makeup_small)

# scale by Redfield ratio
res_phy_makeup_small *= makeup_base_small[:, None]
res_phy_makeup_large *= makeup_base_large[:, None]

res_phy_makeup_ratio = np.concatenate((res_phy_makeup_small, res_phy_makeup_large), axis=1)

res_phy_stoich_ratio = np.concatenate(([np.array(res_phy_makeup_ratio)[c.NIT_INDEX_SHORT, :]],
                                       res_phy_makeup_ratio), axis=0)

##########################################

# NUTRIENT SATURATION

phy_coef_scale = 1
phy_small_coef = phy_coef_scale
phy_large_coef = 4 * phy_small_coef

# speed of oscillations governed by strength of deviations
phy_growth_base_small = np.array([[1, 0.7, 1.1],
                                  [np.nan, np.nan, np.nan],  # no NH4
                                  [np.nan, np.nan, np.nan],  # no silicate
                                  [1.1, 1., 0.7],
                                  [0.7, 1.1, 1.]])
phy_growth_base_small[2, :] = np.nan

phy_growth_base_large = np.array(phy_growth_base_small)
phy_growth_sat_large = phy_large_coef * phy_growth_base_large
phy_growth_sat_small = phy_small_coef * phy_growth_base_small

sat_scale_factor = 1
phy_growth_sat_small[[0, 2, 3, 4], :] *= sat_scale_factor * makeup_base_small[:, None]
phy_growth_sat_large[[0, 2, 3, 4], :] *= sat_scale_factor * makeup_base_large[:, None]

phy_growth_sat_small = phy_growth_sat_small.tolist()
phy_growth_sat_large = phy_growth_sat_large.tolist()

phy_growth_sat = np.concatenate((phy_growth_sat_small, phy_growth_sat_large), axis=1)

########################################################################

small_mort = 0.1
large_mort = 0.1

phy_mort_rate = [small_mort, small_mort, small_mort, large_mort, large_mort, large_mort]  # phy mortality

########################################################################

# ZOOPLANKTON

zoo_grazing_rate_max = (1.15 * np.array([1.5, 0.5])).tolist()
zoo_mort_rate = 0.015
zoo_grazing_sat = [10., 10.]
zoo_slop_feed = [1., 1.]

zoo_low_pref = 0.
zoo_high_pref = 1.
zoo_zoo_pref = 1.  # this value seems to matter a lot when grazing is strong (?)

########################################################################

# Specify initial conditions
phy_0 = 1e-3
res_0 = 1e-5
zoo_0 = 1e-2

# Bio parameters go directly into rhs_build function
bio = {'zoo_low_pref': zoo_low_pref, 'zoo_high_pref': zoo_high_pref, 'phy_small_indices': phy_small_indices,
       'zoo_slop_feed': zoo_slop_feed, 'noise_amp': noise_amp,
       'noise_freq': noise_freq, 'noise_filter': noise_filter,
       'res_phy_makeup_ratio': res_phy_makeup_ratio,
       'phy_large_indices': phy_large_indices, 'res_export_frac': res_export_frac,
       'zoo_zoo_pref': zoo_zoo_pref, 'res_forcing_scale': res_forcing_scale,
       'phy_growth_rate_max': phy_growth_rate_max, 'phy_growth_sat': phy_growth_sat,
       'phy_mort_rate': phy_mort_rate, 'zoo_mort_rate': zoo_mort_rate,
       'zoo_grazing_rate_max': zoo_grazing_rate_max, 'zoo_grazing_sat': zoo_grazing_sat,
       'num_phy': num_phy, 'turnover_rate': turnover_rate, 'include_zoo': include_zoo
       }

# Combine bio with other stored parameters
params = {'bio': bio, 'phy_0': phy_0, 'res_0': res_0, 'zoo_0': zoo_0, 'num_years': num_years}

########################################################################

# Save
helpers.save(params_name, 'params', 'single', output=params)

# Run
single.run(params_name)
