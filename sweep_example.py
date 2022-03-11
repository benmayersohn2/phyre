"""
single_multi.py: Example of a single run that comes with PHYRE model.
"""

import numpy as np

from phyre import helpers
import os

########################################################################

# Key switches we tend to change a bunch

params_name = os.path.basename(__file__)[os.path.basename(__file__).find('_')+1:-3]

res_forcing_scale = 18.1  # in nitrogen units
turnover_rate = 0.04

debug = False
include_zoo = True

########################################################################

# TIME & DATA STORAGE

# Time parameters
num_years = 100

# how many of each compartment?
num_phy = 6
num_zoo = 2
num_res = 3

########################################################################

# PHYTOPLANKTON + RESOURCES

# Max growth rate
small_rate = 1.
large_rate = 2.
phy_growth_rate_max = 0.306 * np.ones(num_phy,)
phy_growth_rate_max[:int(num_phy/2)] *= small_rate
phy_growth_rate_max[int(num_phy/2):] *= large_rate

phy_mort_rate = 0.1

#############################################################

# STOICHIOMETRY

large_ent = [(0, 0), (1, 1), (2, 2), (0, 3), (1, 4), (2, 5)]
mid_ent = [(0, 1), (1, 2), (2, 0), (0, 4), (1, 5), (2, 3)]
small_ent = [(0, 2), (1, 0), (2, 1), (0, 5), (1, 2), (2, 4)]

# assign forcing amps from scale, times Redfield-like properties.
makeup_base = np.array([0.15, 1e-2, 1e-5])

res_forcing_amps = res_forcing_scale * makeup_base / makeup_base[0]

dev_factor = 0.1  # 10 percent

largest_scale = 1.3 / (1 + dev_factor)
medium_scale = 1.0
small_scale = 1.0

res_phy_makeup_small = np.array([[1 + dev_factor, 1., 1 - dev_factor],
                                 [1 - dev_factor, 1 + dev_factor, 1.],
                                 [1., 1 - dev_factor, 1 + dev_factor]])

res_phy_makeup_large = np.array(res_phy_makeup_small)

res_phy_makeup_small *= makeup_base[:, None]
res_phy_makeup_large *= makeup_base[:, None]

res_phy_stoich_ratio = np.concatenate((res_phy_makeup_small, res_phy_makeup_large), axis=1)

for ent in large_ent:
    res_phy_stoich_ratio[tuple(ent)] *= largest_scale

for ent in mid_ent:
    res_phy_stoich_ratio[tuple(ent)] *= medium_scale

for ent in small_ent:
    res_phy_stoich_ratio[tuple(ent)] *= small_scale

##########################################

# NUTRIENT SATURATION

# scale:
large_ent = [(0, 2), (1, 0), (2, 1), (0, 5), (1, 3), (2, 4)]
mid_ent = [(0, 0), (1, 1), (2, 2), (0, 3), (1, 4), (2, 5)]
small_ent = [(0, 1), (1, 2), (2, 0), (0, 4), (1, 5), (2, 3)]

dev_factor_high = 0.1
dev_factor_low = 0.2

largest_scale = 1.0
medium_scale = 1.0
small_scale = 0.7 / (1 - dev_factor_low)

phy_coef_scale = 1
phy_small_coef = phy_coef_scale
phy_large_coef = 4 * phy_small_coef

# speed of oscillations governed by strength of deviations
phy_growth_base_small = np.array([[1, 1 - dev_factor_low, 1 + dev_factor_high],
                                  [1 + dev_factor_high, 1., 1 - dev_factor_low],
                                  [1 - dev_factor_low, 1 + dev_factor_high, 1.]])

phy_growth_base_large = np.array(phy_growth_base_small)
phy_growth_sat_large = phy_large_coef * phy_growth_base_large
phy_growth_sat_small = phy_small_coef * phy_growth_base_small

sat_scale_factor = 1
phy_growth_sat_small *= sat_scale_factor * makeup_base[:, None]
phy_growth_sat_large *= sat_scale_factor * makeup_base[:, None]

phy_growth_sat_small = phy_growth_sat_small.tolist()
phy_growth_sat_large = phy_growth_sat_large.tolist()

phy_growth_sat = np.concatenate((phy_growth_sat_small, phy_growth_sat_large), axis=1)

for ent in large_ent:
    phy_growth_sat[tuple(ent)] *= largest_scale

for ent in mid_ent:
    phy_growth_sat[tuple(ent)] *= medium_scale

for ent in small_ent:
    phy_growth_sat[tuple(ent)] *= small_scale

########################################################################

# ZOOPLANKTON

zoo_grazing_rate_max = (1.15 * np.array([1.5, 0.5])).tolist()
zoo_grazing_sat = [10., 10.]
zoo_slop_feed = [1., 1.]

zoo_prey_pref = np.transpose([[1, 1, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 1, 1, 1, 0]]).tolist()

########################################################################

# Specify initial conditions
phy_0 = 1e-3
res_0 = 1e-5
zoo_0 = 1e-2

# Bio parameters go directly into rhs_build function
bio = {'zoo_prey_pref': zoo_prey_pref, 'zoo_slop_feed': zoo_slop_feed,
       'num_zoo': num_zoo, 'num_res': num_res, 'turnover_rate': turnover_rate,
       'res_phy_stoich_ratio': res_phy_stoich_ratio,
       'res_forcing_scale': res_forcing_scale,
       'phy_growth_rate_max': phy_growth_rate_max, 'phy_growth_sat': phy_growth_sat,
       'phy_mort_rate': phy_mort_rate,
       'zoo_grazing_rate_max': zoo_grazing_rate_max, 'zoo_grazing_sat': zoo_grazing_sat,
       'num_phy': num_phy, 'include_zoo': include_zoo
       }

# Combine bio with other stored parameters
params = {'bio': bio, 'phy_0': phy_0, 'res_0': res_0, 'zoo_0': zoo_0, 'num_years': num_years}

########################################################################

num_clusters = 2
sweep_params = [['zoo_mort_rate', None, np.linspace(0.015, 0.5, 4).tolist()]]

prod = 1
for param in sweep_params:
    prod *= len(param[-1])

# num_clusters should divide total number of parameter combinations
assert prod % num_clusters == 0

params.update({'sweep': {'pp': sweep_params}})

# Save
helpers.save(params_name, 'params', 'sweep', output=params, cluster_kw={'num_clusters': num_clusters},
             gen_hpc_scripts=False)
