"""
plot_single_example_noise.py: Time series plot of all phytoplankton from single_example_noise.py

We have three small and three large phytoplankton.

P1, P2, P3 = phy[0], phy[1], phy[2] = P1^s, P2^s, P3^s ("s" for small phytoplankton)
P4, P5, P6 = phy[3], phy[4], phy[5] = P1^l, P2^l, P3^l ("l" for large phytoplankton)
"""

import matplotlib.pyplot as plt
import seaborn as sns
from phyre.analysis.plot_single import SinglePlotter
from phyre import helpers
import numpy as np
from phyre import constants as c

########################################################################

params_name = 'example_noise'

########################################################################

plotting_threshold = 0
num_years = 5

# SINGLE PLOT

plotter = SinglePlotter(params_name=params_name)
num_years_simulation = plotter.num_years

data = helpers.load(params_name, 'data', 'single', data_label='time_series')
params = helpers.load(params_name, 'params', 'single')

sns.set_context('paper')
sns.set_style('white')

legend_kw = {'fontsize': 12, 'loc': 'upper center', 'bbox_to_anchor': (0.5, -0.2), 'ncol': 3}
color_kw = {'cmap': 'tab20b'}

################################################################################################

# TIME SERIES

_, ax = plt.subplots(1, 1)
plotter.time_series_plot(kind='indiv', legend_kw=legend_kw, ax=ax, color_kw=color_kw,
                         plotting_threshold=plotting_threshold, num_years=num_years)

num_steps = 2 * num_years
xlist = np.linspace(num_years_simulation - num_years, num_years_simulation, num_steps)
xlim = [xlist[0], xlist[-1]]
ax.set_xlim(xlim)
ax.set_xticks(xlist)
ax.set_xticklabels(((xlist-xlist[0]) * c.NUM_DAYS_PER_YEAR).astype(int))

ax.set_xlabel('Time (Days)', fontsize=12)
ax.set_ylabel('Concentrations', fontsize=12)

plt.savefig('plots/single_example_noise.pdf', bbox_inches='tight', padding=0)
plt.show()
