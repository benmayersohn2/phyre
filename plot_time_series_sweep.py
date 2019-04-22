"""
plot_time_series_sweep.py: Plot an individual time series from a sweep
This requires us to have saved the time_series during the sweep (see run_hpc.py)
"""

import matplotlib.pyplot as plt
import phyre.helpers as helpers
import numpy as np
from phyre.analysis.plot_single import SinglePlotter
import seaborn as sns
from phyre import constants as c

########################################################################

num_clusters = 2
params_name = 'example'

########################################################################

num_years = 3  # plot last three years

data_label = 'time_series'
sweep_label = 'zoo_mort_rate'

# pick second cluster (i.e. index 1)
cluster = 1

kwargs = {'data_label': data_label, 'cluster_kw': {'num_clusters': num_clusters, 'cluster': cluster}}
params = helpers.load(params_name, 'params', 'sweep', **kwargs)

num_years_simulation = params.get('num_years')

num_phy = params['bio']['num_phy']

# get vals
sweep = params['sweep']['pp']

data_kw = {'in_labels': ('zoo_mort_rate',)}
kwargs.update(**{'pd_kw': data_kw})

data_2d = helpers.load(params_name, 'data', 'sweep', **kwargs)
output = np.squeeze(data_2d['output'][0])  # output stored as a list of lists, so extract values

color_kw = {'cmap': 'tab20b'}

kind = 'indiv'
compartments = [{'phy': 'all'}]

legend_kw = {'fontsize': 12, 'loc': 'upper center', 'bbox_to_anchor': (0.5, -0.2), 'fancybox': True,
             'shadow': True, 'ncol': 6}

time = helpers.load(params_name, 'data', 'sweep', data_label='time')

single_params = helpers.load(params_name, 'params', 'sweep', **kwargs)
plotter = SinglePlotter(eco_in=output, params=single_params, t_in=time)

print('zoo mort = {}'.format(single_params['sweep']['pp'][0][-1][0]))

fig, ax = plt.subplots(1, 1)

sns.set_context('paper')
sns.set_style('white')

plotter.time_series_plot(kind=kind, legend_kw=legend_kw, color_kw=color_kw, compartments=compartments,
                         num_years=num_years, nit_combine=True, ax=ax)

num_steps = 2 * num_years
xlist = np.linspace(num_years_simulation - num_years, num_years_simulation, num_steps)
xlim = [xlist[0], xlist[-1]]
ax.set_xlim(xlim)
ax.set_xticks(xlist)
ax.set_xticklabels(((xlist-xlist[0]) * c.NUM_DAYS_PER_YEAR).astype(int))
ax.set_title('Time Series for Zoo Mort Rate = 0.337')
ax.set_xlabel('Time (Days)')
ax.set_ylabel('Concentrations')

plt.show()
