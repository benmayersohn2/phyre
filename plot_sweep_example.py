"""
plot_sweep_example.py: Plot the average of the total phytoplankton - see run_hpc.py

As the zooplankton mortality rate goes up, so does the average phytoplankton biomass.
This is expected due to the top-down control of predators on phytoplankton.
"""

import matplotlib.pyplot as plt
import phyre.helpers as helpers

plt.switch_backend('Qt5Agg')

########################################################################

params_name = 'example'
num_clusters = 2

data_label = 'avg_phy_total'
sweep_label = 'zoo_mort_rate'

kwargs = {'data_label': data_label, 'cluster_kw': {'num_clusters': num_clusters}}
params = helpers.load(params_name, 'params', 'sweep', **kwargs)

data_kw = {'in_labels': [sweep_label], 'out_label': data_label}
kwargs.update(**{'pd_kw': data_kw})

data = helpers.load(params_name, 'data', 'sweep', **kwargs)
data[data_label] = data[data_label].map(lambda x: x[0])

print(data.head())

fig, ax = plt.subplots(1, 1)

data.plot.scatter(sweep_label, data_label, ax=ax, legend=False)
ax.set_title('Average Total P Biomass vs. Z Mortality Rate')
ax.set_xlabel('Z Mortality Rate')
ax.set_ylabel('Avg P')

plt.savefig('plots/sweep_example.pdf', bbox_inches='tight', pad_inches=0)
plt.show()
