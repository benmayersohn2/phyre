"""
plot_single_example_multiplot.py: Six-panel plot of output from single_example.py

We have three small and three large phytoplankton.

P1, P2, P3 = phy[0], phy[1], phy[2] = P1^s, P2^s, P3^s ("s" for small phytoplankton)
P4, P5, P6 = phy[3], phy[4], phy[5] = P1^l, P2^l, P3^l ("l" for large phytoplankton)

The two zooplankton are Z1, Z2 = zoo[0], zoo[1] = Z^s, Z^l
"""

import matplotlib.pyplot as plt
import seaborn as sns
from phyre.analysis.plot_single import SinglePlotter
from phyre import helpers
from phyre.analysis import analysis as al
from phyre import constants as c
import numpy as np
from matplotlib.ticker import FormatStrFormatter

plt.switch_backend('Qt5Agg')

########################################################################

params_name = 'example'

#######################################################################

num_years_ts = 3  # Number of years to display in time series

# Set up plotter class
plotter = SinglePlotter(params_name=params_name)

num_years_simulation = plotter.num_years  # 50 years total in simulation

data = helpers.load(params_name, 'data', 'single')
params = helpers.load(params_name, 'params', 'single')

sns.set_context('paper')
sns.set_style('white')

################################################################################################

# TIME SERIES

# PHYTO

phy_indiv_alpha = 0.8
phy_total_alpha = 0.8

phy_total_lw = 1.0
phy_indiv_lw = 1.5

eco = plotter.eco_in

fig, ax = plt.subplots(3, 2, figsize=(10, 9))
fig.subplots_adjust(hspace=0.2)
ax = ax.ravel(order='F')

phy_legend_kw = {'fontsize': 11, 'loc': 'upper center', 'bbox_to_anchor': (0.5, 0), 'ncol': 3}

t_y = np.linspace(0, num_years_ts, int(num_years_ts * c.NUM_DAYS_PER_YEAR)) + num_years_simulation - num_years_ts

# individuals
num_phy = plotter.params.get('bio').get('num_phy')
num_res = 3
num_zoo = 2

phy_indices = [3]
phy_labels = ['P$^l_1$', 'Total P']
phy_colors = ['#9966cc', '#22aa33']

phy = helpers.get_last_n_years(plotter.eco_in[:num_phy, :], num_years_ts)

for i in range(len(phy_labels)-1):
    ax[0].plot(t_y, phy[phy_indices[i], :], label=phy_labels[i], color=phy_colors[i], linewidth=phy_indiv_lw,
               alpha=phy_indiv_alpha)

ax[0].plot(t_y, np.sum(phy, axis=0).tolist(), color=phy_colors[-1],
           label=phy_labels[-1], linewidth=phy_total_lw, alpha=phy_total_alpha, zorder=100)
ax[0].legend(**phy_legend_kw)
ax[0].tick_params(labelsize=9)
ax[0].set_xticks([])

# Reduce size of subplots
for i in range(6):
    box1 = ax[i].get_position()
    ax[i].set_position([box1.x0, box1.y0 + box1.height * 0.2, box1.width, box1.height * 0.9])

#########################

# NUTRIENTS

res_colors = ['#029386', '#a9561e', '#c875c4', '#c79fef']
res_legend_kw = {'fontsize': 11, 'loc': 'upper center', 'bbox_to_anchor': (0.5, -0.05), 'ncol': 4}

res_labels = ['N', 'PO$_4$', 'Fe', 'R$^{*}$']
res_indices = [0]  # just plot nitrogen
res = al.res_to_carbon_conversions(plotter.eco_in, plotter.params, num_years=num_years_ts)[res_indices, :]

for i in range(len(res_indices)):
    ax[1].plot(t_y, res[i, :], label=res_labels[i], color=res_colors[i], linewidth=1)

ax[1].legend(**res_legend_kw)
ax[1].set_xticks([])

#########################

# ZOOPLANKTON

zoo_indices = [1]  # Just plot Z^l (Z2)

zoo_indiv_alpha = 0.6
zoo_total_alpha = 0.8

zoo_indiv_lw = 1.5
zoo_total_lw = 1.0

zoo = helpers.get_last_n_years(plotter.eco_in[helpers.eco_indices('zoo', params=params), :], num_years_ts)

zoo_legend_kw = {'fontsize': 11, 'loc': 'upper center', 'bbox_to_anchor': (0.5, -0.3), 'ncol': 3}

zoo_colors = ['#0033cb', '#008800']

zoo_labels = ['Z$^l$', 'Total Z']

for i in range(len(zoo_indices)):
    ax[2].plot(t_y, zoo[zoo_indices[i], :], label=zoo_labels[i], color=zoo_colors[i], linewidth=zoo_indiv_lw,
               alpha=zoo_indiv_alpha)

# plot total
ax[2].plot(t_y, np.sum(zoo, axis=0), color=zoo_colors[-1], label='Total Z', linewidth=zoo_total_lw,
           alpha=zoo_total_alpha)
ax[2].legend(**zoo_legend_kw)

num_steps = 2 * num_years_ts
xlist = np.arange(num_years_simulation - num_years_ts, num_years_simulation, 0.5)  # last 3 years
xlim = [xlist[0], xlist[-1]]

for i in range(3):
    ax[i].set_xlim(xlim)

ax[2].set_xticks(xlist)
ax[2].set_xticklabels(((xlist-xlist[0]) * c.NUM_DAYS_PER_YEAR).astype(int))

#########################

num_years_spectrum = 30

spectrum_kw = {'detrend': 'constant', 'sort': 'freqs', 'normalize': True, 'window': 'hamming', 'thres': 1e-12}
alpha = 0.6
phy_freqs, phy_amps = al.filtered_spectrum(eco, params, num_years=num_years_spectrum,
                                           compartments=[{'phy': phy_indices}], **spectrum_kw)
phy_tot_freqs, phy_tot_amps = al.filtered_spectrum(eco, params, num_years=num_years_spectrum, kind='total',
                                                   compartments=[{'phy': 'all'}], **spectrum_kw)

res_freqs = list()
res_amps = list()
res_longer = al.res_to_carbon_conversions(plotter.eco_in, plotter.params, num_years=num_years_spectrum)[res_indices, :]
for i in range(len(res_indices)):
    freqs, amps = al.filtered_spectrum_simple(res_longer[i, :], **spectrum_kw)
    res_freqs.append(freqs)
    res_amps.append(amps)

res_amps = np.array(res_amps)
res_freqs = np.array(res_freqs)

zoo_freqs, zoo_amps = al.filtered_spectrum(eco, params, num_years=num_years_spectrum,
                                           compartments=[{'zoo': 'all'}], **spectrum_kw)
zoo_tot_freqs, zoo_tot_amps = al.filtered_spectrum(eco, params, num_years=num_years_spectrum, kind='total',
                                           compartments=[{'zoo': 'all'}], **spectrum_kw)


# power spectrum
ax[3].plot(phy_tot_freqs[0, :], phy_tot_amps[0, :], color=phy_colors[-1], label=phy_labels[-1], linewidth=phy_total_lw,
           alpha=phy_total_alpha, zorder=100)

ax[3].plot(phy_freqs[0, :], phy_amps[0, :], color=phy_colors[0], label=phy_labels[0], linewidth=phy_indiv_lw,
           alpha=phy_indiv_alpha)
ax[4].plot(res_freqs[0, :], res_amps[0, :], color=res_colors[0], label=res_labels[0], linewidth=1)

for i in range(len(zoo_indices)):
    ax[5].plot(zoo_freqs[zoo_indices[i], :], zoo_amps[zoo_indices[i], :], color=zoo_colors[i], label=zoo_labels[i],
               linewidth=zoo_indiv_lw, alpha=zoo_indiv_alpha)

ax[5].plot(zoo_tot_freqs[0, :], zoo_tot_amps[0, :], color=zoo_colors[-1], label=zoo_labels[-1], linewidth=zoo_total_lw,
           alpha=zoo_total_alpha)

[ax[i].set_xscale('log') for i in range(3, 6)]

ax[3].set_xticks([])
ax[4].set_xticks([])

vline_color = 'black'

for i in range(6):
    ax[i].tick_params(labelsize=9)
    ax[i].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

colors = list()
ann_legend_kw = {'fontsize': 10, 'loc': 'upper center', 'bbox_to_anchor': (0.5, 0), 'ncol': 7}
for i in range(3, 6):
    ax[i].set_ylim(-0.1, 1.1)

ax[0].set_title('Time Series ($\mu$M C)', fontsize=11)
ax[3].set_title('Fourier Spectra (Normalized)', fontsize=11)

ax[2].set_xlabel('Time (days)')
ax[5].set_ylabel('Amplitude')
ax[5].set_xlabel('Period (days)')

# change xticks
# 2 days is the Nyquist frequency, because spacing is 1 day
xlim = np.array([1./1000, 1./2])
xlist = 1./np.array([600, 200, 80, 30, 10, 4, 2])

for i in range(3, 6):
    ax[i].set_xlim(xlim)

ax[5].set_xticks(xlist)
ax[5].set_xticklabels((1./xlist).astype(int))

for i, label in enumerate(('A', 'B', 'C', 'D', 'E', 'F')):
    ax[i].text(0.06, 1.14, label, transform=ax[i].transAxes,
               fontsize=16, fontweight='bold', va='top', ha='right')

plt.savefig('plots/single_example_multiplot.pdf', bbox_inches='tight')
plt.show()
