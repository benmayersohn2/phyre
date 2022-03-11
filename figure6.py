import matplotlib
matplotlib.use('Qt5Agg')
matplotlib.rc('text', usetex = True)
matplotlib.rc('font', family='sans-serif')
matplotlib.rc('text.latex', preamble=r'\usepackage{amsmath}')

import matplotlib.pyplot as plt
import seaborn as sns
from phyre import helpers
from phyre.analysis import analysis as al
from phyre import constants as c
import numpy as np
import os
import pickle
import string
from matplotlib.ticker import FormatStrFormatter
from numpy.random import default_rng

#######################################################################

plotting_threshold = 1e-3
data_ext = 'npy'

load_spectrum = True

num_years = 1
num_years_spectrum = 150

# TIME SERIES

# We'll structure this so each row contains a sweep:
# Column 1: Individual P
# Column 2: Total P and Z (if applicable)
# Column 3 (if applicable): Individual Z
# This makes for twelve panels total for the constant tau sweep

sns.set_context('paper')
sns.set_style('white')

phy_indiv_alpha = 1
phy_total_alpha = 1

total_lw = 0.8
indiv_lw = 1.0

phy_legend_kw = {'fontsize': 9, 'loc': 'upper center', 'bbox_to_anchor': (0.5, -0.5), 'ncol': 3}
res_legend_kw = {'fontsize': 9, 'loc': 'upper center', 'bbox_to_anchor': (0.5, -0.2), 'ncol': 3}
zoo_legend_kw = {'fontsize': 9, 'loc': 'upper center', 'bbox_to_anchor': (0.5, -0.2), 'ncol': 3}

fig, ax = plt.subplots(1, 2, figsize=(8, 4))
fig.subplots_adjust(hspace=0.2, top=0.8, wspace=0.2, bottom=0.2)

# individuals
params_name = 'strongpred_light_2'
compartment_list = ['all', [0]]
suffixes = ['', '_p1s']
colors = ['#1f77b4', 'k']
labels = [r'P$_{tot}$', r'$P^s_{1}$']
for q, compartment in enumerate(compartment_list):
    eco = helpers.load(params_name, 'data', 'single', data_ext=data_ext)
    params = helpers.load(params_name, 'params', 'single')
    total_p = helpers.restrict_ts(eco, params, kind='total', num_years=num_years_spectrum, compartments=[{'phy': compartment}])

    # compute spectrum
    pkl_filename = f'{params_name}{suffixes[q]}.pkl'
    if load_spectrum and os.path.exists(pkl_filename):
        with open(pkl_filename, 'rb') as f:
            print(f'Reading output from {pkl_filename}')
            result = pickle.load(f)
    else:
        result = al.wavelet_spectrum_simple(total_p, sort='freq', normalize=False, num_suboctaves=12, return_raw=True)
        with open(pkl_filename, 'wb') as f:
            print(f'Writing output to {pkl_filename}')
            pickle.dump(result, f)

    periods = result['period']
    scale = result['scale']
    amps = np.sqrt(result['global_ws'])
    amps = amps / amps.max()

    xlim = [10, 54000]
    xticks = np.array([10 ** i for i in range(1, 5)])
    xticklabels = [f'$10^{i}$' for i in range(1, 5)]
    ax[1].plot(periods, amps, color=colors[q], label=labels[q])
    ax[1].set_xscale('log')
    ax[1].set_xlim(xlim)
    ax[1].set_xticks(xticks)
    ax[1].set_xticklabels(xticklabels)
    ax[1].set_title('(Normalized) Wavelet Spectrum')
    ax[1].grid(True)

    # ax[1].axvline(x=int(360*3), color='k', linewidth=1, linestyle='dashed', alpha=0.5)
    num_years_run = params['num_years']

    # Time in years and days
    t_y = np.linspace(0, num_years, int(num_years * c.NUM_DAYS_PER_YEAR)) + num_years_run - num_years
    t = (t_y - t_y[0]) * c.NUM_DAYS_PER_YEAR
    last_year = np.array(list(range(int(c.NUM_DAYS_PER_YEAR))))

    if q == 0:
        bio = params['bio']
        turnover_min = bio['turnover_min']
        turnover_max = bio['turnover_max']
        light_min = bio['light_min']
        light_max = bio['light_max']
        mixed_layer_ramp_lengths = bio['mixed_layer_ramp_lengths']
        mixed_layer_ramp_times = bio['mixed_layer_ramp_times']
        phase_shift = bio.get('turnover_phase_shift', 0)

        light = helpers.light_profile(low_val=light_min, high_val=light_max,
                                      mixed_layer_ramp_lengths=mixed_layer_ramp_lengths,
                                      mixed_layer_ramp_times=mixed_layer_ramp_times, t=last_year)

        turnover = helpers.ml_profile(low_val=turnover_min, high_val=turnover_max, phase_shift=phase_shift,
                                      mixed_layer_ramp_lengths=mixed_layer_ramp_lengths,
                                      mixed_layer_ramp_times=mixed_layer_ramp_times, t=last_year)

        # Add some noise
        def s(t, T1, T2, seed=0):
            t_final = t[-1]
            t0 = t[0]
            # Assumes T2 > T1
            noise_freq = [1/T2, 1/T1]
            noise_filter = 'bandpass'
            noise_kwargs = {'t_final': t_final, 'cutoff_freq': noise_freq, 't0': t0,
                            'filter_type': noise_filter, 'uniform': False, 'seed': seed}
            return helpers.generate_noise(**noise_kwargs)

        diff = turnover_max - turnover_min
        series_max = max(turnover)
        series_min = min(turnover)
        turnover = diff * (turnover - series_min) / (series_max - series_min) + turnover_min

        ax_light = ax[0]
        ax_turnover = ax_light.twinx()
        ax_light.plot(last_year, light, color='r')
        ax_turnover.plot(last_year, turnover, color='#1f77b4')

        ax_light.set_xlabel('Month')
        ax_light.set_xticks([30 * i for i in range(12)])
        ax_light.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
        ax_light.grid(True)
        ax_light.set_xlim([0, 359])
        ax_light.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax_turnover.set_yticks([0, 0.01, 0.02, 0.03, 0.04])
        ax_light.set_yticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1'])
        ax_turnover.set_yticklabels(['0', '0.01', '0.02', '0.03', '0.04'])

        ax_light.set_ylabel('$I(t)$', fontsize=14, rotation=0, color='r', labelpad=20)
        ax_turnover.set_ylabel(r'$\tau(t)$', fontsize=14, rotation=0, color='#1f77b4', labelpad=20)

        ax_light.tick_params(axis='y', colors='r')
        ax_turnover.tick_params(axis='y', colors='#1f77b4')

ax[1].legend()

fig.tight_layout(pad=3.0)
for i, label in enumerate(('A', 'B')):
    ax[i].text(0.06, 1.10, rf'\textbf{{{label}}}', transform=ax[i].transAxes,
               fontsize=16, fontweight='bold', va='top', ha='right')

# fig.suptitle('Ramp-like Extrinsic Variability in Nutrient and Light', fontsize=11)
# plt.savefig('paper/tau_light_ramp_spectrum_tot_p1s.pdf', bbox_inches='tight')
plt.show()
