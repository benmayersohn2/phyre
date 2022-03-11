import matplotlib
matplotlib.use('Qt5Agg')
matplotlib.rc('text', usetex = True)
matplotlib.rc('font', family='sans-serif')
matplotlib.rc('text.latex', preamble=r'\usepackage{amsmath}')
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import numpy.ma as ma
import phyre.helpers as helpers
from phyre import constants as c
import pandas as pd
from matplotlib.ticker import FormatStrFormatter
from phyre.analysis import analysis as al
import pickle
import string
import matplotlib.patheffects as patheffects

########################################################################

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

err_on_fail = False

sns.set_context('paper')
sns.set_style('white')

vmin = 0
vmax = 1

level_spacing = 0.1
levels = np.arange(vmin, vmax+level_spacing, level_spacing)

fig, ax_spectra = plt.subplots(2, 1, figsize=(10, 8))

load_spectrum = True

# round to nearest whole number
class nf(float):
    def __repr__(self):
        return f'{self:.0f}'

params_names = ['tau_pt4_strongpred_no_osc', 'tau_pt0_pt4_strongpred_18', 'tau_pt0_pt4_strongpred_120', 'tau_pt0_pt4_strongpred_1800']
labels = [r'Constant Forcing', r'T$_{\text{for}}=18\text{ days}$', r'T$_{\text{for}}=120\text{ days}$', r'T$_{\text{for}}=1800\text{ days}$']
legend_kw = {'fontsize': 10, 'loc': 'upper right', 'ncol': 1}

# Colorblind-friendly
# colors = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c']
colors = ['#000000', '#e69f00', '#cc79a7', '#1f78b4']
styles = ['-', '-', '-', '-']

num_years = 10
t = np.array(list(range(int(num_years * c.NUM_DAYS_PER_YEAR))))

s0 = 2
start_period = 2
num_suboctaves = 32
dj = 1.0 / num_suboctaves
t_long = np.array(list(range(int(150 * c.NUM_DAYS_PER_YEAR))))
j1 = int(np.floor(np.log2(len(t_long) * 1 / start_period) * num_suboctaves))
scale = s0 * 2**(dj * np.arange(j1 + 1))

compartments = ['all', [0]]

pkl_filenames = ['amps_{}.pkl', 'amps_p1s_{}.pkl']
titles = ['Global Wavelet Spectra: Total Phytoplankton', 'Global Wavelet Spectra: Single Species (P$^s_1$)']
for j, ax_spectrum in enumerate(ax_spectra):
    ax_spectrum.set_title(titles[j], fontsize=12)
    ax_spectrum.tick_params(reset=True, labelsize=12, direction='in', which='both')
    for k, params_name in enumerate(params_names):
        the_plot = None
        pkl_filename = pkl_filenames[j].format(params_name)
        eco = helpers.load(params_name, 'data', 'single', data_label='time_series')
        params = helpers.load(params_name, 'params', 'single')

        phy = helpers.restrict_ts(eco, params, compartments=[{'phy': compartments[j]}], num_years=150, kind='total')

        # Get spectrum
        if load_spectrum:
            try:
                with open(pkl_filename, 'rb') as f:
                    print(f'Reading output from {pkl_filename}')
                    result = pickle.load(f)
            except:
                result = al.wavelet_spectrum_simple(phy, sort='freq', rectify_bias=False, num_suboctaves=32, return_raw=True)
                with open(pkl_filename, 'wb') as f:
                    print(f'Writing output to {pkl_filename}')
                    pickle.dump(result, f)
            with open(pkl_filename, 'rb') as f:
                print(f'Reading output from {pkl_filename}')
                result = pickle.load(f)
        else:
            result = al.wavelet_spectrum_simple(phy, sort='freq', rectify_bias=False, num_suboctaves=32, return_raw=True)
            with open(pkl_filename, 'wb') as f:
                print(f'Writing output to {pkl_filename}')
                pickle.dump(result, f)
        periods = result['period']
        amps = result['global_ws']

        # Correct bias
        # amps /= scale

        amps = np.sqrt(amps)

        # Normalize by max for period < 10000
        lp = np.where(periods < 10000)[0]
        hp = np.where(periods >= 10000)[0]
        if periods[np.where(amps == amps.max())[0][0]] > 10000:
            amps[lp] /= amps[lp].max()
            amps[hp] /= amps[hp].max()
        else:
            amps /= amps.max()
        ax_spectrum.plot(periods, amps, color=colors[k], linestyle=styles[k], label=labels[k])

    xticks = (np.array([10 ** i for i in range(1, 5)]))
    ax_spectrum.set_xscale('log')
    ax_spectrum.set_xticks(xticks)
    if j == 1:
        ax_spectrum.set_xticklabels([f'10$^{i}$' for i in range(1, 5)])
        ax_spectrum.set_xlabel('T$_W$ (days)', fontsize=12, labelpad=0)
    else:
        ax_spectrum.legend(**legend_kw)
        ax_spectrum.set_xticklabels([])
        ax_spectrum.set_ylabel('(Normalized) Wavelet Amplitude', fontsize=12)

    ax_spectrum.set_ylim([0, 1.2])
    ax_spectrum.axvline(x=18, ymin=0.9, linestyle=styles[1], linewidth=1, color=colors[1])
    ax_spectrum.axvline(x=120, ymin=0.9, linestyle=styles[2], linewidth=1, color=colors[2])
    ax_spectrum.axvline(x=1800, ymin=0.9, linestyle=styles[3], linewidth=1, color=colors[3])

    t1 = ax_spectrum.text(20, 1.09, '18 days', color=colors[1])
    t2 = ax_spectrum.text(135, 1.09, '120 days', color=colors[2])
    t3 = ax_spectrum.text(2000, 1.09, '1800 days', color=colors[3])

    t1.set_path_effects([patheffects.Stroke(linewidth=0.1, foreground='black'), patheffects.Normal()])
    t2.set_path_effects([patheffects.Stroke(linewidth=0.1, foreground='black'), patheffects.Normal()])
    t3.set_path_effects([patheffects.Stroke(linewidth=0.1, foreground='black'), patheffects.Normal()])

    # Add vertical dashed line where boundary effects start to kick in
    # ax_spectrum.axvline(10000, color='gray', linewidth=1.5, linestyle='--')

    # Patches
    ymax = 1.2
    ymin = 0.001
    xlim = ax_spectrum.get_xlim()
    ax_spectrum.add_patch(Rectangle((xlim[0], ymin), 90 - xlim[0], ymax - ymin, alpha=0.1))
    ax_spectrum.add_patch(Rectangle((90, ymin), 360 - 90, ymax - ymin, color='green', alpha=0.1))
    ax_spectrum.add_patch(Rectangle((360, ymin), xlim[-1] - 360, ymax - ymin, color='yellow', alpha=0.1))

alphabet = list(string.ascii_uppercase)
alphabet = [alphabet[x] for x in (0, 1)]
for i, label in enumerate(alphabet[:2]):
    y = 1.07
    ax_spectra[i].text(0.05, y, rf'\textbf{{{label}}}', transform=ax_spectra[i].transAxes,
               fontsize=16, fontweight='bold', va='top', ha='right')
    ax_spectra[i].grid(True, which='major')

# plt.savefig('paper/vary_taumax_0.4_3_biascorrect.pdf', bbox_inches='tight', pad_inches=0)
# plt.savefig('paper/vary_taumax_0.4_3.pdf', bbox_inches='tight', pad_inches=0)
plt.show()
