import matplotlib
matplotlib.use('Qt5Agg')
matplotlib.rc('text', usetex = True)
matplotlib.rc('font', family='sans-serif')
matplotlib.rc('text.latex', preamble=r'\usepackage{amsmath}')
matplotlib.rc('hatch', linewidth=0.25)
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
import seaborn as sns
import numpy.ma as ma
import phyre.helpers as helpers
import matplotlib.colors as colors
from phyre import constants as c
import pandas as pd
from matplotlib.ticker import FormatStrFormatter
from phyre.analysis import analysis as al
import pickle
import string

########################################################################

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

num_clusters = 40

err_on_fail = False

data_ext = 'npy'

xy = [('turnover_max', r'$\tau_{max}$ (day$^{-1}$)')]
sweep_param = 'turnover_max'

data_labels = ['wavelet_phy_total', 'wavelet_phy_p1s']

kwargs = {'cluster_kw': {'num_clusters': num_clusters}, 'data_ext': data_ext}
data_kw = {'in_labels': tuple([v[0] for v in xy])}
kwargs.update(**{'pd_kw': data_kw})

sns.set_context('paper')
sns.set_style('white')

vmin = 0
vmax = 1

level_spacing = 0.1
levels = np.arange(vmin, vmax+level_spacing, level_spacing)

fig, ax_arr = plt.subplots(3, 3, figsize=(15, 12), gridspec_kw={'height_ratios': [1, 2, 2]})

load_spectrum = True

# round to nearest whole number
class nf(float):
    def __repr__(self):
        return f'{self:.0f}'

params_names = ['strongpred_vary_taumax_stoch_annual', 'strongpred_vary_taumax_stoch_annual_sub_correction', 'strongpred_vary_taumax_stoch_multiannual_only']
labels = ['Annual', 'Annual + Subseasonal', 'Multiannual']
legend_kw = {'fontsize': 10, 'loc': 'lower center', 'bbox_to_anchor': (1.7, -1.6), 'ncol': 5}

from phyre.helpers import generate_noise

# annual cycle with additional freqs
amps_1 = 1/32 * np.array([4, 4, 14, 10, 0])

# Annual cycle with stronger high frequency variability
# amps_2 = 1/32 * np.array([2, 10, 16, 2, 2])
amps_2 = 1/32 * np.array([2, 10, 16, 4, 0])  # correction

# Stronger Low frequency variability
# amps_3 = 1/32 * np.array([0, 1, 3, 4, 24])

# For m-only
amps_3 = 1/32 * np.array([0, 0, 0, 8, 24])

assert np.isclose(sum(amps_1), 1)
assert np.isclose(sum(amps_2), 1)
assert np.isclose(sum(amps_3), 1)

forcing_periods = np.array([30, 180, 360, 720, 1800])

num_years = 10
t_long = np.array(list(range(int(150 * c.NUM_DAYS_PER_YEAR))))
t = np.array(list(range(int(num_years * c.NUM_DAYS_PER_YEAR))))
# Generate bandpass filtered noise between periods T1 and T2
def s(t, T1, T2, seed=0):
    t_final = t[-1]
    t0 = t[0]
    # Assumes T2 > T1
    noise_freq = [1/T2, 1/T1]
    noise_filter = 'bandpass'
    noise_kwargs = {'t_final': t_final, 'cutoff_freq': noise_freq, 't0': t0, 'additive': False,
                    'filter_type': noise_filter, 'uniform': False, 'seed': seed}
    return generate_noise(**noise_kwargs)

ax_arr = ax_arr.ravel(order='F')
amps_all = (amps_1, amps_2, amps_3)

def get_first(x):
    try:
        return x[0]
    except:
        return c.NAN_VALUE

xlim = [10, 54000]
xticks = np.array([10 ** i for i in range(1, 5)])
xticklabels = [f'$10^{i}$' for i in range(1, 5)]

# Compute scale exactly as done in waipy
s0 = 2
start_period = 2
num_suboctaves = 32
dj = 1.0 / num_suboctaves
j1 = int(np.floor(np.log2(len(t_long) * 1 / start_period) * num_suboctaves))
scale = s0 * 2**(dj * np.arange(j1 + 1))
print(scale)

for k in range(3):
    params_name = params_names[k]
    ax_spectrum, ax_dual = ax_arr[int(3*k)], ax_arr[int(3*k) + 1: int(3*k) + 3]
    params = helpers.load(params_name, 'params', 'sweep', **kwargs)
    num_phy = params['bio']['num_phy']

    for j, ax in enumerate(ax_dual):
        data_label = data_labels[j]
        kwargs['data_label'] = data_label
        data_2d = helpers.load(params_name, 'data', 'sweep', **kwargs, err_on_fail=err_on_fail)
        data_2d['output'] = data_2d['output'].map(get_first)
        if j == 0:
            ax_spectrum.set_title(labels[k], fontsize=12)
            amps = amps_all[k]
            series = np.array([amps[i + 1] * s(t_long, forcing_periods[i], forcing_periods[i + 1]) for i in range(len(amps) - 1)]).sum(axis=0)
            series += amps[0] * np.sin(2 * np.pi * (t_long / 360 + np.random.random_sample()))

            # Get spectrum
            pkl_filename = f'amps_{params_name}.pkl'
            if load_spectrum:
                try:
                    with open(pkl_filename, 'rb') as f:
                        print(f'Reading output from {pkl_filename}')
                        result = pickle.load(f)
                except:
                    result = al.wavelet_spectrum_simple(series, sort='freq', rectify_bias=False, num_suboctaves=32,
                                                        return_raw=True)
                    with open(pkl_filename, 'wb') as f:
                        print(f'Writing output to {pkl_filename}')
                        pickle.dump(result, f)
                    with open(pkl_filename, 'rb') as f:
                        print(f'Reading output from {pkl_filename}')
                        result = pickle.load(f)
            else:
                result = al.wavelet_spectrum_simple(series, sort='freq', rectify_bias=False, num_suboctaves=32, return_raw=True)
                with open(pkl_filename, 'wb') as f:
                    print(f'Writing output to {pkl_filename}')
                    pickle.dump(result, f)
            periods = result['period']
            amps = result['global_ws']
            # scale2 = result['scale']
            # amps /= scale2
            amps = np.sqrt(amps)
            amps /= amps.max()

            ax_spectrum.plot(periods, amps)
            ax_spectrum.tick_params(reset=True, labelsize=12, which='both')
            ax_spectrum.set_xscale('log')
            ax_spectrum.set_xlim(xlim)
            ax_spectrum.set_xticks(xticks)
            ax_spectrum.set_xticklabels(xticklabels)

        ax.tick_params(reset=True, labelsize=12, which='both')
        ax.axvline(10000, color='gray', linewidth=1.5, linestyle='--')

        # Now flatten dataframe
        # x_label will be sat_scale, y_label will be frequency, and output will be amplitude
        data_flat = pd.DataFrame()
        for i in range(len(data_2d)):
            row = data_2d.iloc[i]
            if not isinstance(row['output'], (int, float)):
                freqs = np.array(row['output'][0])
                amps = np.array(row['output'][1])

                # Correct bias
                # amps /= scale[::-1]

                amps = np.sqrt(amps)

                periods = 1 / freqs
                lp = np.where(periods < 10000)[0]
                hp = np.where(periods >= 10000)[0]
                if periods[np.where(amps == amps.max())[0][0]] > 10000:
                    amps[lp] /= amps[lp].max()
                    amps[hp] /= amps[hp].max()
                else:
                    amps /= amps.max()

                # normalize amplitudes
                # amps = amps / np.amax(amps)
                sweep_vals = row[sweep_param]
                data_flat = pd.concat((data_flat, pd.DataFrame({'freq': freqs, 'amp': amps, 'sweep_param': sweep_vals})))

        # Get rid of zero frequency
        data_flat['period'] = 1/data_flat['freq']

        # Get matrices
        xf, yf, outf = helpers.data_from_pandas(data_flat, x_label='period', y_label='sweep_param',
                                                out_label='amp', nan=c.NAN_VALUE, extra_dim=False)
        outf = ma.masked_invalid(outf)

        ax.pcolormesh(xf, yf, outf, shading='auto', norm=colors.Normalize(vmin=vmin, vmax=vmax), cmap='binary')

        ymax = 0.04
        ymin = 0.001
        ax.add_patch(Rectangle((xlim[0], ymin), 90 - xlim[0], ymax - ymin, alpha=0.1))
        ax.add_patch(Rectangle((90, ymin), 360 - 90, ymax - ymin, color='green', alpha=0.1))
        ax.add_patch(Rectangle((360, ymin), 50000 - 360, ymax - ymin, color='yellow', alpha=0.1))

        ax.set_xscale('log')
        ax.set_xlim(xlim)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        yticks = [0.01, 0.02, 0.03, 0.04]
        ax.set_yticks([0] + yticks)
        if k == 0:
            if j == 1:
                ax.set_ylabel(xy[0][1], fontsize=14)
                ax.yaxis.set_label_coords(-0.25, 1.2)
                cbar_ax = fig.add_axes([0.82, 0.13, 0.02, 0.44])
                norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
                sm = plt.cm.ScalarMappable(norm=norm, cmap='binary')
                sm.set_array(np.array([]))
                cb = fig.colorbar(sm, ticks=levels, cax=cbar_ax)
                ticks = levels
                cbar_ax.set_ylabel('(Normalized) Wavelet Amplitude', labelpad=20, fontsize=12, rotation=270)
                cb.set_ticks(ticks)
                cbar_ax.set_yticklabels([int(x) if x >= 1 else x for x in ticks], fontsize=12)
                cb.ax.minorticks_off()
                cbar_ax.tick_params(labelsize=12)
                cbar_ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                ax.set_xlabel('T$_W$ (days)', fontsize=12, labelpad=0)
            fig.subplots_adjust(left=0.15, right=0.8, bottom=0.1, top=0.8, hspace=0.3)

            ax_spectrum.set_ylabel('(Normalized)\nWavelet Amplitude', fontsize=12, labelpad=20)
            # fig.suptitle(rf'$\tau_{{min}} = 0$ day$^{{-1}}$', fontsize=12)
            ax.set_yticklabels(['$0$'] + [f'${y:.2f}$' for y in yticks])
        else:
            ax.set_yticklabels([])
            ax_spectrum.set_yticklabels([])

alphabet = list(string.ascii_uppercase)
alphabet = [alphabet[x] for x in (0, 3, 6, 1, 4, 7, 2, 5, 8)]
for i, label in enumerate(alphabet[:ax_arr.shape[0]]):
    y = 1.2 if i in (0, 3, 6) else 1.09
    ax_arr[i].text(0.07, y, rf'\textbf{{{label}}}', transform=ax_arr[i].transAxes,
               fontsize=16, fontweight='bold', va='top', ha='right')
    ax_arr[i].grid(True, which='major')

plt.figtext(0.055, 0.71 - 0.17, r'{\bf Total P}', fontsize=13)
plt.figtext(0.07, 0.45 - 0.28, r'{\bf P$^s_1$}', fontsize=13)

# plt.savefig('paper/vary_taumax_stoch_2_monly.pdf', bbox_inches='tight', pad_inches=0)
# plt.savefig('paper/vary_taumax_stoch_2_renorm_biascorrect.pdf', bbox_inches='tight', pad_inches=0)
# plt.savefig('paper/vary_taumax_stoch_2_renorm_adjusted.pdf', bbox_inches='tight', pad_inches=0)
# plt.savefig('paper/vary_taumax_stoch_2_renorm_adjusted_monly.pdf', bbox_inches='tight', pad_inches=0)

# plt.savefig('paper/vary_taumax_stoch_2_renorm_adjusted_monly.pdf', bbox_inches='tight', pad_inches=0)
plt.show()
