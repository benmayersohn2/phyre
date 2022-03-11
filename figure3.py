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
import string

########################################################################

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

num_clusters = 200

err_on_fail = False

data_ext = 'npy'

xy = [('turnover_max', r'$\tau_{max}$ (day$^{-1}$)'), ('turnover_period', r'$T_{\text{for}}$ (days)')]
sweep_param = 'turnover_period'

sns.set_context('paper')
sns.set_style('white')

vmin = 0
vmax = 1

level_spacing = 0.1
levels = np.arange(vmin, vmax+level_spacing, level_spacing)

num_cols = 3
fig, ax_all = plt.subplots(2, num_cols, figsize=(15, 10))
ax_all = ax_all.ravel()
ax_all_grouped = [ax_all[:num_cols], ax_all[num_cols:]]

num_plots = len(ax_all)

# round to nearest whole number
class nf(float):
    def __repr__(self):
        return f'{self:.0f}'

# params_names = ['nopred_vary_taumax', 'simplecom_vary_taumax', 'weakpred_vary_taumax', 'strongpred_vary_taumax']
# labels = ['R-only', 'Z-only', 'R+Z (weak)', 'R+Z']
params_names = ['nopred_vary_taumax', 'simplecom_vary_taumax', 'strongpred_vary_taumax']
labels = ['R-only', 'Z-only', 'R+Z']
legend_kw = {'fontsize': 10, 'loc': 'upper center', 'bbox_to_anchor': (0.5, 1.15), 'ncol': 2}

proxy = []

xlim = [10, 54000]
xticks = np.array([10 ** i for i in range(1, 5)])
xticklabels = [f'$10^{i}$' for i in range(1, 5)]

# Compute scale exactly as done in waipy
s0 = 2
start_period = 2
num_suboctaves = 32
dj = 1.0 / num_suboctaves
t_long = np.array(list(range(int(150 * c.NUM_DAYS_PER_YEAR))))
j1 = int(np.floor(np.log2(len(t_long) * 1 / start_period) * num_suboctaves))
scale = s0 * 2**(dj * np.arange(j1 + 1))

data_labels = ['wavelet_phy_total', 'wavelet_phy_p1s']
for j in range(2):
    data_label = data_labels[j]
    ax_arr = ax_all_grouped[j]
    kwargs = {'data_label': data_label, 'cluster_kw': {'num_clusters': num_clusters}, 'data_ext': data_ext}
    data_kw = {'in_labels': tuple([v[0] for v in xy])}
    kwargs.update(**{'pd_kw': data_kw})
    for k, ax in enumerate(ax_arr):
        params_name = params_names[k]
        the_plot = None

        # Add lines for individual simulations from Figure 2
        if k == 0:
            pass

        params = helpers.load(params_name, 'params', 'sweep', **kwargs)
        num_phy = params['bio']['num_phy']

        data_2d = helpers.load(params_name, 'data', 'sweep', **kwargs, err_on_fail=err_on_fail)
        def get_first(x):
            try:
                return x[0]
            except:
                return c.NAN_VALUE
        data_2d['output'] = data_2d['output'].map(get_first)

        # Get rid of extra entries
        extra_periods = [30, 90, 120, 180, 270, 360, 540, 720, 1080, 1440]
        data_2d = data_2d.loc[~data_2d['turnover_period'].isin(extra_periods)].reset_index(drop=True)

        sweep_vals = {}
        period_vals = data_2d['turnover_period'].unique()
        max_vals = data_2d['turnover_max'].unique()

        # max_ind = 19
        max_ind = 39
        ymax = 1900
        ax.add_patch(Rectangle((xlim[0], 9), 90-xlim[0], ymax, alpha=0.1))
        ax.add_patch(Rectangle((90, 9), 360-90, ymax, color='green', alpha=0.1))
        ax.add_patch(Rectangle((360, 9), 50000 - 360, ymax, color='yellow', alpha=0.1))

        ax.set_xscale('log')
        ax.set_xlim(xlim)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)

        # ax.axhline(y=90, linestyle='--', linewidth=0.5)
        # ax.axhline(y=360, linestyle='--', linewidth=0.5)
        # ax.add_patch(Rectangle((xlim[0], 9), 55108, 90-9, facecolor='#1f77b400', edgecolor='#9467bd10'))
        # ax.add_patch(Rectangle((xlim[0], 90), 55108, 360-90, facecolor='#1f77b400', edgecolor='#9467bd10', hatch='//'))
        # ax.add_patch(Rectangle((xlim[0], 360), 55108, ymax-360, facecolor='#1f77b400', edgecolor='#9467bd10', hatch='||'))

        if k == 0:
            print(max_vals[max_ind])
            print(len(np.unique(data_2d['turnover_period'])))

        select_bool = (data_2d['turnover_max'] == max_vals[max_ind])
        data_2d = data_2d.loc[select_bool, ['output', 'turnover_period']].reset_index(drop=True)

        s = 3

        # Now flatten dataframe
        # x_label will be sat_scale, y_label will be frequency, and output will be amplitude
        data_flat = pd.DataFrame()
        for i in range(len(data_2d)):
            row = data_2d.iloc[i]
            freqs = np.array(row['output'][0])
            amps = np.array(row['output'][1])

            # Bias correct
            # amps /= scale[::-1]

            amps = np.sqrt(amps)

            # normalize amplitudes
            periods = 1/freqs
            lp = np.where(periods < 10000)[0]
            hp = np.where(periods >= 10000)[0]
            if periods[np.where(amps == amps.max())[0][0]] > 10000:
                amps[lp] /= amps[lp].max()
                amps[hp] /= amps[hp].max()
            else:
                amps /= amps.max()
            # amps = np.power(amps, 0.75) / np.amax(np.power(amps, 0.75))
            # amps = np.log(amps) / np.amax(np.log(amps))

            sweep_vals = row[sweep_param]
            data_flat = pd.concat((data_flat, pd.DataFrame({'freq': freqs, 'amp': amps, 'sweep_param': sweep_vals})))

        # Get rid of zero frequency
        data_flat['period'] = 1/data_flat['freq']

        # Get matrices
        xf, yf, outf = helpers.data_from_pandas(data_flat, x_label='period', y_label='sweep_param',
                                                out_label='amp', nan=c.NAN_VALUE, extra_dim=False)

        outf = ma.masked_invalid(outf)

        # the_plot = ax.contourf(xf, yf, outf, norm=colors.Normalize(vmin=vmin, vmax=vmax), cmap='binary',
        #                        levels=levels.tolist(), extend='both', alpha=0.4)  # linewidths=2
        the_plot = ax.pcolormesh(xf, yf, outf, shading='auto', norm=colors.Normalize(vmin=vmin, vmax=vmax), cmap='binary')  # linewidths=2

        # ax.clabel(the_plot, fmt='%r', inline=True, fontsize=10)

        ylim = ax.get_ylim()

        x = np.linspace(max(xlim[0], ylim[0]), min(xlim[-1], ylim[-1]), 1000)
        ax.plot(x, x, color='r', linewidth=2, alpha=0.5)
        if j == 0 and k == 0:
            ax.text(500, 200, r'T$_{\text{W}}$ = T$_{\text{for}}$', color='r', fontsize=10)

        if j == 0:
            ax.set_title(labels[k], fontsize=12)
        ax.set_yscale('log')
        ax.set_xscale('log')
        # periods = data_flat['period'].unique()[::-1]
        # yticks = periods[::10]
        # ax.set_yticks(yticks)
        # ax.set_yticklabels([f'{y:.0f}' for y in yticks])

        ax.tick_params(reset=True, labelsize=12, which='both')
        # ax.axvline(10000, color='gray', linewidth=1.5, linestyle='--')

        if j == 1 and k == 1:
            ax.set_xlabel('T$_{W}$ (days)', fontsize=14)

        if j == 1 and k == 0:
            ax.set_ylabel(xy[1][1], fontsize=14)
            ax.yaxis.set_label_coords(-0.2, 1.1)
            cbar_ax = fig.add_axes([0.82, 0.15, 0.02, 0.7])
            norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
            sm = plt.cm.ScalarMappable(norm=norm, cmap='binary')
            sm.set_array(np.array([]))
            cb = fig.colorbar(sm, ticks=levels, cax=cbar_ax)
            cbar_ax.set_ylabel('(Normalized) Wavelet Amplitude', labelpad=20, fontsize=12, rotation=270)
            ticks = levels
            # fig.suptitle(rf'$\tau_{{\text{{min}}}} = 0$ day$^{{-1}}$, $\tau_{{\text{{max}}}} = {max_vals[max_ind]:.02f}$ day$^{{-1}}$', fontsize=12)
            cb.set_ticks(ticks)
            cbar_ax.set_yticklabels([int(x) if x >= 1 else x for x in ticks], fontsize=12)
            cb.ax.minorticks_off()
            cbar_ax.tick_params(labelsize=12)
            cbar_ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

        if k != 0:
            ax.set_yticklabels([])

        # if j == 1:
        #     ax.set_xticklabels([])

        fig.subplots_adjust(right=0.78, bottom=0.22, hspace=0.3, top=0.8)

    alphabet = list(string.ascii_uppercase)
    for i, label in enumerate(alphabet[:num_plots]):
        ax_all[i].text(0.07, 1.1, rf'\textbf{{{label}}}', transform=ax_all[i].transAxes,
                   fontsize=16, fontweight='bold', va='top', ha='right')
        ax_all[i].grid(True, which='major')

    # ax_arr[0].axvline(190, ymin=0.2, ymax=0.3, linewidth=2, color='blue', alpha=0.4)
    # ax_arr[0].text(250, 30, '180 days', color='blue', alpha=0.6)

    # ax_arr[2].axvline(39, ymin=0.9, ymax=1, linewidth=2, color='purple', alpha=0.4)
    # ax_arr[2].text(3, 1300, '39 days', color='purple', alpha=0.6)
    # plt.savefig('paper/vary_taumax_period_total_p1s.pdf', bbox_inches='tight')

plt.figtext(0.025, 0.67, r'{\bf Total P}', fontsize=13)
plt.figtext(0.04, 0.35, r'{\bf P$^s_1$}', fontsize=13)
# plt.savefig('paper/vary_taumax_period_total_p1s_renorm.pdf', bbox_inches='tight')
plt.show()
