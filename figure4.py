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
sweep_param = 'turnover_max'

sns.set_context('paper')
sns.set_style('white')

vmin = 0
vmax = 1

level_spacing = 0.1
levels = np.arange(vmin, vmax+level_spacing, level_spacing)

fig, ax_all = plt.subplots(2, 3, figsize=(15, 10))
ax_all = ax_all.ravel()
ax_all_grouped = [ax_all[:3], ax_all[3:]]

# round to nearest whole number
class nf(float):
    def __repr__(self):
        return f'{self:.0f}'

params_names = ['nopred_vary_taumax', 'simplecom_vary_taumax', 'strongpred_vary_taumax']
labels = ['R-only', 'Z-only', 'R+Z']
legend_kw = {'fontsize': 10, 'loc': 'upper center', 'bbox_to_anchor': (0.5, 1.15), 'ncol': 2}

proxy = []
data_labels = ['wavelet_phy_total', 'wavelet_phy_p1s']

xlim = [10, 54000]
xticks = np.array([10 ** i for i in range(1, 5)])
xticklabels = [f'$10^{i}$' for i in range(1, 5)]

# Compute scale exactly as done in waipy
s0 = 2
start_period = 2
num_suboctaves = 32
t_long = np.array(list(range(int(150 * c.NUM_DAYS_PER_YEAR))))
dj = 1.0 / num_suboctaves
j1 = int(np.floor(np.log2(len(t_long) * 1 / start_period) * num_suboctaves))
scale = s0 * 2**(dj * np.arange(j1 + 1))

for j in range(2):
    data_label = data_labels[j]
    ax_arr = ax_all_grouped[j]
    kwargs = {'data_label': data_label, 'cluster_kw': {'num_clusters': num_clusters}, 'data_ext': data_ext}
    data_kw = {'in_labels': tuple([v[0] for v in xy])}
    kwargs.update(**{'pd_kw': data_kw})
    for k, ax in enumerate(ax_arr):
        params_name = params_names[k]
        the_plot = None

        params = helpers.load(params_name, 'params', 'sweep', **kwargs)
        num_phy = params['bio']['num_phy']

        data_2d = helpers.load(params_name, 'data', 'sweep', **kwargs, err_on_fail=err_on_fail)
        def get_first(x):
            try:
                return x[0]
            except:
                return c.NAN_VALUE
        data_2d['output'] = data_2d['output'].map(get_first)

        sweep_vals = {}
        period_vals = data_2d['turnover_period'].unique()
        max_vals = data_2d['turnover_max'].unique()
        period_ind = 26

        ymax = 0.04
        ymin = 0.001
        ax.add_patch(Rectangle((xlim[0], ymin), 90 - xlim[0], ymax - ymin, alpha=0.1))
        ax.add_patch(Rectangle((90, ymin), 360 - 90, ymax - ymin, color='green', alpha=0.1))
        ax.add_patch(Rectangle((360, ymin), 50000 - 360, ymax - ymin, color='yellow', alpha=0.1))

        ax.set_xscale('log')
        ax.set_xlim(xlim)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)

        select_bool = (data_2d['turnover_period'] == period_vals[period_ind])
        data_2d = data_2d.loc[select_bool, ['output', 'turnover_max']].reset_index(drop=True)

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

            periods = 1 / freqs
            lp = np.where(periods < 10000)[0]
            hp = np.where(periods >= 10000)[0]
            if periods[np.where(amps == amps.max())[0][0]] > 10000:
                amps[lp] /= amps[lp].max()
                amps[hp] /= amps[hp].max()
            else:
                amps /= amps.max()

            # # normalize amplitudes
            # amps = amps / np.amax(amps)

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

        ax.axvline(x=period_vals[period_ind], color='r', linewidth=2, alpha=0.5)

        if j == 0:
            ax.set_title(labels[k], fontsize=12)
        # periods = data_flat['period'].unique()[::-1]
        # yticks = periods[::10]
        # ax.set_yticks(yticks)
        # ax.set_yticklabels([f'{y:.0f}' for y in yticks])

        ax.tick_params(reset=True, labelsize=12, which='both')

        if k == 0 and j == 0:
            xloc = 600
            ax.text(xloc, 0.02, r'T$_\text{{W}}$ = T$_{\text{for}}$', color='r', fontsize=10)

        if j == 1 and k == 1:
            ax.set_xlabel('T$_{W}$ (days)', fontsize=14)

        if j == 1 and k == 0:
            ax.set_ylabel(xy[0][1], fontsize=14)
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

        yticks = [0.01, 0.02, 0.03, 0.04]
        ax.set_yticks([0] + yticks)
        if k != 0:
            ax.set_yticklabels([])
        else:
            ax.set_yticklabels(['$0$'] + [f'${y:.2f}$' for y in yticks])

        # ax.axvline(10000, color='gray', linewidth=1.5, linestyle='--')

        fig.subplots_adjust(right=0.78, bottom=0.22, hspace=0.3, top=0.8)

    alphabet = list(string.ascii_uppercase)
    for i, label in enumerate(alphabet[:6]):
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
# plt.savefig('paper/vary_taumax_taumax_total_p1s_renorm.pdf', bbox_inches='tight')
plt.show()
