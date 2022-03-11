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
import string

#######################################################################

plotting_threshold = 1e-3
data_ext = 'npy'

num_years = 3

# TIME SERIES

# We'll structure this so each row contains a sweep:
# Column 1: Individual P
# Column 2: Total P and Z (if applicable)
# Column 3 (if applicable): Individual Z
# This makes for twelve panels total for the constant tau sweep

sns.set_context('paper')
sns.set_style('white')

params_names = ['tau_pt3_ronly_no_osc', 'tau_pt3_zonly_no_osc', 'tau_pt3_strongpred_no_osc']

fig_titles = ['R-only', 'Z-only', 'R+Z']

phy_indiv_alpha = 1
phy_total_alpha = 1

total_lw = 0.8
indiv_lw = 1.0

fig, ax_arr = plt.subplots(2, 3, figsize=(10, 7))
fig.subplots_adjust(top=0.9, bottom=0.2)
# individuals
ylimits = [0, 30]

compartments = ['all', [0]]

for k in range(2):
    ax = ax_arr[k, :]
    for p_ind, params_name in enumerate(params_names):
        ax[p_ind].tick_params(reset=True, labelsize=12, which='both')
        eco = helpers.load(params_name, 'data', 'single', data_ext=data_ext)
        params = helpers.load(params_name, 'params', 'single')
        bio = params['bio']

        num_years_run = params['num_years']

        t_y = np.linspace(0, num_years, int(num_years * c.NUM_DAYS_PER_YEAR)) + num_years_run - num_years

        phy = helpers.restrict_ts(eco, params, compartments=[{'phy': compartments[k]}], num_years=num_years, kind='total')

        ################################################################################################

        ax[p_ind].plot(t_y, phy, linewidth=indiv_lw, alpha=phy_indiv_alpha)
        ax[p_ind].set_ylim(ylimits)

        if p_ind == 0 and k == 1:
            ax[p_ind].set_xlabel('Time (days)', labelpad=10, fontsize=11)
        if p_ind == 0 and k == 1:
            ax[p_ind].set_ylabel('Concentrations ($\mu$M C)', labelpad=10, fontsize=11)
            ax[p_ind].yaxis.set_label_coords(-0.2, 1.1)

        if p_ind != 0:
            ax[p_ind].set_yticklabels([])

        xlist = num_years_run - np.arange(num_years, -0.5, -.5)
        xlim = [num_years_run - num_years, num_years_run]

        ax[p_ind].set_xlim(xlim)
        ax[p_ind].set_xticks(xlist)
        xlist = ((xlist-xlist[0]) * c.NUM_DAYS_PER_YEAR).astype(int).tolist()

        if k == 1:
            ax[p_ind].set_xticklabels(xlist)
        else:
            ax[p_ind].set_xticklabels([])
        if k == 0:
            ax[p_ind].set_title(fig_titles[p_ind], fontsize=12)

    for i in range(3):
        ax[i].grid(True, which='major')

alphabet = list(string.ascii_uppercase)
ax = ax_arr.ravel()
for i in range(6):
    label = alphabet[i]
    ax[i].text(0.09, 1.1, rf'\textbf{{{label}}}', transform=ax[i].transAxes,
               fontsize=16, fontweight='bold', va='top', ha='right')
    ax[i].grid(True, which='major')

plt.figtext(0.01, 0.77, r'{\bf Total P}', fontsize=13)
plt.figtext(0.025, 0.35, r'{\bf P$^s_1$}', fontsize=13)

fig.suptitle('Intrinsic Variability Cases', fontsize=13)
# plt.savefig('paper/timeseries_r_z_rz_total_p1s.pdf', bbox_inches='tight')
plt.show()
