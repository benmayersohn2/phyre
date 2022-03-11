"""
plot_single_example_debug.py: Plot of RHS terms stored during run of single_example.py (see `debug_dict_setup` in helpers.py)
"""

import matplotlib.pyplot as plt
from phyre import helpers
from phyre import constants as c
import numpy as np
import seaborn as sns

plt.switch_backend('Qt5Agg')

########################################################################

params_name = 'example'

################################################################################################

# DEBUG

out_list = list()
out_keys = list()
xaxis = None

num_years_plot = 3

params = helpers.load(params_name, 'params', 'single')
res_phy_stoich_ratio = params.get('bio').get('res_phy_stoich_ratio')
num_phy = params.get('bio').get('num_phy')
num_zoo = params.get('bio').get('num_zoo')

debug = helpers.load(params_name, 'debug', 'single')
eco = helpers.load(params_name, 'data', 'single')
phy = eco[helpers.eco_indices('phy', params=params), :]
zoo = eco[helpers.eco_indices('zoo', params=params), :]
zoo_prey_pref = params.get('bio').get('zoo_prey_pref')

# we can compute res_zoo_makeup_ratio
prey = np.zeros((num_phy+1, num_zoo))
t, ind = np.unique(debug['t'][0], return_index=True)

t_y = helpers.get_last_n_years(t, num_years_plot) / c.NUM_DAYS_PER_YEAR

index = 0  # 0 = nitrate (or nitrogen if single_nit = True), 2 = silicate, 3 = phosphate, 4 = iron

keys = list(debug.keys())
keys.remove('res_zoo_remin_frac')
keys.remove('t')

single_nit = params.get('bio').get('single_nit')

res_zoo_remin_frac = debug['res_zoo_remin_frac']
res_zoo_stoich_ratio = np.array(res_zoo_remin_frac)

fig, ax = plt.subplots(1, 1)
plt.subplots_adjust(bottom=0.3)

if not single_nit:
    res_zoo_stoich_ratio[:, 0, :] = res_zoo_stoich_ratio[:, 1, :]

for key in keys:

    mat = debug.get(key)
    if mat is not None:
        total = np.zeros((len(mat[0, :]),))

        if 'res_' in key:
            total = mat[index, :]
        if 'phy_' in key:
            for i in range(num_phy):
                total += res_phy_stoich_ratio[index, i] * mat[i, :]

        if 'zoo_' in key:
            for i in range(len(total)):
                for n in range(num_zoo):
                    total[i] += res_zoo_stoich_ratio[n, index, i] * mat[n, i]

        if np.amax(np.abs(total)) > 1e-8:
            out_list.append(total.tolist())
            out_keys.append(key)

        ax.plot(t_y, helpers.get_last_n_years(total[ind], num_years_plot), label=key)


sns.set_context('paper')
sns.set_style('white')

ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4)
ax.set_xlabel('Time (Days)')
ax.set_title('RHS Terms (Nitrogen Units)')
num_steps = 2 * (t_y[-1] - t_y[0])
xlist = np.linspace(t_y[0], t_y[-1], int(num_steps))
xlim = [xlist[0], xlist[-1]]

for i in range(3):
    ax.set_xlim(xlim)

ax.set_xticks(xlist)
ax.set_xticklabels(((xlist - xlist[0]) * c.NUM_DAYS_PER_YEAR).astype(int))

plt.savefig('plots/single_example_debug.pdf', bbox_inches='tight', pad_inches=0)

if len(out_list) == 0:
    print('No contributions exceed threshold! Nothing to plot.')
else:
    plt.show()
