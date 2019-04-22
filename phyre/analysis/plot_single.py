"""
plot_single.py: plot results from single simulation
"""

from typing import Dict, List, Union, Optional, Tuple
import matplotlib.axes as mat_ax
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from phyre import helpers
import phyre.constants as c
import phyre.analysis.analysis as al
import copy


class SinglePlotter(object):
    """ Class for plotting output from a single run

    Attributes
    ----------
    params_name: str, optional
        Name of parameter set to load
    eco_in: numpy.ndarray, optional
        Ecosystem we're passing to plotter
    t_in: numpy.ndarray, optional
        Times saved in ecosystem
    params: Dict, optional
        We can also pass params to init and use this to load other things

    num_phy: int
        number of phytoplankton
    num_compartments: int
        number of compartments total
    num_days: int
        number of days in simulation
    num_years: int
        number of years in simulation

    """

    def __init__(self, params_name: str=None, params: Dict=None, eco_in: np.ndarray=None, t_in: np.ndarray=None,
                 details_str: str=None):

        """Initialize SinglePlotter

        Parameters
        ----------
        params_name
            Name of parameter set to load
        details_str
            for loading specific files
        eco_in
            Ecosystem we're passing to plotter
        t_in
            Times saved in ecosystem
        params
            We can also pass params to init and use this to load other things

        """

        if params is None:  # use this if we're loading params from directory
            self.params_name = params_name
            self.eco_in = helpers.load(self.params_name, 'data', 'single', data_label='time_series',
                                       details_str=details_str)
            self.t_in = helpers.load(self.params_name, 'data', 'single', data_label='time', details_str=details_str)
            self.params = helpers.load(self.params_name, 'params', 'single', details_str=details_str)

        # pass directly
        else:
            self.eco_in = eco_in
            self.t_in = t_in
            self.params = params

        # establish common parameter values
        self.num_phy = self.params['bio']['num_phy']
        self.num_compartments = self.params['bio']['num_compartments']
        self.num_days = self.params['num_days']
        self.num_years = self.num_days / c.NUM_DAYS_PER_YEAR

    def time_series_plot(self, kind: str='indiv', ax: mat_ax.Axes=None,
                         plotting_threshold: float=-np.infty, labels: List=None, return_lines:bool=True,
                         num_years: int=None, num_days: int=None, color_kw: Dict=None,
                         plot_kw: Dict=None, legend_kw: Dict=None, legend_off: bool=False,
                         resize_box: tuple=(0, 0.2, 0, 0.7),
                         compartments: List[Dict]=[{'phy': 'all'}], res_to_carbon: Union[bool, int, list]=True,
                         nit_combine: bool=False, phy_index: int=None):

        """Plot one or more time series from the stored ecosystem

        Parameters
        ----------
        kind
            'indiv' (default), 'shannon', 'total'
        ax
            Axis we'd like to plot on.
        num_years
            How many years to include
        num_days
            How many days?
        plotting_threshold
            Don't plot any items below this threshold
        labels
            Legend labels
        resize_box
            Compress figure box
        legend_off
            Leave off legend
        return_lines
            Returns lines associated with plot object
        res_to_carbon
            Convert resources to carbon units
        nit_combine
            Combine nitrate and ammonium?
        phy_index
            If we're plotting resource, which phyto compartment(s) to use to convert to carbon units
        plot_kw
            Keyword arguments to pass to matplotlib
        color_kw
            colors: list of colors to use for plotting
            cmap: used to create list of colors
        legend_kw
            dict of keyword arguments to be passed directly to `legend` method of `ax`
        compartments
            List of compartment dictionaries

        """

        ty_vec = self.t_in / c.NUM_DAYS_PER_YEAR

        if ax is None:
            ax = plt.subplot(111)

        if num_years is not None:
            ty_vec = helpers.get_last_n_years(self.t_in, num_years) / c.NUM_DAYS_PER_YEAR
        if num_days is not None:
            ty_vec = helpers.get_last_n_days(self.t_in, num_days) / c.NUM_DAYS_PER_YEAR
            num_years = num_days / c.NUM_DAYS_PER_YEAR

        box1 = ax.get_position()
        ax.set_position([box1.x0 + box1.width * resize_box[0], box1.y0 + box1.height * resize_box[1],
                         box1.width + box1.width * resize_box[2], box1.height * resize_box[3]])

        ax.set_xlim(left=self.num_years - num_years, right=self.num_years)

        plt_obj = list()

        # colors
        if kind in ('shannon', 'total'):
            eco = helpers.restrict_ts(self.eco_in, self.params, num_years=num_years, num_days=num_days, kind=kind,
                                      compartments=compartments)
            if color_kw is not None:
                if 'colors' in color_kw:
                    colors = iter(color_kw['colors'])
                else:
                    colors = iter(helpers.color_cycle(1, cmap=color_kw.get('cmap')))
            else:
                colors = iter(helpers.color_cycle(1))

            if np.mean(eco) > plotting_threshold:
                line, = ax.plot(ty_vec, eco, **plot_kw if plot_kw is not None else {})
                line.set_color(next(colors))
                plt_obj.append(line)
            if return_lines:
                return plt_obj
        else:
            eco = helpers.restrict_ts(self.eco_in, self.params, num_years=num_years, num_days=num_days)
            name_list = helpers.get_name_list(self.params, compartments=compartments, for_plot=True)

            new_name_list = list()

            for list_dict in compartments:
                key = list(list_dict.keys())[0]

                if list_dict[key] == 'all' or list_dict[key] is None:
                    indices = list(range(self.params['bio']['num_{}'.format(key)]))
                else:
                    indices = list_dict[key]

                conversion = None

                # get name_list for just this set of compartments
                curr_name_list = copy.deepcopy([name_list[x] for x in range(len(indices))])
                single_nit = self.params.get('bio').get('single_nit')
                silicate_off = self.params.get('bio').get('silicate_off')
                if key == 'res':
                    if res_to_carbon not in (False, 0):
                        conversion = al.res_to_carbon_conversions(eco, self.params, phy_index=phy_index)

                    if silicate_off not in (False, 0) and c.SIL_INDEX in indices:
                        indices.remove(c.SIL_INDEX)
                        curr_name_list.remove(r'R$_5$')  # we only have four nutrients now, remove last one...

                    if single_nit in (1, True) or nit_combine in (1, True) and c.NH4_INDEX in indices:  # single_nit overrides nit_combine
                        indices.remove(c.NH4_INDEX)

                    if nit_combine in (1, True) and single_nit in (False, None, 0):
                        if r'R$_1$' in name_list and r'R$_2$' in name_list:
                            ind = name_list.index(r'R$_1$')
                            curr_name_list[ind] = r'R$_1$ + R$_2$'
                            curr_name_list.remove(r'R$_2$')

                    if single_nit in (1, True):
                        if r'R$_4$' in name_list:
                            curr_name_list.remove(r'R$_4$')

                for ind, r in enumerate(indices):
                    start_index = list(helpers.eco_indices(key, self.params['bio']))[0]
                    name = curr_name_list[ind]
                    output = np.squeeze(eco[start_index + r, :])

                    # convert resource to phyto units
                    if key == 'res':
                        if nit_combine and r == 0:
                            output += np.squeeze(eco[start_index + 1, :])

                        if res_to_carbon is not False:
                            output = conversion[r, :]

                    if np.mean(output) >= plotting_threshold:
                        new_name_list.append(name)
                        line, = ax.plot(ty_vec, output, label=name, **plot_kw if plot_kw is not None else {})
                        plt_obj.append(line)

            # NOW set colors
            if color_kw is not None:
                if 'colors' in color_kw:
                    colors = iter(color_kw.get('colors'))
                else:
                    colors = iter(helpers.color_cycle(len(new_name_list), cmap=color_kw.get('cmap')))
            else:
                colors = iter(helpers.color_cycle(len(new_name_list)))

            for i in range(len(plt_obj)):
                plt_obj[i].set_color(next(colors))

            if labels:
                new_name_list = labels

            if len(new_name_list) > 0:
                if not legend_off:
                    if legend_kw is not None:
                        ax.legend(labels=new_name_list, **legend_kw)
                    else:
                        ax.legend(labels=new_name_list, loc='upper center', bbox_to_anchor=(0.5, -0.2),
                                  fancybox=True, shadow=True, ncol=5)

            if return_lines:
                return plt_obj

    # plot total biomass
    def biomass_plot(self, num_years: int=None, num_days: int=None, ax: mat_ax.Axes=None):

        """Plot the total biomass

        Parameters
        ----------
        ax
            Axis we'd like to plot on
        num_years
            How many years to include
        num_days
            How many days?

        """

        ty_vec = self.t_in / c.NUM_DAYS_PER_YEAR
        num_years = self.num_years

        if num_years is not None:
            ty_vec = helpers.get_last_n_years(self.t_in, num_years) / c.NUM_DAYS_PER_YEAR
        if num_days is not None:
            ty_vec = helpers.get_last_n_days(self.t_in, num_days) / c.NUM_DAYS_PER_YEAR
            num_years = num_days / c.NUM_DAYS_PER_YEAR

        biomass = al.get_total(self.eco_in, self.params, num_years=num_years)

        # biomass

        # Clean up formatting (no scientific notation)
        if ax is None:
            ax = plt.gca()

        ax.plot(ty_vec, biomass, label='Total Biomass', linewidth=2)

    def spectral_plot(self, kind: str='indiv', ax: mat_ax.Axes=None, compartments: List[Dict]=None, num_years: int=None,
                      num_days: int=None, spectrum_kw: Dict=None, plot_kw: Dict=None, power_spectrum: bool=False,
                      labels: List = None, legend_kw: Dict=None, return_plots: bool=False) \
            -> Optional[List]:

        """Plot one or more time series from the stored ecosystem

        Parameters
        ----------
        ax
            Axis we'd like to plot on.
        kind
            What are we plotting the spectrum of? ('indiv', 'shannon', 'total')
        num_years
            How many years to include
        num_days
            How many days?
        power_spectrum
            Instead of amps, plot abs(amps)^2
        legend_kw
            Keywords for formatting legend
        spectrum_kw
            Dict of keyword arguments to pass to filtered_spectrum method of analysis
            See `analysis.filtered_spectrum` for keyword arguments
        plot_kw
            Arguments to pass to matplotlib
        return_plots
            Return figure handles
        compartments
            List of compartment dictionaries

        """

        handles = list()

        if spectrum_kw is None:
            spectrum_kw = {}

        freqs, amps = al.filtered_spectrum(self.eco_in, self.params, compartments=compartments, num_years=num_years,
                                           num_days=num_days,
                                           kind=kind, **spectrum_kw)

        name_list = helpers.get_name_list(self.params, compartments=compartments, for_plot=True)

        # sort by frequency, small to large
        for i in range(len(freqs)):
            ind = np.argsort(freqs)[i]
            freqs_curr = np.array(freqs[i])[ind]
            amps_curr = np.power(np.abs(np.array(amps[i])[ind]), 2) if power_spectrum else np.array(amps[i])[ind]
            if ax is None:
                handles.append(plt.plot(freqs_curr, amps_curr, **plot_kw if plot_kw is not None else {})[0])
                ax = plt.gca()
            else:
                handles.append(ax.plot(freqs_curr, amps_curr, **plot_kw if plot_kw is not None else {})[0])

        if labels:
            name_list = labels

        if legend_kw is not None:
            ax.legend(labels=name_list, **legend_kw)
        else:
            ax.legend(labels=name_list, loc='upper center', bbox_to_anchor=(0.5, -0.2),
                      fancybox=True, shadow=True, ncol=5)

        box1 = ax.get_position()
        ax.set_position([box1.x0, box1.y0 + box1.height * 0.2,
                         box1.width, box1.height * 0.7])

        if return_plots:
            return handles

    # plot phase portrait
    def phase_plot(self, ax: mat_ax.Axes=None, fig: List=None, compartments: List[Dict]=None, num_years: int=None,
                   num_days: int=None,
                   plot3d: bool=False, res_phy: bool=False):

        """Phase plot from stored ecosystem

        Parameters
        ----------
        fig
            Figure we're plotting on. Only used if plot3d is true
        ax
            Axis we'd like to plot on.
        num_years
            How many years to include
        num_days
            How many days to include?
        num_days
            How many days?
        plot3d
            Plot the three arguments with the highest average concentrations against each other in 3D
        res_phy
            Plot total resources vs. total phytoplankton. Only applies if plot3d=False
        compartments
            List of compartment dictionaries

        """

        eco = helpers.restrict_ts(self.eco_in, self.params, num_years=num_years, num_days=num_days,
                                  compartments=compartments)

        # find time average
        eco_time_avg = np.mean(eco, 1)

        # three largest quantities
        indices = eco_time_avg.argsort()[::-1][:3]

        # indices
        name_list = helpers.get_name_list(self.params, for_plot=True)

        # take the last end_pct of entries
        if plot3d:
            ax = Axes3D(fig)
            num_divisions = 5
            ax.locator_params(nbins=num_divisions, nticks=num_divisions)

            ax.plot(eco[indices[0], :],
                    eco[indices[1], :],
                    eco[indices[2], :])
            ax.set_xlabel(name_list[indices[0]])
            ax.set_ylabel(name_list[indices[1]])
            ax.set_zlabel(name_list[indices[2]])

        else:
            if ax is None:
                ax = fig.add_subplot(111)

            # plot P vs. R
            if res_phy:
                bio = self.params['bio']
                ax.plot(np.sum(eco[helpers.eco_indices('phy', bio), :], 0),
                        np.sum(eco[helpers.eco_indices('res', bio), :], 0),
                        linewidth=2)
                ax.set_xlabel('P')
                ax.set_ylabel('R')

            else:
                # plot two largest quantities against each other
                ax.plot(eco[indices[0], :],
                        eco[indices[1], :],
                        linewidth=2.0)
                ax.set_xlabel(name_list[indices[0]])
                ax.set_ylabel(name_list[indices[1]])

        return ax
