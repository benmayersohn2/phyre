"""
analysis.py: Functions for analyzing time series post-simulation
"""

from typing import List, Dict, Tuple, Union, Callable
import numpy as np
from phyre import helpers
from phyre import constants as c
import copy


def get_ts(eco: np.ndarray, params: Dict, num_years: int=None, compartments: List[Dict]=None,
           num_days: int=None,
           kind: str='indiv'):
    """Returns the input if kind == 'indiv'; shannon entropy if 'shannon'; and total biomass if 'total'

    Parameters
    ----------
    eco
        Ecosystem output
    kind
        'indiv', 'total' or 'shannon'
    params
        Dict of parameters
    num_years
        How many years do we want to include?
    num_days
        How many days to include?
    compartments
        List of compartment dictionaries

    """

    if kind == 'shannon':
        return shannon_ent_rel(eco, params, num_years=num_years, num_days=num_days)

    if kind == 'total':
        return get_total(eco, params, compartments=compartments, num_years=num_years, num_days=num_days)

    return helpers.restrict_ts(eco, params, compartments=compartments, num_years=num_years, num_days=num_days)


def apply_functions(params_name: str, functions: List[Tuple[str, Callable, Dict]], cluster_kw: Dict=None,
                    run_type: str='sweep', data_ext: str='npy'):
    """Apply function(s) to output data. Use for sweep outputs across clusters

    Parameters
    ----------
    params_name
        Name of parameter set
    functions
        The functions we'd like to apply to our output after each run
    cluster_kw
        'cluster' (int): cluster index
        'num_clusters' (int): total number of clusters in sweep
        Only applies if run_type == 'sweep'
    run_type
        'single' or 'sweep' ('sweep' by default)
    data_ext
        data extension to save output

    """

    def apply_single(params_orig):

        eco_in = helpers.load(params_name, 'data', 'single', data_label='time_series')
        fn_count = 0

        t0 = params_orig['t0']
        t_final = params_orig['t_final']
        dt = params_orig['dt_save']
        t_save = np.array(np.linspace(t0, t_final - dt, params_orig['num_days']))

        for output in functions:

            output_name = output[0]
            output_fn = output[1]
            output_kw = output[2]

            if np.isnan(eco_in).any():

                # store as nans with same dimensions as output should be
                fake_output = helpers.build_mock_eco_output(t_save, params_orig)

                out = np.full(np.shape(output_fn(fake_output, t_save, params_orig, output_kw)),
                              c.NAN_VALUE).tolist()
            else:
                out = output_fn(eco_in, t_save, params_orig, output_kw)

            helpers.save(params_name, 'data', 'single', output=out,
                         data_label=output_name, functions=functions, data_ext=data_ext)

            fn_count += 1

    def apply_to_cluster(params_orig):
        sweeps = params_orig['sweep']['pp']
        val_list = tuple(sweep[-1] for sweep in sweeps)
        val_length = len(val_list[0])
        output_matrix = np.zeros((len(functions), val_length), dtype=object)
        eco_bundle = helpers.load(params_name, 'data', 'sweep', cluster_kw=cluster_kw, data_label='time_series',
                                  data_ext=data_ext)

        name_list = tuple(sweep[0] for sweep in sweeps)
        entry_list = tuple(sweep[1] for sweep in sweeps)
        val_list = tuple(sweep[-1] for sweep in sweeps)

        t0 = params_orig['t0']
        t_final = params_orig['num_days'] - 1.
        t_save = np.array(np.linspace(t0, t_final, params_orig['num_days']))

        val_length = len(val_list[0])

        for j in range(val_length):

            eco = np.squeeze(eco_bundle[j])
            fn_count = 0

            params = copy.deepcopy(params_orig)

            for i in range(len(val_list)):
                name = name_list[i]
                entries = entry_list[i]
                val = val_list[i][j]

                # does name end with _scale? Then scale entries by value
                if name.endswith('_scale'):
                    actual_name = '_'.join(name.split('_')[:-1])
                    orig_key = '{}_orig'.format(actual_name)
                    if orig_key not in params['bio']:
                        params['bio'][orig_key] = copy.deepcopy(params['bio'][actual_name])

                    if entries is None:
                        params['bio'][actual_name] = val * params['bio'][orig_key]
                    else:
                        for entry in entries:
                            if isinstance(entry, (np.ndarray, tuple, list)):
                                params['bio'][actual_name][tuple(entry)] = val * params['bio'][orig_key][tuple(entry)]
                            else:
                                params['bio'][actual_name][entry] = val * params['bio'][orig_key][entry]

                    print('{}: {}'.format(actual_name, val))

                else:

                    print('{}: {}'.format(name, val))

                    if entries is None:
                        params['bio'][name] = val
                    else:
                        for entry in entries:
                            if isinstance(entry, (np.ndarray, tuple, list)):
                                params['bio'][name][tuple(entry)] = val
                            else:
                                params['bio'][name][entry] = val

            for output in functions:

                output_name = output[0]
                output_fn = output[1]
                output_kw = output[2]

                if (eco == c.NAN_VALUE).any():

                    # store as nans with same dimensions as output should be
                    fake_output = helpers.build_mock_eco_output(t_save, params)

                    output_matrix[fn_count, j] = np.full(np.shape(output_fn(fake_output, t_save, params, output_kw)),
                                                         c.NAN_VALUE).tolist()
                else:
                    output_matrix[fn_count, j] = output_fn(eco, t_save, params, output_kw)

                # save if this is our last go
                if j == val_length - 1:
                    helpers.save(params_name, 'data', 'sweep', output=output_matrix[fn_count],
                                 data_label=output_name, functions=functions, cluster_kw=cluster_kw,
                                 data_ext=data_ext)

                fn_count += 1

    if run_type == 'sweep':

        num_clusters = cluster_kw.get('num_clusters')

        if 'cluster' in cluster_kw:
            current_pp = helpers.load(params_name, 'params', 'sweep', cluster_kw=cluster_kw)
            apply_to_cluster(current_pp)
        else:
            for k in range(num_clusters):
                # load params
                cluster_kw.update(**{'cluster': k})
                current_pp = helpers.load(params_name, 'params', 'sweep', cluster_kw=cluster_kw)
                apply_to_cluster(current_pp)
    elif run_type == 'single':
        current_pp = helpers.load(params_name, 'params', 'single')
        apply_single(current_pp)


def time_series(eco: np.ndarray, params: Dict, num_years: int=None, num_days: int=None, compartments: List[Dict]=None,
                kind: str='indiv') -> np.ndarray:
    """Useful for saving data

    Parameters
    ----------
    eco
        Ecosystem output
    kind
        'indiv', 'total' or 'shannon'
    t
        Times corresponding to each column of output
    params
        Dict of parameters
    num_years
        How many years do we want to include?
    num_days
        How many days?
    compartments
        List of compartment dictionaries

    Returns
    ---------
    numpy.ndarray
        The time series

    """

    output = copy.deepcopy(eco)

    if num_years is not None:
        output = helpers.get_last_n_years(output, num_years)
    if num_days is not None:
        output = helpers.get_last_n_days(output, num_days)

    if kind == 'total':
        return np.array([get_total(output, params, compartments=compartments)])

    if kind == 'shannon':
        return np.array([shannon_ent_rel(output, params)])

    if compartments is not None:
        return np.array([output[helpers.all_compartment_indices(params, compartments=compartments), :].tolist()])

    return np.array([output.tolist()])


def res_to_carbon_conversions(eco: np.ndarray,
                              params: Dict, num_years: int=None, num_days: int=None, phy_index: int=None,
                              return_factor: bool=False) -> np.ndarray:
    """Convert from inorganic nutrient units to carbon units, as a function of current population

    Parameters
    ----------
    eco
        ecosystem
    params
        Dict of parameters
    num_years
        Last n years
    return_factor
        return just the factor.
    num_days
        Last n days
    phy_index
        Which phytoplankton coefficient to use for conversion? If None, compute factor from all present species

    Returns
    ---------
    numpy.ndarray
        Conversion factor for each nutrient

    """

    res_phy_stoich_ratio = params.get('bio').get('res_phy_stoich_ratio')
    mat = np.array(res_phy_stoich_ratio)
    mat[mat == 0] = np.nan
    res = helpers.restrict_ts(eco, params, compartments=[{'res': 'all'}], num_years=num_years, num_days=num_days)
    if phy_index is None:
        phy = helpers.restrict_ts(eco, params, compartments=[{'phy': 'all'}], num_years=num_years, num_days=num_days)
        phy_total = helpers.restrict_ts(eco, params, compartments=[{'phy': 'all'}], num_years=num_years,
                                        num_days=num_days, kind='total')
        factor = phy_total[None, :] / (np.sum(mat[:, :, None] * phy[None, :, :], axis=1) + 1e-10)
        converted_res = factor * res
    else:
        factor = np.ones_like(res) / mat[:, phy_index, None]
        converted_res = factor * res
    if return_factor:
        return factor
    return converted_res


def stoich_from_phy(phy: np.ndarray, params: dict):
    res_phy_stoich_ratio = params.get('bio').get('res_phy_stoich_ratio')
    phy_total = np.sum(phy)
    factor = np.sum(res_phy_stoich_ratio * phy[None, :], axis=1) / phy_total

    return factor


def coeff_of_variation(eco: np.ndarray, params: Dict, kind: str='indiv',
                       compartments: List[Dict]=None, num_years: int=None, num_days: int=None) -> np.ndarray:
    """Compute coefficient of variation

    Parameters
    ----------
    eco
        Ecosystem output
    kind
        'indiv', 'total' or 'shannon'
    params
        Dict of parameters
    num_years
        How many years do we want to include?
    num_days
        How many days do we want to include?
    compartments
        List of compartment dictionaries

    Returns
    ---------
    numpy.ndarray
        Coefficient of variation

    """

    # first, average
    avg = average_value(eco, params, kind, compartments=compartments, num_years=num_years, num_days=num_days)
    var = variance(eco, params, kind, compartments=compartments, num_years=num_years, num_days=num_days)

    # return avg / std
    coeff = list()
    for i in range(len(avg)):
        if np.isclose(avg[i], 0):
            coeff.append(c.NAN_VALUE)
        else:
            coeff.append(np.divide(np.sqrt(var[i]), avg[i]))
    return np.array(coeff)


def variance(eco: np.ndarray, params: Dict,
             kind: str='indiv', compartments: List[Dict]=None, num_years: int=None, num_days: int=None) -> np.ndarray:
    """Compute variance

    Parameters
    ----------
    eco
        Ecosystem output
    kind
        'indiv', 'total' or 'shannon'
    params
        Dict of parameters
    num_years
        How many years do we want to include?
    num_days
        How many days do we want to include?
    compartments
        List of compartment dictionaries

    Returns
    ---------
    numpy.ndarray
        Variance

    """

    if kind == 'shannon':
        return np.array([np.var(shannon_ent_rel(eco, params, num_years=num_years, num_days=num_days))])

    if kind == 'total':
        return np.array([np.var(get_total(eco, params, compartments=compartments, num_years=num_years,
                                          num_days=num_days))])

    eco = helpers.restrict_ts(eco, params, compartments=compartments, num_years=num_years, num_days=num_days)
    return np.apply_along_axis(np.var, 1, eco)


def max_value(eco: np.ndarray, params: Dict,
              kind: str='indiv', compartments: List[Dict]=None, num_years: int=None,
              num_days: int=None, mean_amp_thresh: int=0) -> np.ndarray:
    """Compute max value in each time series

    Parameters
    ----------
    eco
        Ecosystem output
    kind
        'indiv', 'total' or 'shannon'
    params
        Dict of parameters
    num_years
        How many years do we want to include?
    compartments
        List of compartment dictionaries
    mean_amp_thresh
        Return "dead value" if amplitude does not exceed threshold

    Returns
    ---------
    numpy.ndarray
        Average value

    """

    if kind == 'shannon':
        phy_indices = compartments[0]['phy'] if compartments is not None else None
        shan = shannon_ent_rel(eco, params, num_years=num_years, num_days=num_days, phy_indices=phy_indices)

        # if average value of total exceeds threshold, return shannon entropy
        # otherwise shannon entropy has no meaning
        total = get_total(eco, params, compartments=compartments, num_years=num_years, num_days=num_days)
        helpers.eco_indices('phy', params=params)

        if np.mean(total) > mean_amp_thresh:
            return np.array([np.amax(shan)])
        return np.array([c.NAN_VALUE])

    if kind == 'total':
        return np.array([np.amax(get_total(eco, params, compartments=compartments, num_years=num_years,
                                           num_days=num_days))])

    eco = helpers.restrict_ts(eco, params, compartments=compartments, num_years=num_years, num_days=num_days)

    return np.apply_along_axis(np.amax, 1, eco)


# find average
def average_value(eco: np.ndarray, params: Dict,
                  kind: str='indiv', compartments: List[Dict]=None, num_years: int=None, num_days: int=None,
                  mean_amp_thresh: int=0) -> np.ndarray:
    """Compute average value

    Parameters
    ----------
    eco
        Ecosystem output
    kind
        'indiv', 'total' or 'shannon'
    params
        Dict of parameters
    num_years
        How many years do we want to include?
    num_days
        How many days to include?
    compartments
        List of compartment dictionaries
    mean_amp_thresh
        Return "dead value" if amplitude does not exceed threshold

    Returns
    ---------
    numpy.ndarray
        Average value

    """

    if kind == 'shannon':
        phy_indices = compartments[0]['phy'] if compartments is not None else None
        shan = shannon_ent_rel(eco, params, num_years=num_years, num_days=num_days, phy_indices=phy_indices)

        # if average value of total exceeds threshold, return shannon entropy
        # otherwise shannon entropy has no meaning
        total = get_total(eco, params, compartments=compartments, num_years=num_years, num_days=num_days)
        helpers.eco_indices('phy', params=params)

        if np.mean(total) > mean_amp_thresh:
            return np.array([np.mean(shan)])
        return np.array([c.NAN_VALUE])

    if kind == 'total':
        return np.array([np.mean(get_total(eco, params, compartments=compartments, num_years=num_years,
                                           num_days=num_days))])

    eco = helpers.restrict_ts(eco, params, compartments=compartments, num_years=num_years, num_days=num_days)

    return np.apply_along_axis(np.mean, 1, eco)


def is_a_survivor(eco: np.ndarray, params: Dict,
                  abs_amp_thresh: float=0, rel_amp_thresh: float=None, num_years: int=None,
                  num_days: int=None) -> np.ndarray:

    """Does a given species survive (relative to a threshold)?

    Parameters
    ----------
    eco
        Ecosystem output
    params
        Dict of parameters
    abs_amp_thresh
        if max amp less than this, all species have died
    rel_amp_thresh
        If average amplitude for a species greater than `rel_amp_thresh` times the maximum average amplitude, then
        the species survives.
        If not specified, use `abs_amp_thresh`
    num_years
        How many years do we want to include?
    num_days
        How many days to include?
    compartments
        List of compartment dictionaries

    Returns
    ---------
    numpy.ndarray
        Returns 1 if the species survived, 0 if it died

    """

    compartments = [{'phy': 'all'}]

    # get eco for phytoplankton only
    eco = eco[helpers.all_compartment_indices(params, compartments=compartments), :]

    num_phy = params['bio']['num_phy']
    survived = np.zeros(num_phy,)

    max_amp = np.amax(average_value(eco, params, compartments=compartments, num_years=num_years, num_days=num_days))

    if num_years is not None:
        eco = helpers.get_last_n_years(eco, num_years)
    if num_days is not None:
        eco = helpers.get_last_n_days(eco, num_days)

    if max_amp > abs_amp_thresh:
        for i in range(num_phy):
            avg = np.mean(eco[i, :])
            if rel_amp_thresh is not None:
                if avg / max_amp > rel_amp_thresh:
                    survived[i] = 1
            else:
                if avg > abs_amp_thresh:
                    survived[i] = 1

    return survived


def shannon_ent_rel(eco_in: np.ndarray, params: Dict, num_years: int=None, num_days: int=None,
                    phy_indices: tuple=None) -> np.ndarray:
    """Relative shannon entropy

    Parameters
    ----------
    eco_in
        Ecosystem output
    params
        Dict of parameters
    num_years
        How many years do we want to include?
    num_days
        How many days do we want to include?
    phy_indices
        Which indices to include? By default, all phytoplankton (but for example, can split gleaners and opportunists)

    Returns
    ---------
    numpy.ndarray
        Shannon entropy, divided by log base 2 of total number of phytoplankton. Ranges between 0 and 1
    """

    num_phy = params['bio']['num_phy']
    if phy_indices is None:
        phy_indices = np.linspace(0, num_phy-1, num_phy)
    else:
        num_phy = len(phy_indices)

    return shannon_ent(eco_in, params, num_years=num_years, num_days=num_days,
                       phy_indices=phy_indices) / np.log2(num_phy)


def shannon_ent(eco_in: np.ndarray, params: Dict, num_years: int=None, num_days: int=None,
                phy_indices: tuple=None) -> np.ndarray:
    """Shannon entropy, which describes the 'evenness' of the distribution of species across compartments

    Parameters
    ----------
    eco_in
        Ecosystem output
    params
        Dict of parameters
    num_years
        How many years do we want to include?
    num_days
        How many days do we want to include?
    phy_indices
        Which indices to include? By default, all phytoplankton (but for example, can split gleaners and opportunists)

    Returns
    ---------
    numpy.ndarray
        Shannon entropy
    """

    eco = helpers.restrict_ts(eco_in, params, num_years=num_years, num_days=num_days)

    # sometimes very small negative values will pop up, get rid of them
    eco[eco <= 0] = np.finfo(float).eps

    if num_days is None:
        num_days = eco.shape[-1]

    num_phy = params['bio']['num_phy']

    if phy_indices is None:
        phy_indices = tuple([int(x) for x in np.linspace(0, num_phy-1, num_phy)])
    else:
        phy_indices = [int(x) for x in phy_indices]
        num_phy = len(phy_indices)

    # Compute phytoplankton fractions

    global_ind = np.array(helpers.eco_indices('phy', params=params))[phy_indices]
    phyto_total = np.sum(eco[global_ind, :], 0)
    phyto_fracs = np.zeros((num_phy, num_days))

    for i in range(0, num_phy):
        phyto_fracs[i, :] = np.divide(eco[global_ind[i], :], phyto_total)

    # Set values that are zero to machine epsilon
    # (otherwise we will get an error with the Shannon entropy)
    phyto_fracs[phyto_fracs == 0] = np.finfo(float).eps

    return -np.sum(np.multiply(phyto_fracs, np.log2(phyto_fracs)), 0)


def num_unique_vals(input_list: Union[np.ndarray, List], tol: float=1e-2) -> int:

    """List of unique values in a list, using a tolerance to filter out close values

    Parameters
    ----------
    input_list
        List of inputs
    tol
        If two values are within this tol, one is removed

    Returns
    ---------
    int
        Number of unique values
    """

    return len(unique_vals(input_list, tol=tol))


def unique_vals(input_list: Union[np.ndarray, List], tol: float=1e-2) -> np.ndarray:
    """Unique values in a list, using a tolerance to filter out close values

    Parameters
    ----------
    input_list
        List of inputs
    tol
        If two values are within this tol, one is removed

    Returns
    ---------
    numpy.ndarray
        List of unique values
    """

    # go through list and remove duplicate values (within tolerance)
    the_list = list(input_list.tolist() if isinstance(input_list, np.ndarray) else input_list)  # make a copy
    old_count = len(the_list)
    curr_count = 0

    output_list = list()

    if len(the_list) == 1:
        return np.array(the_list)

    if len(the_list) == 2:
        return np.array([the_list[0]] if np.abs(the_list[0]-the_list[1]) < tol else the_list)

    # if we're here, 3 or more values
    while old_count != curr_count:
        old_count = curr_count
        left_position = 0  # start from beginning
        while left_position < len(the_list) - 1:  # proceed until we're at the end
            right_position = left_position + 1
            while right_position < len(the_list):
                if np.abs(the_list[right_position] - the_list[left_position]) < tol:
                    del the_list[right_position]
                else:
                    right_position += 1
            output_list.append(the_list[left_position])
            left_position += 1
        curr_count = len(the_list)

    return np.unique(output_list)


def filtered_spectrum_simple(ts: np.ndarray, dt: float=1.0, thres: float=0, sort: str='amps',
                             subtract_mean: bool=False, normalize: bool=False) \
        -> Tuple[np.ndarray, np.ndarray]:
    """Calculate the Fourier spectrum of a simple time series (filtered by a frequency threshold)

    Parameters
    ----------
    ts
        time series
    dt
        what is the time interval of data recording? (probably 1 day)
    sort
        'freqs': sort by frequency (ascending)
        'amps': sort by amplitude (descending)
    subtract_mean
        Remove mean from signal before computing transform?
    normalize
        Scale by maximum amplitude?
    thres
        Include frequencies above this threshold

    Returns
    ---------
    Tuple[np.ndarray, np.ndarray]
        frequencies and amplitudes in tuple
    """

    freqs, amps = freq_spectrum_simple(ts, dt, sort=sort, subtract_mean=subtract_mean, normalize=normalize)

    amps_filter = np.array(amps[np.where(freqs >= thres)[0]].tolist())
    freqs_filter = np.array(freqs[np.where(freqs >= thres)[0]].tolist())

    return freqs_filter, amps_filter


def get_total(eco: np.ndarray, params: Dict, compartments: List[Dict]=None, num_years: int=None,
              num_days: int=None) -> np.ndarray:
    """Biomass for carbon-based compartments (phy or zoo)

    Parameters
    ----------
    eco
        Ecosystem output
    params
        Dict of parameters
    num_years
        How many years do we want to include?
    num_days
        How many days?
    compartments
        List of compartment dictionaries

    Returns
    ---------
    numpy.ndarray
        Total amount of matter, by summing over all rows
    """

    eco = helpers.restrict_ts(eco, params, num_years=num_years, num_days=num_days)

    total = np.zeros(eco.shape[-1], )

    if compartments is None:
        compartments = [{'phy': 'all'}]

    for compartment in compartments:
        indices = helpers.all_compartment_indices(params, compartments=[compartment])
        output = eco[indices, :]

        total += np.sum(output, 0)

    return total


def filtered_spectrum(eco: np.ndarray, params: Dict,
                      kind: str='indiv', num_years: int=None, num_days: int=None,
                      compartments: List[Dict]=None,
                      subtract_mean: bool=False, sort: str='amps',
                      normalize: bool=False, thres: float=0) -> \
        Tuple[np.ndarray, np.ndarray]:

    """Fourier spectrum of ecosystem, filtered by a frequency threshold

    Parameters
    ----------
    eco
        Ecosystem output
    kind
        'indiv', 'shannon' or 'total'
    params
        Dict of parameters
    num_years
        How many years do we want to include?
    num_days
        Alternate to num_years
    compartments
        List of compartment dictionaries
    subtract_mean
        Remove mean from signal before computing transform?
    normalize
        Scale by maximum amplitude?
    thres
        frequency threshold
    sort
        'freqs': sort by frequency (ascending)
        'amps': sort by amplitude (descending)

    Returns
    ---------
    Tuple[numpy.ndarray, numpy.ndarray]
        Frequencies and amplitudes from Fourier transform
    """

    freqs, amps = freq_spectrum(eco, params, kind=kind, num_years=num_years, num_days=num_days, sort=sort,
                                compartments=compartments, subtract_mean=subtract_mean, normalize=normalize)

    # if it's just a vector...
    if len(np.shape(freqs)) == 1:
        amps_filter = amps[np.where(freqs >= thres)[0]].tolist()
        freqs_filter = freqs[np.where(freqs >= thres)[0]].tolist()

    # otherwise it's a matrix, go row by row
    else:
        amps_filter = list()
        freqs_filter = list()

        for i in range(len(freqs[:, 0])):
            amps_filter.append(amps[i, np.where(freqs[i, :] >= thres)[0]].tolist())
            freqs_filter.append(freqs[i, np.where(freqs[i, :] >= thres)[0]].tolist())

    return np.array(freqs_filter), np.array(amps_filter)


def freq_spectrum_simple(ts: np.ndarray, dt: float=1.0, sort: str='amps',
                         normalize: bool=False, subtract_mean: bool=False) \
        -> Tuple[np.ndarray, np.ndarray]:
    """Calculate the Fourier spectrum of a simple time series

    Parameters
    ----------
    ts
        time series
    subtract_mean
        remove mean?
    normalize
        scale by maximum?
    dt
        what is the time interval of data recording? (probably 1 day)
    sort
        'freqs': sort by frequency (ascending)
        'amps': sort by amplitude (descending)

    Returns
    ---------
    Tuple[np.ndarray, np.ndarray]
        frequencies and amplitudes in tuple
    """

    # get rid of 0 frequency mode
    if subtract_mean:
        ts -= np.mean(ts)

    amps = 1 / ts.size * np.abs(np.fft.fft(ts))
    freqs = np.fft.fftfreq(ts.size, dt)

    if sort == 'amps':
        ind = amps.argsort()[::-1]
        freqs = freqs[ind]
        amps = amps[ind]
    if sort == 'freqs':
        ind = freqs.argsort()
        freqs = freqs[ind]
        amps = amps[ind]

    if normalize and np.amax(amps) > 0:
        amps = amps / np.amax(amps)

    return freqs, amps


def freq_spectrum(eco: np.ndarray, params: Dict, kind: str='indiv', num_years: int=None, num_days: int=None,
                  compartments: List[Dict]=None, subtract_mean: bool=False, normalize: bool=False,
                  sort: str='amps') -> Tuple[np.ndarray, np.ndarray]:
    """Fourier spectrum of ecosystem

    Parameters
    ----------
    eco
        Ecosystem output
    params
        Dict of parameters
    kind
        'indiv', 'shannon' or 'total'
    num_years
        How many years do we want to include?
    num_days
        How many days?
    sort
        'freqs': sort by frequency (ascending)
        'amps': sort by amplitude (descending)
    compartments
        List of compartment dictionaries
    subtract_mean
        Remove mean from signal before computing transform?
    normalize
        Scale by maximum amplitude?

    Returns
    ---------
    Tuple[numpy.ndarray, numpy.ndarray]
        Frequencies and amplitudes from Fourier transform
    """

    if kind == 'shannon':

        # Compute relative Shannon entropy
        shannon_end = shannon_ent_rel(eco, params, num_years=num_years, num_days=num_days)

        freqs, amps = freq_spectrum_simple(shannon_end, params['dt_save'], sort=sort,
                                           subtract_mean=subtract_mean, normalize=normalize)

        return np.array([freqs]), np.array([amps])

    if compartments is None:
        compartments = [{'phy': 'all'}]

    indices = helpers.all_compartment_indices(params, compartments=compartments)

    eco_end = helpers.restrict_ts(eco, params, compartments=compartments, num_years=num_years, num_days=num_days)
    num_days_end = eco_end.shape[-1]

    if kind == 'total':
        eco_end = np.reshape(get_total(eco, params, compartments=compartments, num_years=num_years, num_days=num_days)
                             , (1, num_days_end))
        indices = [0]

    if len(np.shape(eco_end)) == 1:
        eco_end = np.reshape(eco_end, (1, num_days_end))
        indices = [0]

    num_compartments = len(indices)

    amps = np.zeros((num_compartments, num_days_end))
    freqs = np.zeros((num_compartments, num_days_end))

    # Amps/freqs of compartments
    for i in range(0, num_compartments):
        current_series = np.array(eco_end[i, :])

        freqs[i, :], amps[i, :] = freq_spectrum_simple(current_series, params['dt_save'], sort=sort,
                                                       subtract_mean=subtract_mean,
                                                       normalize=normalize)

    return freqs, amps


def largest_phy_freq_count(eco: np.ndarray, params: Dict, spectrum_kw: Dict=None, abs_amp_thresh: float=0.1,
                           rel_amp_thresh: float=1e-2, num_years: int=None, num_days: int=None) -> int:
    """Frequency count for most abundant species in the ecosystem

    Parameters
    ----------
    eco
        Ecosystem output
    params
        Dict of parameters
    spectrum_kw
        keyword arguments for Fourier spectrum calculation
    abs_amp_thresh
        threshold for a species to be considered existing
    rel_amp_thresh
        cutoff threshold as a fraction of max amplitude achieved by time series
    num_years
        How many years do we want to include?
    num_days
        How many days?

    Returns
    ---------
    int
        Number of frequencies
    """

    index = None
    max_val = -99999
    for i in range(params['bio']['num_phy']):
        the_mean = np.mean(eco[i, :])
        if the_mean > max_val:
            max_val = the_mean
            index = i

    return freq_count(eco, params, num_years=num_years, num_days=num_days,
                      compartments=[{'phy': [index]}],
                      spectrum_kw=spectrum_kw, abs_amp_thresh=abs_amp_thresh, rel_amp_thresh=rel_amp_thresh)[0]


def freq_count(eco: np.ndarray, params: Dict, kind: str='indiv', spectrum_kw: Dict=None, abs_amp_thresh: float=0.0,
               num_days: int=None, power_spectrum: bool=False,
               rel_amp_thresh: float=1e-2, num_years: int=None, compartments: List[Dict]=None,
               skip_interval: int=1) \
        -> np.ndarray:
    """Frequency count for most abundant species in the ecosystem

    Parameters
    ----------
    eco
        Ecosystem output
    params
        Dict of parameters
    compartments
        list of compartment dictionaries
    kind
        'indiv', 'total' or 'shannon'
    spectrum_kw
        keyword arguments for Fourier spectrum calculation
    abs_amp_thresh
        threshold for a species to be considered existing
    skip_interval
        for sampling frequencies
    rel_amp_thresh
        cutoff threshold as a fraction of max amplitude achieved by time series
    power_spectrum
        measure amplitudes of power spectrum, rather than Fourier spectrum? (just square amplitudes)
    num_years
        How many years do we want to include?
    num_days
        How many days?

    Returns
    ---------
    np.ndarray
        Frequency counts for whatever compartments we want
    """

    _, amps = filtered_spectrum(eco, params, **spectrum_kw if spectrum_kw is not None else {},
                                kind=kind, num_years=num_years, num_days=num_days, compartments=compartments)

    amps = np.array(amps)

    if power_spectrum:
        amps = np.power(np.abs(amps), 2)

    if len(np.shape(amps)) == 1:
        amps = np.reshape(amps, (1, len(amps)))

    s = np.shape(amps)

    counts = np.zeros(s[0],)
    for i in range(s[0]):
        max_amp = np.amax(amps[i, ::skip_interval])

        if max_amp < abs_amp_thresh:
            counts[i] = 0
        else:
            locs = np.where(np.divide(amps[i, ::skip_interval], max_amp) > rel_amp_thresh)[0]
            counts[i] = len(amps[i, locs])

    return counts


def adjacent_freq_count(eco: np.ndarray, params: Dict, kind: str='indiv', spectrum_kw: Dict=None,
                        abs_amp_thresh: float=0.0, num_days: int=None, power_spectrum: bool=False,
                        rel_amp_thresh: float=1e-2, num_years: int=None, compartments: List[Dict]=None,
                        segment_frac: float=0.1) -> np.ndarray:
    """Frequency count for most abundant species in the ecosystem

    Parameters
    ----------
    eco
        Ecosystem output
    params
        Dict of parameters
    compartments
        list of compartment dictionaries
    kind
        'indiv', 'total' or 'shannon'
    spectrum_kw
        keyword arguments for Fourier spectrum calculation
    abs_amp_thresh
        threshold for a species to be considered existing
    segment_frac
        length of window for sampling, relative to length of whole interval, on log scale
    rel_amp_thresh
        cutoff threshold as a fraction of max amplitude achieved by time series
    power_spectrum
        measure amplitudes of power spectrum, rather than Fourier spectrum? (just square amplitudes)
    num_years
        How many years do we want to include?
    num_days
        How many days?

    Returns
    ---------
    np.ndarray
        Frequency counts for whatever compartments we want
    """

    freqs, amps = filtered_spectrum(eco, params, **spectrum_kw if spectrum_kw is not None else {},
                                kind=kind, num_years=num_years, num_days=num_days, compartments=compartments)

    amps = np.array(amps)

    if power_spectrum:
        amps = np.power(np.abs(amps), 2)

    if len(np.shape(amps)) == 1:
        amps = np.reshape(amps, (1, len(amps)))

    s = np.shape(amps)
    window_length = int(segment_frac * len(freqs[0, :]))
    current_window = np.array(list(range(window_length))).astype(int)

    counts = np.zeros(s[0],)
    for i in range(s[0]):
        max_amp = np.amax(amps[i, :])
        if max_amp < abs_amp_thresh:
            counts[i] = 0
        else:
            max_count = -np.inf
            while current_window[-1] < s[-1]:
                locs = np.where(np.divide(amps[i, current_window], max_amp) > rel_amp_thresh)[0]
                if len(locs) > max_count:
                    max_count = len(locs)
                    print(len(locs)/(1. * window_length))
                current_window += 1
            counts[i] = max_count
    return counts


def dom_freq(eco: np.ndarray, params: Dict, spectrum_kw: Dict=None, mean_amp_thresh: float=0, kind: str='indiv',
             rel_amp_thresh: float=0, num_days: int=None, freq_interval: tuple=(0, np.inf),
             num_years: int = None, compartments: List[Dict] = None) -> np.ndarray:
    """Dominant frequency for desired ecosystem compartments
    A value of 0 for a subtract_meaned (mean subtracted) time series ==> the species achieved equilibrium.
    A value of c.DEAD_VALUE means that the species died.

    Parameters
    ----------
    eco
        Ecosystem output
    kind
        'indiv', 'shannon', 'total'
    params
        Dict of parameters
    compartments
        list of compartment dictionaries
    spectrum_kw
        keyword arguments for Fourier spectrum calculation
    mean_amp_thresh
        threshold for a species to be considered existing
    rel_amp_thresh
        relative threshold for second largest frequency to be considered "large" compared to mean freq
    freq_interval
        interval of frequencies to look at when computing dominant frequency
    num_years
        How many years do we want to include?
    num_days
        How many days do we want to include?

    Returns
    ---------
    np.ndarray
        Dominant frequencies for our desired compartments
    """

    # make sure spectrum_kw['sort'] = 'amps' if it isn't already
    kw = dict(spectrum_kw) if spectrum_kw is not None else dict()
    kw['sort'] = 'amps'

    freqs, amps = filtered_spectrum(eco, params, **kw, kind=kind,
                                    num_years=num_years, compartments=compartments, num_days=num_days)

    amps = np.array(amps)
    freqs = np.array(freqs)

    # get means
    means = average_value(eco, params, kind=kind, compartments=compartments, num_years=num_years, num_days=num_days)

    # take mean of total (we need this for shannon entropy)
    means_total = average_value(eco, params, kind='total', compartments=compartments, num_years=num_years,
                                num_days=num_days)[0]

    s = np.shape(amps)

    if len(s) == 1:
        s = (1, s[0])
        freqs = np.reshape(freqs, s)

    dom_freqs = np.zeros(s[0],)

    if kind == 'shannon' and not (means_total > mean_amp_thresh):
        return np.array([c.NAN_VALUE])

    for i in range(s[0]):

        good_locs = np.where((freqs[i, :] >= freq_interval[0]) & (freqs[i, :] <= freq_interval[-1]))[0]
        freqs_good = freqs[i, good_locs]
        amps_good = amps[i, good_locs]

        if means[i] > mean_amp_thresh:
            if kw.get('subtract_mean'):
                if amps_good[i] / means[i] > rel_amp_thresh:
                    dom_freqs[i] = freqs_good[i]
                else:
                    dom_freqs[i] = 0
            else:
                dom_freqs[i] = freqs_good[i]
        else:
            dom_freqs[i] = c.NAN_VALUE

    return dom_freqs


def average_cycle_simple(x: np.ndarray, period: float=c.NUM_DAYS_PER_YEAR, subtract: bool=False,
                         truncate: bool=False) -> np.ndarray:
    """Average cycle of period T (in days) from time series

        Parameters
        ----------
        x
            1D input
        period
            period of cycle (in days)
        subtract
            if true, return time series with average cycle subtracted. if false, just return average cycle
        truncate
            only returns vector up to length of period.

        Returns
        ---------
        np.ndarray
            The average cycle computed from TS. Length is same as the time series if truncate is False
        """

    out_dict = {index: list() for index in range(int(period))}

    for j in range(len(x)):
        index = j % period
        out_dict[index].append(x[j])

    avg = np.array([np.mean(out_dict[i]) for i in range(int(period))])

    if truncate:
        last_index = int(period)
    else:
        last_index = len(x)

    the_output = np.zeros(np.shape(x[:last_index]))

    curr_index = 0
    while last_index > curr_index:
        the_range = range(int(curr_index), int(curr_index + period))
        the_output[the_range] = x[the_range] - avg if subtract else avg
        curr_index += period

    return the_output


# now for the ecosystem
def average_cycle(eco: np.ndarray, params: Dict, kind: str='indiv', period: float=c.NUM_DAYS_PER_YEAR,
                  num_years: int=None, num_days: int=None, subtract: bool=False, truncate: bool=False,
                  compartments: List[Dict]=None) -> np.ndarray:
    """Average cycle of period T (in days) from ecosystem

        Parameters
        ----------
        eco
            1D input
        params
            Dict of params
        kind
            'indiv', 'shannon' or 'total'
        num_years
            How many years are we using?
        num_days
            How many days?
        compartments
            Which compartments are we computing this for?
        period
            period of cycle
        subtract
            if true, return time series with average cycle subtracted. if false, just return average cycle
        truncate
            only returns vector up to length of period.

        Returns
        ---------
        np.ndarray
            The average cycle computed from TS. Length is same as the time series if truncate is False
        """

    if compartments is None:
        compartments = [{'phy': 'all'}]

    eco_end = helpers.restrict_ts(eco, params, compartments=compartments, num_years=num_years, num_days=num_days,
                                  kind=kind)

    if kind == 'shannon':
        # Compute relative Shannon entropy
        shannon_end = shannon_ent_rel(eco, params, num_years=num_years, num_days=num_days)
        return np.array([average_cycle_simple(shannon_end, period=period, subtract=subtract, truncate=truncate)])

    indices = helpers.all_compartment_indices(params, compartments=compartments)

    num_days_end = eco_end.shape[-1]

    if kind == 'total':
        eco_end = np.reshape(get_total(eco, params, compartments=compartments, num_days=num_days_end),
                             (1, num_days_end))
        indices = [0]

    if len(np.shape(eco_end)) == 1:
        eco_end = np.reshape(eco_end, (1, num_days_end))
        indices = [0]

    num_compartments = len(indices)

    output = np.zeros((num_compartments, num_days_end))
    for i in range(num_compartments):
        current_series = np.array(eco_end[i, :])
        output[i, :] = average_cycle_simple(current_series, period=period, subtract=subtract)

    return output

