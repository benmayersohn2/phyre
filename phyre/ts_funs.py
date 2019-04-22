"""
These functions MUST return output as standard Python lists, not numpy arrays.
Parameters to function must be: (eco, t, params, kw_dict)

Notes
----------
Most of these methods do not require time, but some do (e.g. time series). Underscore notation is
used in this case.

Most of the methods are from the `analysis` package. See `analysis.py` for function documentation
"""

from typing import Dict, List, Callable
import phyre.analysis.analysis as al
import numpy as np


# convenience function
def f_wrap(func: Callable, eco: np.ndarray, t: np.ndarray, pp: Dict, kw: Dict = None, listify: bool=True,
           time_included: bool=False, **kwargs) -> List:
    if kw is not None:
        out = func(np.array(eco), t, pp, **kw, **kwargs) if time_included else func(np.array(eco), pp, **kw, **kwargs)
    else:
        out = func(np.array(eco), t, pp, **kwargs) if time_included else func(np.array(eco), pp, **kwargs)

    if listify:
        if isinstance(out, np.ndarray):
            return list(out.tolist())
        if isinstance(out, float) or isinstance(out, int):
            return [out]
    return out


def time_series(eco: np.ndarray, t, pp: Dict, kw: Dict=None) -> List:
    return f_wrap(al.time_series, eco, t, pp, kw)


def spectrum_fun(eco: np.ndarray, t, pp: Dict, kw: Dict=None) -> List:
    freqs, amps = f_wrap(al.filtered_spectrum, eco, t, pp, kw, listify=False)

    # structure properly
    spectrum = list()
    freqs = list(freqs.tolist())
    amps = list(amps.tolist())
    for i in range(len(amps)):
        spectrum.append([freqs[i], amps[i]])

    return spectrum


def freq_count(eco: np.ndarray, t, pp: Dict, kw: Dict=None) -> List:
    return f_wrap(al.freq_count, eco, t, pp, kw)


def dom_freq(eco: np.ndarray, t, pp: Dict, kw: Dict=None) -> List:
    return f_wrap(al.dom_freq, eco, t, pp, kw)


def coeff_of_variation(eco: np.ndarray, t, pp: Dict, kw: Dict=None) -> List:
    return f_wrap(al.coeff_of_variation, eco, t, pp, kw)


def is_a_survivor(eco: np.ndarray, t, pp: Dict, kw: Dict=None) -> List:
    return f_wrap(al.is_a_survivor, eco, t, pp, kw)


def average_value(eco: np.ndarray, t, pp: Dict, kw: Dict=None) -> List:
    return f_wrap(al.average_value, eco, t, pp, kw)


def max_value(eco: np.ndarray, t, pp: Dict, kw: Dict=None) -> List:
    return f_wrap(al.max_value, eco, t, pp, kw)


def species_richness(eco: np.ndarray, t, pp: Dict, kw: Dict=None):
    compartments = [{'phy': 'all'}]
    num_years = kw.get('num_years')
    mean_amp_thresh = kw.get('mean_amp_thresh')
    means = al.average_value(eco, pp, kind='indiv', compartments=compartments, num_years=num_years)
    s = np.shape(means)
    richness = 0

    for i in range(s[0]):
        if means[i] >= mean_amp_thresh:
            richness += 1
    return [richness]
