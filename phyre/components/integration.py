"""
integration.py: Time stepping and RHS building
"""

from typing import Dict, Tuple
import numpy as np
from scipy.integrate import ode
from scipy.integrate import odeint
from phyre import constants as c
from phyre import helpers
import phyre.components.bio as bio

#########################################################################


def integrate(eco: np.ndarray, params: Dict, method: str='odeint', data_ext: str='npy', cluster_kw: Dict=None,
              odeint_kw: Dict=None) -> Tuple[np.ndarray, np.ndarray]:
    """Integrate ecosystem using the specified method

    Parameters
    ----------
    eco
        Ecosystem output
    params
        Dict of parameters
    data_ext
        Extension with which to save data. Used only if we rely on a previous run as a forcing
    cluster_kw
        How many clusters, and what cluster are we on? Used only if we rely on a previous run as a forcing
    odeint_kw
        Keywords for ODEINT integrator
    method
        What method to use? Either 'odeint' or 'ode'

    Returns
    ---------
    Tuple[np.ndarray, np.ndarray]
        Return ecosystem and corresponding time series
    """

    # try to integrate, catch warnings about invalid values, treat as bad run

    # raise FloatingPointError if we have any problems (except underflow)
    np.seterr(all='raise')
    np.seterr(under='ignore')

    # time vector
    t0 = params['t0']
    t_final = params['t_final']
    dt = params['dt_save']
    t_save = np.array(np.linspace(t0, t_final - dt, params['num_days']))

    def do_int(ee, pp):
        sol = None
        try:
            if method == 'odeint':
                sol = integrate_odeint(ee, pp, odeint_kw)
            else:
                sol = integrate_ode(ee, pp)

        except FloatingPointError:
            ee = np.full((ee.shape[0], t_save.shape[0]), np.nan)

        if sol is None:
            return np.squeeze(ee), t_save

        return sol

    return do_int(eco, params)


def integrate_ode(eco: np.ndarray, params: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """Integrate using scipy's ODE solver and python loops...SLOW

    Parameters
    ----------
    eco
        Ecosystem output
    params
        Dict of parameters

    Returns
    ---------
    Tuple[np.ndarray, np.ndarray]
        Return ecosystem and corresponding time series
    """
    
    # time vector
    t0 = params['t0']
    t_final = params['t_final']
    dt = params['dt_save']

    num_compartments = params.get('bio').get('num_compartments')

    t_save = np.zeros(1, )
    t_save[0] = t0

    # define rhs function
    def f(tt, eco_in):
        return bio.bio_build(eco_in, tt, params.get('bio'))

    r = ode(f)
    r.set_integrator('lsoda')
    r.set_initial_value(eco, t0)

    eco = np.reshape(eco, (num_compartments, 1))

    while r.successful() and r.t < t_final:
        # Add current solution to eco
        eco = np.concatenate((eco, np.reshape(r.integrate(r.t + dt), (num_compartments, 1))), axis=1)
        t_save = np.append(t_save, r.t + dt)

    # The 'lsoda' integrator may produce repeated times/values, so compute unique values
    (t_save, t_ind) = np.unique(t_save, return_index=True)

    return eco[:, t_ind], t_save


def integrate_odeint(eco: np.ndarray, params: Dict, kwargs: Dict=None) -> Tuple[np.ndarray, np.ndarray]:
    """Integrate using odeint. Currently fastest way to integrate and used by default.

    Parameters
    ----------
    eco
        Ecosystem output
    params
        Dict of parameters
    kwargs
        Keyword arguments for odeint

    Returns
    ---------
    Tuple[np.ndarray, np.ndarray]
        Return ecosystem and corresponding time series
    """

    # time vector
    t0 = params['t0']
    t_final = params['t_final']
    t_save = np.array(np.linspace(t0, t_final, params['num_days']))

    if kwargs is None:
        eco = odeint(bio.bio_build, eco, t_save, args=(params.get('bio'),))
    else:
        eco = odeint(bio.bio_build, eco, t_save, args=(params.get('bio'),), **kwargs)

    return np.transpose(eco), t_save
