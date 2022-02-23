"""
integration.py: Time stepping and RHS building
"""

from typing import Dict, Tuple
import numpy as np
import phyre.components.bio as bio
from scipy.integrate import ode
from scipy.integrate import odeint

#########################################################################


def integrate(eco: np.ndarray, params: Dict, method: str = 'odeint',
              odeint_kw: Dict = None, raise_errors: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Integrate ecosystem using the specified method

    Parameters
    ----------
    eco
        Ecosystem output
    params
        Dict of parameters
    odeint_kw
        Keywords for ODEINT integrator
    method
        What method to use? 'odeint', 'ode', or 'euler' (you must use the latter if noise)
    raise_errors
        Raise errors if integration doesn't work

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
            elif method == 'ode':
                sol = integrate_ode(ee, pp)
        except FloatingPointError as e:
            if raise_errors:
                raise e
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

    bb = params.get('bio')

    num_compartments = bb.get('num_compartments')

    t_save = np.zeros(1, )
    t_save[0] = t0

    if 'noise_sd' in bb:
        del bb['noise_sd']

    # define rhs function
    def f(tt, eco_in):
        return bio.bio_build(eco_in, tt, bb)

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


def integrate_odeint(eco: np.ndarray, params: Dict, kwargs: Dict = None) -> Tuple[np.ndarray, np.ndarray]:
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

    bb = params.get('bio')

    if 'noise_sd' in bb:
        del bb['noise_sd']

    if kwargs is None:
        eco = odeint(bio.bio_build, eco, t_save, args=(bb,))
    else:
        eco = odeint(bio.bio_build, eco, t_save, args=(bb,), **kwargs)

    return np.transpose(eco), t_save
