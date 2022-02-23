"""
integration.py: Time stepping and RHS building
"""

from typing import Dict, Tuple
import numpy as np
import phyre.components.bio as bio
import phyre.components.numerics as numerics
import sdepy
from scipy.integrate import ode
from scipy.integrate import odeint
import sdeint
from scipy.interpolate import interp1d

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
            elif method == 'sdeint':
                sol = integrate_sdeint(ee, pp)
            elif method == 'euler_maruyama':
                sol = integrate_eulermaruyama(ee, pp)
            elif method == 'sdepy':
                sol = integrate_sdepy(ee, pp)
            else:
                sol = integrate_custom(ee, pp, method=method)

        except FloatingPointError as e:
            if raise_errors:
                raise e
            ee = np.full((ee.shape[0], t_save.shape[0]), np.nan)

        if sol is None:
            return np.squeeze(ee), t_save

        return sol

    return do_int(eco, params)


# TODO: Set this up for sdepy library
def integrate_sdepy(eco: np.ndarray, params: Dict) -> Tuple[np.ndarray, np.ndarray]:
    # time vector
    t0 = params['t0']
    t_final = params['t_final']
    bb = params.get('bio')
    num_compartments = bb['num_compartments']

    # dt = 1/10
    steps = 100
    t_save = np.array(np.linspace(t0, t_final, params['num_days']))
    t_int = np.linspace(t0, t_final, steps * params['num_days'])

    bb = params.get('bio')
    noise_sd = bb.pop('noise_sd')

    # Don't need this
    del bb['noise']

    nb = {'noise_sd': noise_sd, 'res_forcing_amps': bb['res_forcing_amps']}
    nb['include_zoo'] = nb.get('include_zoo', False)

    def f(eco_in, tt):
        return bio.bio_build(eco_in, tt, bb)

    def G(eco_in, tt):
        return bio.noise_build(eco_in, tt, nb)

    @sdepy.integrate
    def my_process(t, x):
        return {'dt': f(x, t), 'dw': G(x, t)}

    out = my_process(x0=eco, vshape=num_compartments, steps=steps, paths=1)(t_save)
    print(out.shape)
    return out, t_save

# Simple Euler-Maruyama implementation
def integrate_eulermaruyama(eco: np.ndarray, params: Dict) -> Tuple[np.ndarray, np.ndarray]:
    # time vector
    t0 = params['t0']
    t_final = params['t_final']
    bb = params.get('bio')

    # dt = 1/10
    t_save = np.array(np.linspace(t0, t_final, params['num_days']))

    bb = params.get('bio')
    noise_sd = bb.pop('noise_sd')

    # Don't need this
    del bb['noise']

    nb = {'noise_sd': noise_sd}
    eco_in = eco.copy()
    eco = np.zeros((eco_in.shape[0], len(t_save)))
    eco[:, 0] = eco_in

    # We can vary dt if we need to
    dt = 1

    def dW(dt=dt):
        """Sample a random number at each call."""
        return np.random.normal(loc=0.0, scale=np.sqrt(dt))

    def f(eco_in, tt):
        return bio.bio_build(eco_in, tt, bb)

    def G(eco_in, tt):
        return bio.noise_build(eco_in, tt, nb)

    for key in ('num_phy', 'num_res', 'num_zoo', 'res_forcing_amps', 'include_zoo'):
        nb[key] = bb.get(key)

    for i in range(1, len(t_save)):
        y = eco[:, i-1]
        t = t_save[i]
        eco[:, i] = y + f(y, t) * dt + G(y, t) * dW(dt=dt)
        # print(eco[:, i])

    return eco, t_save


def integrate_sdeint(eco: np.ndarray, params: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """
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

    # sdeint integrators
    # sdeint.itoEuler: the Euler-Maruyama algorithm for Ito equations.
    # sdeint.stratHeun: the Stratonovich Heun algorithm for Stratonovich equations
    # sdeint.itoSRI2: the Roessler2010 order 1.0 strong Stochastic Runge-Kutta algo SRI2 for Ito equations
    # sdeint.stratSRS2: the Roessler2010 order 1.0 strong Stochastic Runge-Kutta algo SRS2 for Stratonovich equations
    # sdeint.stratKP2iS: the Kloeden and Platen two-step implicit order 1.0 strong algo for Stratonovich equations

    # We convert to a Stratonovich equation to use some of the better integrators

    # time vector
    t0 = params['t0']
    t_final = params['t_final']
    bb = params.get('bio')

    t_save = np.linspace(t0, t_final, params['num_days'])

    # Make time steps sufficiently small
    t_eval = np.linspace(t0, t_final, 10 * (params['num_days'] - 1))

    bb = params.get('bio')
    noise_sd = bb.pop('noise_sd')

    # Don't need this
    del bb['noise']

    nb = {'noise_sd': noise_sd}
    num_compartments = len(eco)

    for key in ('num_phy', 'num_res', 'num_zoo', 'res_forcing_amps', 'include_zoo'):
        nb[key] = bb.get(key)

    def f(eco_in, tt):
        return bio.bio_build(eco_in, tt, bb)

    def G(eco_in, tt):
        return bio.noise_build(eco_in, tt, nb).reshape((num_compartments, 1))

    eco = np.transpose(sdeint.itoint(f, G, eco, t_eval))

    # Interpolate eco
    eco_new = np.zeros((len(eco), len(t_save)))
    for i in range(len(eco)):
        eco_new[i, :] = interp1d(t_eval, eco[i, :], kind='cubic')(t_save)
    return eco_new, t_save


# TODO: Finish writing this
def integrate_custom(eco: np.ndarray, params: Dict, method: str = 'implicitRKMil') -> Tuple[np.ndarray, np.ndarray]:
    """
    Parameters
    ----------
    eco
        Ecosystem output
    params
        Dict of parameters
    method
        What method to use for integration? These are customly coded

    Returns
    ---------
    Tuple[np.ndarray, np.ndarray]
        Return ecosystem and corresponding time series
    """

    # time vector
    t0 = params['t0']
    t_final = params['t_final']

    dt = params['num_days']
    t_save = np.array(np.linspace(t0, t_final, dt))

    bb = params.get('bio')
    noise_sd = bb.pop('noise_sd')

    nb = {'noise_sd': noise_sd}

    for key in ('num_phy', 'num_res', 'num_zoo', 'res_forcing_amps', 'include_zoo'):
        nb[key] = bb.get(key)

    def f(eco_in, tt):
        return bio.bio_build(eco_in, tt, bb)

    def g(eco_in, tt):
        return bio.noise_build(eco_in, tt, nb)

    stoch_keys = [x for x in dir(numerics) if x.startswith('s_')]
    det_keys = [x for x in dir(numerics) if x.startswith('d_')]
    all_keys = stoch_keys + det_keys

    if method in all_keys:
        custom_func = eval(f"numerics.{method}")
        if method in stoch_keys:
            del bb['noise']
            eco = custom_func(f, g, noise_sd, eco, t_save)
        else:
            # Noise will be added from vector
            eco = custom_func(f, eco, t_save)
    else:
        raise ValueError(f"Invalid method selected! Must be one of: {', '.join(all_keys)}")

    return eco, t_save


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
