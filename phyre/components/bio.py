"""
bio.py: The biological components used by integrate.py
"""

from typing import Dict, Tuple, Union, Callable
import numpy as np
from phyre import helpers
from phyre import constants as c

########################################################################


def bio_build(eco: np.ndarray, t: int, bio: Dict) -> np.ndarray:
    """Wrapper for rhs_build that is passed to ODE solver

    Parameters
    ----------
    eco
        ecosystem state at time t
    t
        Current time
    bio
        Dictionary of bio parameters, whose contents are to be passed to `rhs_build` as keyword arguments

    Returns
    ----------
    numpy.ndarray
        RHS for further integration
    """
    return rhs_build(eco, t, **bio)


# RHS - assuming noise is being added to turnover rate
def noise_build(eco: np.ndarray, t: int, bb: Dict) -> np.ndarray:
    noise_sd = bb.get('noise_sd')
    res_forcing_amps = bb.get('res_forcing_amps')
    include_zoo = bb.get('include_zoo')

    rhs = np.zeros_like(eco)
    phy_indices = helpers.eco_indices('phy', bio=bb)
    res_indices = helpers.eco_indices('res', bio=bb)
    zoo_indices = helpers.eco_indices('zoo', bio=bb)
    rhs[phy_indices] = -noise_sd * eco[phy_indices]
    rhs[res_indices] = noise_sd * (res_forcing_amps - eco[res_indices])

    if include_zoo in (1, True, None):
        rhs[zoo_indices] = -noise_sd * eco[zoo_indices]

    return rhs


# BUILD RHS
def rhs_build(eco: np.ndarray, t: int,
              num_phy: int,
              num_zoo: int,
              num_res: int,
              res_phy_stoich_ratio: np.ndarray,
              res_phy_remin_frac: np.ndarray,
              phy_growth_sat: np.ndarray,
              res_forcing_amps: np.ndarray,
              num_compartments: int,
              phy_mort_rate: Union[np.ndarray, float],
              phy_growth_rate_max: np.ndarray,
              turnover_rate: float = 0.04,
              zoo_hill_coeff: float = 1,
              phy_self_shade: float = 0,
              shade_background: float = 0,
              turnover_min: float = None,
              turnover_max: float = None,
              turnover_radius: float = None,
              turnover_period: Union[float, np.ndarray] = 360,
              turnover_series: np.ndarray = None,
              turnover_phase_shift: float = 0,
              light_series: np.ndarray = None,
              light_min: float = None,
              light_max: float = None,
              light_kind: str = 'ramp',
              turnover_kind: str = 'sine',
              mixed_layer_ramp_times: np.ndarray = None,
              mixed_layer_ramp_lengths: np.ndarray = None,
              phy_source: float = 0,
              res_source: float = 0,
              zoo_source: float = 0,
              dilute_zoo: bool = True,
              zoo_mort_rate: tuple = None,
              linear_zoo_mort_rate: tuple = None,
              zoo_grazing_rate_max: np.ndarray = None,
              zoo_slop_feed: np.ndarray = None,
              zoo_prey_pref: np.ndarray = None,
              zoo_grazing_sat: np.ndarray = None,
              noise: np.ndarray = None,
              noise_additive: bool = False,
              include_zoo: bool = False,
              res_uptake_type: str = 'perfect',
              zoo_model_type: str = 'Real',
              debug_dict: Dict = None,
              dummy_param: int = 0
              ) -> Union[np.ndarray, Tuple[np.ndarray, Dict]]:
    """Build the RHS of the bio equations

    Parameters
    ----------
    eco
        Current values of ecosystem variables
    t
        Current time
    debug_dict
        Return key-value pairs for RHS terms, to study relative magnitudes
    num_phy
        Number of phytoplankton
    num_zoo
        Number of zooplankton
    num_res
        Number of resources
    res_phy_stoich_ratio
        `num_res` by `num_phy` matrix of stoichiometric coefficients. Convert from carbon to nutrient units
    res_phy_remin_frac
        Converts carbon from phytoplankton into resource units
    turnover_series
        If we want to pass the full turnover series in as a function
    light_series
        If we want to pass the full light series in as a function
    mixed_layer_ramp_times
        If we want to represent mixed layer: at what times do ramps occur?
    mixed_layer_ramp_lengths
        If we want to represent mixed layer: how long are the ramps?
    turnover_rate
        (Constant) Dilution coefficient
    turnover_min
        Minimum dilution coefficient (for time-varying case)
    turnover_max
        Maximum dilution coefficient (for time-varying case)
    turnover_radius
        Fraction of mean for radius: between 0 and 1 (for time-varying case)
    turnover_period
        Period of nutrient delivery term. If a list, assume a range (or spectrum; TODO: decide on this)
    shade_background
        Background light attenuation
    phy_self_shade
        Self-shading coefficient for phyto
    phy_growth_sat
        `num_res` by `num_phy` matrix of Monod/Michaelis-Menten half-saturation coefficients.
    res_forcing_amps
        Deep nutrient supply in resource equations
    zoo_grazing_rate_max
        max growth rates of zooplankton
    zoo_prey_pref
        Prey preference matrix; built off zoo_high_pref and zoo_low_pref
    zoo_slop_feed
        fraction of phyto biomass that is utilized by zoo when grazing
    phy_source
        additional source to add to phytoplankton compartments
    res_source
        additional source to add to resource compartments
    zoo_source
        additional source to add to zooplankton compartments
    dilute_zoo
        Subject zooplankton to dilution?
    zoo_mort_rate
        quadratic mortality rate for zooplankton
    linear_zoo_mort_rate
        (optional) linear mortality rate for zooplankton
    zoo_grazing_sat
        saturation coefficients for zooplankton grazing
    num_compartments
        total number of stored compartments in ecosystem. should be `num_phy` + 7, even if zooplankton disabled
    zoo_hill_coeff
        Exponent for grazing.
        n=1: Holling II
        n=2: Holling III
    phy_mort_rate
        mortality rates for phytoplankton
    phy_growth_rate_max
        maximal growth rates for phytoplankton
    include_zoo
        are we including zooplankton?
    res_uptake_type
        how are we uptaking resources?
        - 'perfect' (law of the minimum) by default
        - 'interactive' (product of Monod terms, rather than the minimum) also an option
    no3_inhibit_decay
        controls how much nitrate intake is inhibited by ammonium
    zoo_model_type
        form of zooplankton grazing. 'Real' (i.e. Michaelis-Menten) by default. 'KTW' also an option
    noise
        optional vector of noise to add to turnover rate
    noise_additive
        is noise additive or multiplicative (default)?
    dummy_param
        Dummy parameter for running the same simulation multiple times for sweeps (e.g. for ensemble of noisy sims)

    Returns
    ---------
    np.ndarray
        RHS in a `num_compartments` x 1 vector
    """

    ########################################################################

    # Assign indices for phyto, resources, predator
    num_dict = {'num_phy': num_phy, 'num_res': num_res, 'num_zoo': num_zoo}
    phy_indices = helpers.eco_indices('phy', num_dict)
    res_indices = helpers.eco_indices('res', num_dict)
    zoo_indices = helpers.eco_indices('zoo', num_dict)

    res_zoo_remin_frac = None

    def compute_turnover(turnover_rate=turnover_rate):
        if turnover_series is not None:
            if turnover_min is None or turnover_max is None:
                return turnover_series[int(t)]
            else:
                # Scale by turnover min and turnover max
                series_max = np.max(turnover_series)
                series_min = np.min(turnover_series)
                return turnover_min + (turnover_series[int(t)] - series_min) / (series_max - series_min) * (turnover_max - turnover_min)
        if None not in (turnover_min, turnover_max):
            # Mixed layer representation
            if turnover_kind == 'ramp':
                turnover_rate = helpers.ml_profile(low_val=turnover_min, high_val=turnover_max,
                                                   phase_shift=turnover_phase_shift,
                                                   mixed_layer_ramp_lengths=mixed_layer_ramp_lengths,
                                                   mixed_layer_ramp_times=mixed_layer_ramp_times, t=t)
                return turnover_rate
            elif turnover_kind == 'sine':
                mean_value = 0.5 * (turnover_min + turnover_max)
                return mean_value + 0.5 * (turnover_max - turnover_min) * np.sin(2 * np.pi / turnover_period * t + np.pi/4)
        if turnover_radius is not None:
            tau_max = turnover_rate * (1 + turnover_radius)
            tau_min = turnover_rate * (1 - turnover_radius)
            return turnover_rate + 0.5 * (tau_max - tau_min) * np.sin(2 * np.pi / turnover_period * t)
        else:
            return turnover_rate

    turnover_rate = compute_turnover()

    if noise is not None:
        # Additive or multiplicative noise?
        if noise_additive:
            turnover_rate += turnover_rate * noise[int(t)]
        else:
            turnover_rate *= noise[int(t)]

    if debug_dict:
        # add current time (if integer, approximately)
        if len(debug_dict['t']) == 0 or int(t) != int(debug_dict['t'][-1]):
            for key in debug_dict:
                if len(np.unique(debug_dict[key][0])) == 1 and np.unique(debug_dict[key][0])[0] == c.NAN_VALUE:
                    debug_dict[key][0] = np.zeros(debug_dict[key][0].shape)
                else:
                    debug_dict[key].append(np.zeros(debug_dict[key][-1].shape))
            debug_dict['t'][-1] = np.array([t])

    rhs = np.zeros((num_compartments,))

    # res_forcing/phy_forcing are prescribed forcings
    # May be "Newtonian cooling" like term for Tilman-like model

    phy = eco[phy_indices]
    res = eco[res_indices]
    zoo = eco[zoo_indices]

    prey = np.zeros(num_phy + num_zoo,)
    prey[:num_phy] = phy
    prey[num_phy:] = zoo

    #  grazing terms
    food_total_const = np.zeros(num_zoo, )  # F_rho in Vallina
    prey_preference_var = np.zeros((num_phy + num_zoo, num_zoo))  # phi in Vallina
    food_total_var = np.zeros(num_zoo, )  # F_phi in Vallina
    zoo_graze_mat = np.zeros((num_phy + num_zoo, num_zoo))  # grazing matrix
    zoo_grazing_sat_var = np.zeros(num_zoo, )  # variable saturation coefficient

    if include_zoo:
        prey_preference_product = prey[:, None] * zoo_prey_pref  # [num_phy+num_zoo, num_zoo]
        food_total_const = np.sum(prey_preference_product, axis=0)  # [num_zoo,]
        prey_preference_var = prey_preference_product / (food_total_const[None, :] + 1e-6)
        food_total_var = np.sum(prey[:, None] * prey_preference_var, axis=0)

        # variable saturation coefficient
        zoo_grazing_sat_var = zoo_grazing_sat * food_total_var / (food_total_const + 1e-6)

        # compute res_zoo_remin_frac dynamically
        # add small factor to prey/food
        if zoo_model_type == 'Real':
            # num_res x num_zoo
            res_zoo_remin_frac = (res_phy_remin_frac @ prey_preference_product[:num_phy, :]) / \
                                 (food_total_const[None, :] + 1e-6)

            res_zoo_remin_frac += (res_zoo_remin_frac @ prey_preference_product[num_phy:, :]) / \
                                        (food_total_const[None, :] + 1e-6)

        elif zoo_model_type == 'KTW':
            res_zoo_remin_frac = (res_phy_remin_frac @ prey_preference_var[:num_phy, :]) / \
                                 (food_total_var[None, :] + 1e-6)

            res_zoo_remin_frac += (res_zoo_remin_frac @ prey_preference_var[num_phy:, :]) / \
                                  (food_total_var[None, :] + 1e-6)

        if debug_dict:
            debug_dict['res_zoo_remin_frac'][-1] = res_zoo_remin_frac

    # phy terms
    phy_mort_term = phy_mort_rate if isinstance(phy_mort_rate, (list, np.ndarray)) \
        else phy_mort_rate * np.ones(num_phy,)

    # used to build zoo_graze_mat
    def get_grazing_term():

        switching_term = feeding_probability = None

        if zoo_model_type == 'Real':
            switching_term = zoo_prey_pref / (food_total_const[None, :] + 1e-6)
            feeding_probability = (food_total_const ** zoo_hill_coeff
                                   / (zoo_grazing_sat ** zoo_hill_coeff
                                      + food_total_const ** zoo_hill_coeff))

        elif zoo_model_type == 'KTW':
            switching_term = prey_preference_var / (food_total_var[None, :] + 1e-6)
            feeding_probability = (food_total_var ** zoo_hill_coeff
                                   / (zoo_grazing_sat_var ** zoo_hill_coeff
                                      + food_total_var ** zoo_hill_coeff))

        return zoo_grazing_rate_max[None, :] * switching_term * feeding_probability[None, :]

    phy_growth_list = eco[res_indices, None] / (eco[res_indices, None] + phy_growth_sat)

    # TODO: Add factor for changes due to light
    # We could add a parameter that allows us to yield a function of time
    # We want to ramp down gradually during winter, and up quickly during winter-spring bloom
    # This will be a piecewise linear combo
    if light_series is not None:
        light_factor = light_series[int(t)]
    elif None not in (light_min, light_max):
        if light_kind == 'ramp':
            light_factor = helpers.light_profile(low_val=light_min, high_val=light_max,
                                                 mixed_layer_ramp_lengths=mixed_layer_ramp_lengths,
                                                 mixed_layer_ramp_times=mixed_layer_ramp_times, t=t)
        elif light_kind == 'sine':
            mean_value = (light_min + light_max)/2
            light_factor = mean_value + (light_max - light_min)/2 * np.sin(2 * np.pi / c.NUM_DAYS_PER_YEAR * t + 5 * np.pi/4)
    else:
        light_factor = 1

    phy_res_growth_term = None
    if res_uptake_type == 'interactive':
        phy_res_growth_term = np.prod(phy_growth_list, axis=0)
    elif res_uptake_type == 'perfect':
        phy_res_growth_term = np.min(phy_growth_list, axis=0)

    phy_growth_vec = phy_growth_rate_max * phy_res_growth_term
    phy_growth_vec *= np.exp(-(shade_background + phy_self_shade * np.sum(phy)))

    # Scale growth rate by light
    phy_growth_vec *= light_factor

    # Gain in phytoplankton biomass corresponds to loss in resource
    res_uptake = -(phy_growth_vec[None, :] * res_phy_stoich_ratio) @ phy

    # -cji * U(I(t), R)
    rhs[res_indices] += res_uptake

    if debug_dict:
        debug_dict['res_uptake'][-1] += res_uptake

    if include_zoo:
        zoo_graze_mat = get_grazing_term()
        phy_zoomort = (-zoo_graze_mat[:num_phy, :] @ zoo) * phy

        # -sum_n Gin * Z[n]
        rhs[phy_indices] += phy_zoomort

        if debug_dict:
            debug_dict['phy_zoomort'][-1] += phy_zoomort

    # Phytoplankton terms
    phy_netgrowth = (phy_growth_vec - phy_mort_term - turnover_rate) * phy

    # (1-fe) * rji * phy_mort_term
    rhs[phy_indices] += phy_netgrowth

    if debug_dict:
        debug_dict['phy_growth'][-1] += phy_growth_vec * phy
        debug_dict['phy_mort'][-1] += -phy_mort_term * phy
        debug_dict['phy_turnover'][-1] += -turnover_rate * phy

    if include_zoo:

        zoo_mort_term = np.array(zoo_mort_rate) * zoo

        if linear_zoo_mort_rate is not None:
            zoo_mort_term += np.array(linear_zoo_mort_rate)

        zoo_graze_sum = zoo_slop_feed * (zoo_graze_mat[:num_phy, :].T @ phy)

        # beta_n * sum (Gin * Pi) - zoo_mort_term - tau
        rhs[zoo_indices] += (zoo_graze_sum - zoo_mort_term) * zoo

        if dilute_zoo:
            rhs[zoo_indices] += -turnover_rate * zoo

        if debug_dict:
            debug_dict['zoo_growth'][-1] += zoo_graze_sum * zoo
            debug_dict['zoo_mort'][-1] += -zoo_mort_term * zoo - (turnover_rate if dilute_zoo else 0) * zoo

        zoo_zoo_graze = zoo_slop_feed * (zoo_graze_mat[num_phy:, :].T @ zoo)
        zoo_zoo_mort = -zoo_graze_mat[num_phy:, :] @ zoo

        rhs[zoo_indices] += (zoo_zoo_graze + zoo_zoo_mort) * zoo

        if debug_dict:
            debug_dict['zoo_growth'][-1] += zoo_zoo_graze * zoo
            debug_dict['zoo_mort'][-1] += zoo_zoo_mort * zoo

    res_forcing_rhs = turnover_rate * (res_forcing_amps - res)
    rhs[res_indices] += res_forcing_rhs  # add forcing

    if debug_dict:
        debug_dict['res_forcing'][-1] += res_forcing_rhs

    # If any value is below extinction threshold and rhs < 0, set rhs to 0 and eco to 0 there
    # this prevents negative blowup.
    # phy_condition = np.where(phy < c.EXTINCT_THRESH)[0]
    # rhs[phy_indices][phy_condition] = 0
    # eco[phy_indices][phy_condition] = 0

    # any additional sources
    rhs[phy_indices] += phy_source
    rhs[res_indices] += res_source

    if include_zoo:
        rhs[zoo_indices] += zoo_source

    return rhs
