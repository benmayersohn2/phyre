"""
bio.py: The biological components used by integrate.py
"""

from typing import Dict, Tuple, Union
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


# BUILD RHS
def rhs_build(eco: np.ndarray, t: int,
              num_phy: int,
              num_zoo: int,
              num_res: int,
              res_phy_stoich_ratio: np.ndarray,
              res_phy_remin_frac: np.ndarray,
              turnover_rate: float,
              phy_growth_sat: np.ndarray,
              res_forcing_amps: np.ndarray,
              num_compartments: int,
              phy_mort_rate: Union[np.ndarray, float],
              phy_growth_rate_max: np.ndarray,
              zoo_hill_coeff: float=1,
              phy_self_shade: float=0,
              silicate_off: bool=True,
              single_nit: bool=True,
              phy_small_indices: np.ndarray=None,
              phy_source: float=0,
              res_source: float=0,
              zoo_source: float=0,
              dilute_zoo: bool=True,
              zoo_mort_rate: float=None,
              zoo_grazing_rate_max: np.ndarray=None,
              zoo_slop_feed: np.ndarray=None,
              zoo_prey_pref: np.ndarray=None,
              zoo_grazing_sat: np.ndarray=None,
              res_export_frac: float=c.RES_EXPORT_DEFAULT,
              include_zoo: bool=False,
              res_uptake_type: str='perfect',
              no3_inhibit_decay: float=c.NO3_DECAY_DEFAULT,
              zoo_model_type: str='Real',
              noise: np.ndarray=None,
              debug_dict: Dict=None
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
    phy_small_indices
        Which indices are assigned to small phyto? Values range from 0 through num_phy-1
    res_phy_stoich_ratio
        `num_res` by `num_phy` matrix of stoichiometric coefficients. Convert from carbon to nutrient units
    res_phy_remin_frac
        Converts carbon from phytoplankton into resource units
    turnover_rate
        Dilution coefficient
    phy_self_shade
        Self-shading coefficient for phyto
    phy_growth_sat
        `num_res` by `num_phy` matrix of Monod/Michaelis-Menten half-saturation coefficients.
    res_forcing_amps
        Deep nutrient supply in resource equations
    zoo_grazing_rate_max
        max growth rates of zooplankton
    single_nit
        combines nh4 and no3 into a single compartment. res[0] = N, res[1] = nothing
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
    zoo_grazing_sat
        saturation coefficients for zooplankton grazing
    num_compartments
        total number of stored compartments in ecosystem. should be `num_phy` + 7, even if zooplankton disabled
    zoo_hill_coeff
        Exponent for grazing.
        n=1: Holling II
        n=2: Holling III
    photic_depth
        depth of euphotic zone
    phy_mort_rate
        mortality rates for phytoplankton
    phy_growth_rate_max
        maximal growth rates for phytoplankton
    res_export_frac
        fraction of recycled biomass that is lost from the euphotic zone. 0 by default
        this assumes that ALL biomass exported in same proportion.
    include_zoo
        are we including zooplankton?
    silicate_off
        have we turned silicate off?
    mixed_layer_ramp_lengths
        how long do we ramp up and down?
    mixed_layer_ramp_times
        at what times (from 0 to c.NUM_DAYS_PER_YEAR) to we start ramping?
    mixed_layer_ramp_extrema
        Minimum and maximum values of MLD
    res_uptake_type
        how are we uptaking resources?
        - 'perfect' (law of the minimum) by default
        - 'interactive' (product of Monod terms, rather than the minimum) also an option
    no3_inhibit_decay
        controls how much nitrate intake is inhibited by ammonium
    zoo_model_type
        form of zooplankton grazing. 'Real' (i.e. Michaelis-Menten) by default. 'KTW' also an option
    noise
        vector of noise at returned integration times.

    Returns
    ---------
    np.ndarray
        RHS in a `num_compartments` x 1 vector
    """

    def get_phy_growth_list():

        growth_list = list()
        nit_local = [c.NO3_INDEX, c.NH4_INDEX]
        nit_indices = np.array(res_indices)[nit_local]
        nh4_global = nit_indices[1]

        nit_growth = eco[nit_indices, None] / (eco[nit_indices, None] + phy_growth_sat[nit_local, :])

        other_indices = res_indices[c.NIT_COUNT:]
        other_local = range(c.NIT_COUNT, num_res)

        # [num_res, num_phy]
        if single_nit:
            # no growth on ammonium, override inhibition
            nit_growth_list = np.array([nit_growth[c.NO3_INDEX, :], np.zeros_like(nit_growth[c.NH4_INDEX, :])])
        else:
            nit_growth_list = np.array([nit_growth[c.NO3_INDEX, :] * np.exp(-no3_inhibit_decay * eco[nh4_global, None]),
                                        nit_growth[c.NH4_INDEX, :]])

        nit_combined = np.sum(nit_growth_list, axis=0).tolist()

        growth_list.append(nit_combined)

        other_terms = eco[other_indices, None] / (eco[other_indices, None] + phy_growth_sat[other_local, :])

        # Don't include silicate for small species
        if phy_small_indices is not None:
            other_terms[0, phy_small_indices] = 1  # ensure silicate is not limiting

        # leave out silicate entirely if we opted to
        if silicate_off:
            other_terms[0, :] = 1  # ensure silicate is not limiting

        growth_list.extend(other_terms.tolist())

        return np.array(growth_list), np.array(nit_growth_list)


    ########################################################################

    # Assign indices for phyto, resources, predator
    num_dict = {'num_phy': num_phy, 'num_res': num_res, 'num_zoo': num_zoo}
    phy_indices = helpers.eco_indices('phy', num_dict)
    res_indices = helpers.eco_indices('res', num_dict)
    zoo_indices = helpers.eco_indices('zoo', num_dict)

    res_zoo_remin_frac = None

    if noise is not None:
        turnover_rate += noise[int(t)]

    if debug_dict:
        # add current time (if integer, approximately)
        if len(debug_dict['t']) == 0 or int(t) != int(debug_dict['t'][-1]):
            for key in debug_dict:
                if len(np.unique(debug_dict[key][0])) == 1 and np.unique(debug_dict[key][0])[0] == c.NAN_VALUE:
                    debug_dict[key][0] = np.zeros(debug_dict[key][0].shape)
                else:
                    debug_dict[key].append(np.zeros(debug_dict[key][-1].shape))
            debug_dict['t'][-1] = np.array([t])

    rhs_linear = np.zeros((num_compartments, num_compartments))  # RHS matrix (linear part)

    # res_forcing/phy_forcing are prescribed forcings
    # May be "Newtonian cooling" like term for Tilman-like model

    res_slice = slice(res_indices[0], res_indices[0] + len(res_indices))
    phy_slice = slice(phy_indices[0], phy_indices[0] + len(phy_indices))
    zoo_slice = slice(zoo_indices[0], zoo_indices[0] + len(zoo_indices))
    
    phy = eco[phy_indices]
    res = eco[res_indices]
    zoo = eco[zoo_indices]

    prey = np.zeros((num_phy+1, num_zoo))

    # construct prey matrix
    prey[:-1, :] = phy[:, None]
    prey[-1, 1] = zoo[0]

    #  grazing terms
    food_total_const = np.zeros(num_zoo, )  # F_rho in Vallina
    prey_preference_var = np.zeros((num_phy + 1, num_zoo))  # phi in Vallina
    food_total_var = np.zeros(num_zoo, )  # F_phi in Vallina
    zoo_graze_mat = np.zeros((num_phy+1, num_zoo))  # grazing matrix
    zoo_grazing_sat_var = np.zeros(num_zoo, )  # variable saturation coefficient

    if include_zoo:
        prey_preference_product = prey * zoo_prey_pref  # [num_prey, num_zoo]
        food_total_const = np.sum(prey_preference_product, axis=0)  # [num_zoo,]
        prey_preference_var = prey_preference_product / food_total_const[None, :]
        food_total_var = np.sum(prey * prey_preference_var, axis=0)

        # variable saturation coefficient
        zoo_grazing_sat_var = zoo_grazing_sat * food_total_var / food_total_const

        #

        # compute res_zoo_remin_frac dynamically
        # add small factor to prey/food
        if zoo_model_type == 'Real':
            res_zoo_remin_frac = (res_phy_remin_frac @ prey_preference_product[:-1, :]) / \
                                 (food_total_const[None, :] + 1e-6)
            res_zoo_remin_frac[:, 1] += prey_preference_product[-1, 1] * res_zoo_remin_frac[:, 0] / \
                                        (food_total_const[1] + 1e-6)

        elif zoo_model_type == 'KTW':
            res_zoo_remin_frac = (res_phy_remin_frac @ prey_preference_var[:-1, :]) / \
                                 (food_total_var[None, :] + 1e-6)
            res_zoo_remin_frac[:, 1] += prey_preference_var[-1, 1] * res_zoo_remin_frac[:, 0] / \
                                        (food_total_var[1] + 1e-6)

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

    phy_growth_list, nit_list = get_phy_growth_list()

    partitions = None

    if single_nit:
        partitions = np.array([np.ones(num_phy,), np.zeros(num_phy,)])
    else:
        partitions = nit_list / np.sum(nit_list, axis=0)[None, :]

    phy_res_growth_term = None
    if res_uptake_type == 'interactive':
        phy_res_growth_term = np.prod(phy_growth_list, axis=0)
    elif res_uptake_type == 'perfect':
        phy_res_growth_term = np.min(phy_growth_list, axis=0)

    phy_growth_vec = phy_growth_rate_max * phy_res_growth_term
    phy_growth_vec *= np.exp(-phy_self_shade * np.sum(phy))

    # Gain in phytoplankton biomass corresponds to loss in resource
    res_uptake = -phy_growth_vec[None, :] * res_phy_stoich_ratio
    res_uptake[c.NO3_INDEX, :] *= partitions[c.NO3_INDEX, :]
    res_uptake[c.NH4_INDEX, :] *= partitions[c.NH4_INDEX, :]

    # -cji * U(I(t), R)
    rhs_linear[res_slice, phy_slice] += res_uptake

    if debug_dict:
        debug_dict['res_uptake'][-1] += res_uptake @ phy

    if include_zoo:

        zoo_graze_mat = get_grazing_term()
        phy_zoomort = -zoo_graze_mat[:-1, :] @ zoo

        # -sum_n Gin * Z[n]
        rhs_linear[phy_indices, phy_indices] += phy_zoomort

        if debug_dict:
            debug_dict['phy_zoomort'][-1] += phy_zoomort * phy

    # (1-fe) * rji * phy_mort_term
    phy_mort_rhs = (1 - res_export_frac) * res_phy_remin_frac * phy_mort_term[None, :]
    rhs_linear[res_slice, phy_slice] += phy_mort_rhs

    if debug_dict:
        debug_dict['res_phymort'][-1] += phy_mort_rhs @ phy

    # Phytoplankton terms
    phy_netgrowth = phy_growth_vec - phy_mort_term - turnover_rate

    # (1-fe) * rji * phy_mort_term
    rhs_linear[phy_indices, phy_indices] += phy_netgrowth

    if debug_dict:
        debug_dict['phy_growth'][-1] += phy_growth_vec * phy
        debug_dict['phy_mort'][-1] += -phy_mort_term * phy
        debug_dict['phy_turnover'][-1] += -turnover_rate * phy

    if include_zoo:
        # [num_res, num_phy, num_zoo]. sum over zoo axis
        mort_term_phyzoo = (1 - res_export_frac) * np.sum(
            (res_phy_remin_frac[:, :, None] - zoo_slop_feed[None, None, :] *
             res_zoo_remin_frac[:, None, :]) * zoo_graze_mat[None, :-1, :] * zoo[None, None, :], axis=2)

        # (1-fe) * rji * sum_n (1-beta_n) * Gin * Zn
        rhs_linear[res_slice, phy_slice] += mort_term_phyzoo

        if debug_dict:
            debug_dict['res_sloppy'][-1] += mort_term_phyzoo @ phy

        mort_term_zoozoo = zoo_graze_mat[-1, 1] * zoo[1]
        mort_term_zoozoo *= (res_zoo_remin_frac[:, 0] - zoo_slop_feed[1] * res_zoo_remin_frac[:, 1]) \
            * (1 - res_export_frac)

        rhs_linear[res_indices, zoo_indices[0]] += mort_term_zoozoo

        if debug_dict:
            debug_dict['res_sloppy'][-1] += mort_term_zoozoo * zoo[0]

        zoo_mort_term = zoo_mort_rate * zoo

        # this is just phytoplankton, small zoo => large zoo not yet included
        zoo_graze_sum = (zoo_slop_feed[None, :] * zoo_graze_mat[:-1, :]).T @ phy

        # beta_n * sum (Gin * Pi) - zoo_mort_term - tau
        rhs_linear[zoo_indices, zoo_indices] += zoo_graze_sum - zoo_mort_term

        if dilute_zoo:
            rhs_linear[zoo_indices, zoo_indices] += -turnover_rate

        if debug_dict:
            debug_dict['zoo_growth'][-1] += zoo_graze_sum * zoo
            debug_dict['zoo_mort'][-1] += -zoo_mort_term * zoo - (turnover_rate if dilute_zoo else 0)

        # Linear mortality (both zoo)
        res_zoomort_linear = (1 - res_export_frac) * res_zoo_remin_frac * zoo_mort_term
        rhs_linear[res_slice, zoo_slice] += res_zoomort_linear

        # Quadratic mortality
        res_zoomort_quadratic = (1 - res_export_frac) * res_zoo_remin_frac * zoo_mort_term
        rhs_linear[res_slice, zoo_slice] += res_zoomort_quadratic

        if debug_dict:
            debug_dict['res_zoomort'][-1] += (res_zoomort_linear + res_zoomort_quadratic) @ zoo

        # We still need to include growth effect of small zoo on large zoo
        zoo_zoo_graze = zoo_slop_feed[1] * zoo_graze_mat[-1, 1] * zoo[0]
        zoo_zoo_mort = -zoo_graze_mat[-1, 1] * zoo[1]

        rhs_linear[zoo_indices[1], zoo_indices[1]] += zoo_zoo_graze
        rhs_linear[zoo_indices[0], zoo_indices[0]] += zoo_zoo_mort

        if debug_dict:
            debug_dict['zoo_growth'][-1][1] += zoo_zoo_graze * zoo[1]
            debug_dict['zoo_mort'][-1][0] += zoo_zoo_mort * zoo[0]

    rhs = rhs_linear @ eco
    res_forcing_rhs = turnover_rate * (res_forcing_amps - res)
    rhs[res_indices] += res_forcing_rhs  # add forcing

    if debug_dict:
        debug_dict['res_forcing'][-1] += res_forcing_rhs

    # If any value is below extinction threshold and rhs < 0, set rhs to 0 and eco to 0 there
    # this prevents negative blowup.
    phy_condition = np.where(phy < c.EXTINCT_THRESH)[0]
    rhs[phy_indices][phy_condition] = 0
    eco[phy_indices][phy_condition] = 0

    # any additional sources
    rhs[phy_indices] += phy_source
    rhs[res_indices] += res_source

    if include_zoo:
        rhs[zoo_indices] += zoo_source

    return rhs
