"""
single.py: A single run of the PHYRE model for a given parameter set
"""

import numpy as np
import phyre.helpers as helpers
from typing import Optional, Union
import phyre.components.integration as model_int
from typing import List, Tuple, Callable, Dict
from phyre import constants as c
import copy


def run(params_name: str = None, params: Dict = None, method: str = 'odeint', ts_label: str='time_series',
        details_str: str = None,
        return_eco: bool=False, save_eco: bool=True, data_ext: str = c.DATA_EXT_DEFAULT,
        functions: List[Tuple[str, Callable, Dict]] = None, debug: bool = False,
        odeint_kw: Dict = None) -> Optional[Union[np.ndarray, tuple]]:

    """Execute a single run of the model for the parameter set specified

    Parameters
    ----------
    params_name
        Identifier for param set to run. Used to load params if `params` is None.
    data_ext
        When we run, in what format should we save the data?
    params
        Dict of parameters.
    details_str
        Extra bits for filename
    ts_label
        Time series label. By default "time_series"
    functions
        The functions we'd like to apply to our output after each run
    method
        What method to use? Either 'odeint' or 'ode' for now
    debug
        Are we debugging?
    save_eco
        Are we saving ecosystem (and time vector?)
    return_eco
        Are we returning ecosystem?
    odeint_kw
        Arguments to ODEINT integrator
    """

    ########################################################################

    # load params
    if params is None:
        params = helpers.load(params_name, 'params', 'single', details_str=details_str)

    ########################################################################

    # INITIAL CONDITIONS / PREPARATION

    # Set up initial condition vector
    eco_0 = helpers.initial_conditions(params)

    ########################################################################

    # DEBUGGING

    bio = params.get('bio')

    if debug:
        params['bio']['debug_dict'] = helpers.debug_dict_setup(bio)

    ########################################################################

    # SOLVE

    # pop values we won't use
    popped_vals = helpers.additional_bio_setup(params, pop_values=True)
    dd = None

    (eco, t_save) = model_int.integrate(eco_0, params, method=method, odeint_kw=odeint_kw)

    # put keys back
    for key in popped_vals:
        if popped_vals[key] is not None:
            if key.endswith('_TEMP'):
                actual_name = '_'.join(key.split('_')[:-1])
                params['bio'][actual_name] = copy.deepcopy(popped_vals[key])
            else:
                params['bio'][key] = popped_vals[key]

    if debug:
        dd = params.get('bio').get('debug_dict')

        # convert lists to numpy arrays
        for key in dd:
            dd[key] = np.transpose(dd[key])

        # save
        helpers.save(params_name, 'debug', 'single', output=dd)

    if np.isnan(eco).any():
        print('NaN values encountered!')
        print('Problem integrating! System likely too stiff. Check your bio parameters.')
    else:

        if functions is not None:

            fn_count = 0

            for output in functions:

                output_name = output[0]
                output_fn = output[1]
                output_kw = output[2]

                if np.isnan(eco).any():

                    if fn_count == 0:
                        print('NaN values encountered!')
                        print('Problem integrating! System likely too stiff for sweep params below:')

                    # store as nans with same dimensions as output should be
                    fake_output = helpers.build_mock_eco_output(t_save, params)

                    out = np.full(np.shape(output_fn(fake_output, t_save, params, output_kw)),
                                  c.NAN_VALUE).tolist()
                else:
                    out = output_fn(eco, t_save, params, output_kw)

                helpers.save(params_name, 'data', 'single', output=out, data_ext=data_ext,
                             data_label=output_name, functions=functions, details_str=details_str)

                fn_count += 1

        # save time series and time as well
        if save_eco:
            helpers.save(params_name, 'data', 'single', data_label=ts_label, output=eco, data_ext=data_ext,
                         details_str=details_str)
            helpers.save(params_name, 'data', 'single', data_label='time', output=t_save, data_ext=data_ext,
                         details_str=details_str)

    if return_eco:
        if debug:
            return eco, dd
        return eco
