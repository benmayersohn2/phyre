"""
sweep.py: A sweep of runs of the PHYRE model for a given parameter set
"""

import numpy as np
import phyre.helpers as helpers
import phyre.components.integration as model_int
from typing import List, Tuple, Callable, Dict
from phyre import constants as c
import copy


def run(params_name: str, params_orig: Dict = None, method: str='odeint', cluster_kw: Dict = None,
        functions: List[Tuple[str, Callable, Dict]] = None, data_ext: str=c.DATA_EXT_DEFAULT,
        odeint_kw: Dict = None, details_str: str = None):

    """Execute a sweep of runs of the model for the parameter set specified

    Parameters
    ----------
    params_name
        Identifier for param set to run. Used to load params if `params` is None.
    params_orig
        Dict of parameters.
    details_str
        Extra bit to add to filenames
    method
        What method to use? Either 'odeint' or 'ode' for now (with the faster 'odeint' as default)
    data_ext
        What kind of file are we saving the data to?
    functions
        The functions we'd like to apply to our output after each run
    cluster_kw
        'cluster' (int): cluster index
        'num_clusters' (int): total number of clusters in sweep
        If we're not loading for a specific cluster, then 'cluster' kwarg not included
    odeint_kw
        Arguments to ODEINT integrator
    """

    ########################################################################

    # load params
    if params_orig is None:
        params_orig = helpers.load(params_name, 'params', 'sweep', cluster_kw=cluster_kw, details_str=details_str)

    ########################################################################

    # INITIAL CONDITIONS / PREPARATION

    # Set up initial condition vector
    eco_0 = helpers.initial_conditions(params_orig)

    ############################################################################

    sweeps = params_orig['sweep']['pp']

    name_list = tuple(sweep[0] for sweep in sweeps)
    entry_list = tuple(sweep[1] for sweep in sweeps)
    val_list = tuple(sweep[-1] for sweep in sweeps)

    val_length = len(val_list[0])
    output = np.zeros((val_length, len(functions)), dtype=object) if functions is not None else None

    time_saved = False

    # make deep copy of params_orig (because we will modify values as we sweep)

    for j in range(val_length):

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
                print(params['bio'][actual_name])

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
                print(params['bio'][name])

        # Update params
        popped_vals = helpers.additional_bio_setup(params, pop_values=True)

        # Run
        (eco, t_save) = model_int.integrate(eco_0, params, method=method, odeint_kw=odeint_kw, raise_errors=False)

        # put keys back
        for key in popped_vals:
            params['bio'][key] = popped_vals[key]

        fn_count = 0

        if not time_saved:
            helpers.save(params_name, 'data', 'sweep', output=t_save, data_label='time', data_ext=data_ext,
                         details_str=details_str)
            time_saved = True

        for i, f in enumerate(functions if functions is not None else []):

            output_name = f[0]
            output_fn = f[1]
            output_kw = f[2]

            if np.isnan(eco).any():

                print('NaN values encountered for {}!'.format(output_name))

                if fn_count == 0:
                    print('Problem integrating! System likely too stiff for sweep params below:')

                # store as nans with same dimensions as output should be
                fake_output = helpers.build_mock_eco_output(t_save, params)

                output[j, i] = np.full(np.shape(output_fn(fake_output, copy.deepcopy(t_save), params, output_kw)),
                                       c.NAN_VALUE).tolist()
            else:
                # ensure params aren't modified by any changes
                output[j, i] = output_fn(copy.deepcopy(eco), copy.deepcopy(t_save), copy.deepcopy(params),
                                         copy.deepcopy(output_kw))

            # save if this is our last go
            if j == val_length-1:
                helpers.save(params_name, 'data', 'sweep', output=output[:, i], data_ext=data_ext,
                             data_label=output_name, functions=functions, cluster_kw=cluster_kw,
                             details_str=details_str)

            fn_count += 1
