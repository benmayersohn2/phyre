"""
helpers.py: Helper functions to assist with model execution/analysis
"""

from typing import Dict, Union, Tuple, List, Optional, Callable, Sized, Iterable
import numpy as np
import pandas as pd
import numpy.ma as ma
import matplotlib.pyplot as plt
import json
from numpy.random import default_rng
import os
import ast
from functools import partial
from phyre.analysis import analysis as al
from itertools import product as iter_product
import copy
from scipy.interpolate import interp1d
from scipy.signal import butter, sosfilt, freqz

from phyre import constants as c
from phyre import local_constants as lc


def load(name: str, file_type: str, run_type: str, pd_kw: Dict = None, cluster_kw: Dict = None,
         data_label: str = 'time_series', data_ext: str = c.DATA_EXT_DEFAULT, err_on_fail: bool = False,
         details_str: str = None) \
        -> Union[Dict, pd.DataFrame, np.ndarray]:

    """Load params or output data

    Parameters
    ----------
    name
        Label for the set of information we wish to load. Should be reflected in a corresponding directory or filename
    file_type
        Either 'params' or 'data'
    run_type
        Either 'sweep' or 'single'
    pd_kw
        'in_labels' (tuple): labels for input columns in DataFrame
        'out_classes' (tuple): labels for output (e.g. if multiple outputs per run)
    data_label
        If we're loading data, which data are we loading?
    data_ext
        What kind of file are we loading?
    details_str
            Additional details added to output name
    cluster_kw
        'cluster' (int): cluster index
        'num_clusters' (int): total number of clusters in sweep
        If we're not loading for a specific cluster, then 'cluster' kwarg not included
    err_on_fail
        If failure to load, raise error if True.

    Returns
    ----------
    Union[dict, pandas.DataFrame, numpy.ndarray]
        if `file_type` == 'params', returns a dictionary of parameters
        if `file_type` == 'data', returns either a pandas DataFrame or a NumPy array

    """

    # only get output_str if we have a single file, not a sweep distributed across output from multiple clusters
    output_str = get_output_filename(name, file_type, run_type, cluster_kw=cluster_kw, data_label=data_label,
                                     data_ext=data_ext, details_str=details_str)
    output_dir = get_output_dir(name, file_type, run_type)

    output = None
    if not os.path.exists(output_dir):
        print("Load directory does not exist!")

    if file_type in ('params', 'debug'):

        with open(output_str) as the_file:
            output = json.load(the_file)

        convert_lists_to_numpy(output)

    if file_type == 'data':

        if cluster_kw is not None and 'cluster' not in cluster_kw and data_label != 'time':
            num_clusters = cluster_kw.get('num_clusters')

            # load up sweep params for all clusters
            params = load(name, 'params', 'sweep', pd_kw=pd_kw, cluster_kw=cluster_kw)

            sweep = params.get('sweep').get('pp')

            num_samples = len(sweep[0][-1])
            num_samples_per_cluster = int(num_samples / num_clusters)
            output = np.full((num_samples,), fill_value=np.nan, dtype=object)
            pos = list(np.linspace(0, num_samples, num_clusters, endpoint=False))

            clust_kw = dict(cluster_kw)

            for j in range(num_clusters):
                clust_kw.update(**{'cluster': j})
                output_str = get_output_filename(name, file_type, run_type, cluster_kw=clust_kw,
                                                 data_label=data_label, data_ext=data_ext, details_str=details_str)
                try:
                    output_part = load_data(output_str, data_ext=data_ext, err_on_fail=err_on_fail)
                    output[int(pos[j]):int(pos[j] + num_samples_per_cluster)] = output_part.tolist()
                except:
                    print(f'Cluster {j} is missing!')
                    output[int(pos[j]):int(pos[j] + num_samples_per_cluster)] = \
                        (c.NAN_VALUE * np.ones((num_samples_per_cluster,))).tolist()
        # load just for a cluster
        else:
            output = load_data(output_str, data_ext=data_ext, err_on_fail=err_on_fail)

        if pd_kw:
            return data_to_pandas(name, output, pd_kw=pd_kw, cluster_kw=cluster_kw)

    return output


# load matrix of inputs, for sweep
def get_sweep_inputs(params_name: str, pd_kw: Dict = None, cluster_kw: Dict = None) -> np.ndarray:

    """ Get a matrix of sweep inputs using name of parameter set and some keyword arguments

        Parameters
        ----------
        params_name
            Label for the information we wish to load. Should be reflected in a corresponding directory or filename
        pd_kw
            'in_labels' (tuple): labels for input columns in DataFrame
            'out_classes' (tuple): labels for output (e.g. if multiple outputs per run)
        cluster_kw
            'cluster' (int): cluster index
            'num_clusters' (int): total number of clusters in sweep
            If we're not loading for a specific cluster, then 'cluster' kwarg not included

        Returns
        ----------
        numpy.ndarray
            Each row is a single combination of parameter choices in a sweep, and each column corresponds to a parameter
    """

    input_matrix = list()

    params = load(params_name, 'params', 'sweep', pd_kw=pd_kw, cluster_kw=cluster_kw)
    sweeps = params.get('sweep').get('pp')

    val_list = tuple(sweep[-1] for sweep in sweeps)

    for i in range(len(val_list[0])):
        current_entry = list()
        for j in range(len(sweeps)):
            current_entry.append(val_list[j][i])

        input_matrix.append(current_entry)

    # lastly, convert input matrix into a numpy object structure, return
    return np.array(input_matrix, dtype=object)


# data processing is more complicated, so we create a separate function
def load_data(output_str: str, data_ext: str=c.DATA_EXT_DEFAULT, err_on_fail: bool=False) -> Optional[np.ndarray]:

    """ Load output data from a single file

        Parameters
        ----------
        output_str
            Filename of output to be loaded
        data_ext
            What kind of file are we loading?
        err_on_fail
            Either raise an error (True) or return None (False)

        Returns
        ----------
        Optional[numpy.ndarray]
            Returns the data as a numpy array, unless there's an error - then None is returned.
    """

    output = None

    # if loading from csv...
    if data_ext == 'csv':
        try:
            output = np.loadtxt(output_str, delimiter=c.COL_DELIMITER).tolist()

        except ValueError:  # if we get a value error, we stored an object of irregular shape.
            # load data as unicode string, convert from string to true array

            # a newline marks a new entry; in 2D, we also must delimit by column
            raw_output = np.genfromtxt(output_str, delimiter=c.ROW_DELIMITER, dtype='U')

            # Bug found by Ines Mangolte :)
            if raw_output.shape == ():
                raw_output = [str(raw_output)]

            num_rows = np.size(raw_output)

            num_cols = len(raw_output[0].split(c.COL_DELIMITER))

            if num_cols > 1:
                s = (num_rows, num_cols)
            else:
                s = (num_rows,)

            output = np.zeros(s, dtype=object)

            for i in range(s[0]):

                if len(s) == 1:
                    curr_out = ast.literal_eval(raw_output[i].strip())
                    if len(curr_out) == 0:
                        output[i] = None
                    else:
                        output[i] = curr_out
                if len(s) == 2:
                    the_output = raw_output[i].split(c.COL_DELIMITER)
                    for j in range(s[1]):
                        try:
                            curr_out = ast.literal_eval(the_output[j].strip())
                            if len(curr_out) == 0:
                                output[i,j] = None
                            else:
                                output[i,j] = curr_out
                        except ValueError:
                            if err_on_fail:
                                raise Exception('ValueError: Could not convert output to array')
                            else:
                                output[i, j] = None

        except FileNotFoundError:
            if err_on_fail:
                raise Exception('File not found!!')
            else:
                return None

    if data_ext == 'npy':
        try:
            output = np.load(output_str, allow_pickle=True)
        except FileNotFoundError:
            if err_on_fail:
                raise Exception('File not found!!')
            else:
                return None

    return np.array(output)


def data_from_pandas(df: pd.DataFrame, x_label: str, y_label: str, out_label: str='output', nan: Optional[float]=c.NAN_VALUE,
                     extra_dim: bool=True) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    """ Convert 2D pandas DataFrame into a NumPy array

        Parameters
        ----------
        x_label
            Label for variable corresponding to the x-axis
        y_label
            Label for variable corresponding to the y-axis
        out_label
            Label for output column
        extra_dim
            For pcolormesh, x and y typically one dimension larger than C. If true, extra point added at end
        df
            DataFrame to convert to NumPy array.
        nan
            The value representing NaN, before masking. Should be an infeasible number for the dataset

        Returns
        ----------
        Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
            returns x value, y value, and output matrices as a tuple
    """

    output = df[out_label].values
    x_all = df[x_label].values
    y_all = df[y_label].values

    x_unique = list(np.unique(x_all))
    y_unique = list(np.unique(y_all))

    # add an extra gridpoint at the end for pcolor
    if extra_dim:
        x_extra = 2*x_unique[-1] - x_unique[-2]
        y_extra = 2*y_unique[-1] - y_unique[-2]

        x_vec = x_unique + [x_extra]
        y_vec = y_unique + [y_extra]
    else:
        x_vec = np.array(x_unique)
        y_vec = np.array(y_unique)

    x_mat, y_mat = np.meshgrid(x_vec, y_vec, sparse=False, indexing='xy')

    x_dict = dict([(str(x_unique[i]), i) for i in range(len(x_unique))])
    y_dict = dict([(str(y_unique[i]), i) for i in range(len(y_unique))])

    out_mat = np.zeros((len(y_unique), len(x_unique)))

    # we use xy indexing
    for i in range(len(output)):
        x_loc = x_dict[str(x_all[i])]
        y_loc = y_dict[str(y_all[i])]

        out_mat[y_loc, x_loc] = output[i]

    # mask values if nan specified
    if nan:
        out_mat[np.where(out_mat == nan)] = np.NaN
        out_mat = ma.masked_invalid(out_mat)

    return x_mat, y_mat, out_mat


def data_to_pandas(name: str, output: np.ndarray, pd_kw: Dict = None, cluster_kw: Dict = None):
    """ Convert from NumPy array to pandas table

        Parameters
        ----------
        name
            Name of sweep (we use this to get values taken on by labels)
        pd_kw
            Keywords for pandas
        output
            Data to convert
        cluster_kw
            Keywords for clusters


        Returns
        ----------
        pandas.DataFrame
            Pandas DataFrame with output
    """

    if pd_kw is not None and pd_kw.get('in_labels'):
        tuple_of_inputs = pd_kw.get('in_labels')
    else:
        tuple_of_inputs = ('in',)

    inputs = get_sweep_inputs(name, pd_kw=pd_kw, cluster_kw=cluster_kw)

    # create data dictionary
    out_label = pd_kw.get('out_label') if pd_kw is not None and 'out_label' in pd_kw else 'output'
    data = {out_label: list()}

    out_classes = pd_kw.get('out_classes') if pd_kw is not None else None

    if out_classes:
        data.update({'out_class': list()})

    for label in tuple_of_inputs:
        data.update({label: list()})

    for i in range(len(output)):
        current_input = list(inputs[i, :])

        if output[i] is not None:

            if not out_classes:
                the_output = [output[i]]
            else:
                the_output = output[i] if isinstance(output[i], (np.ndarray, tuple, list)) else [np.nan]

            s = np.shape(the_output)
            output_size = s[0]

            for j in range(output_size):
                for k in range(len(tuple_of_inputs)):
                    label = tuple_of_inputs[k]
                    data[label].append(current_input[k])
                if out_classes:
                    data['out_class'].append(out_classes[j])
                    data[out_label].append(the_output[j])
                else:
                    data[out_label].append(the_output[0])

    # return pandas DataFrame
    return pd.DataFrame.from_dict(data)


def save(name: str, file_type: str, run_type: str, output: Union[Dict, np.ndarray], cluster_kw: Dict = None,
         data_label: str='time_series', functions: List[Tuple[str, Callable, Dict]] = None,
         data_ext: str=c.DATA_EXT_DEFAULT, gen_hpc_scripts: bool=True, details_str: str = None):

    """ Save params, output data, or a run script for a sweep

        Parameters
        ----------
        name
            Label for the information we wish to save. Will be reflected in a corresponding directory or filename
        file_type
            Either 'params', 'data', or 'hpc'
        run_type
            Either 'sweep' or 'single'
        output
            Either dict of params or array of data
        data_label
            If we're loading data, which data are we saving?
        data_ext
            File extension we're saving. Also determines saving method.
        details_str
            Additional details added to output name.
        gen_hpc_scripts
            Produce HPC scripts (when saving 'params' for sweeps
        cluster_kw
            'cluster' (int): cluster index
            'num_clusters' (int): total number of clusters in sweep
            If we're not loading for a specific cluster, then 'cluster' kwarg not included
        functions
            List of transforms to apply to output data and save.
            Each tuple contains: name, function, list of args to function, dict of kwargs to function
    """

    output_dir = get_output_dir(name, file_type, run_type)
    output_str = get_output_filename(name, file_type, run_type, cluster_kw=cluster_kw, data_label=data_label,
                                     data_ext=data_ext, details_str=details_str)

    os.makedirs(output_dir, exist_ok=True)

    if file_type in ('params', 'debug'):

        if file_type == 'params':

            # add name to params
            if 'name' not in output:
                output['name'] = name

            # Basic bio setup
            # Add num_res, num_zoo, and num_compartments
            num_phy = output['bio'].get('num_phy')
            num_res = output['bio'].get('num_res')
            num_zoo = output['bio'].get('num_zoo')

            output['bio']['num_res'] = num_res
            output['bio']['num_zoo'] = num_zoo
            output['bio']['num_compartments'] = num_phy + num_zoo + num_res

            # Add tfinal, num_days, dt_save, t0
            output['t0'] = 0.0  # t0 can be between 0 (inclusive) and 360 (exclusive)

            # dt for saving results
            output['dt_save'] = 1.0

            if not output.get('num_days'):
                output['num_days'] = int(output['num_years'] * c.NUM_DAYS_PER_YEAR)

            output['t_final'] = output.get('num_days') - 1

            # Additional setup
            additional_bio_setup(output, safe_to_save=True)

            # for HPC
            if cluster_kw is not None and run_type == 'sweep':

                # make copy of cluster_kw
                clust_kw = dict(cluster_kw)

                new_params = copy.deepcopy(output)
                sweep_params = output.get('sweep')
                sweep = sweep_params['pp']

                num_clusters = cluster_kw.get('num_clusters')
                num_factors = len(sweep)

                # either grid, range, or values
                if 'type' not in sweep_params or sweep_params['type'] == 'grid':
                    # get all permutations of indices in cluster array
                    lengths = tuple(len(v[-1]) for v in sweep)
                    perms = list(iter_product(*[list(range(i)) for i in lengths]))
                    num_samples_total = len(perms)

                    samples = np.zeros((num_samples_total, num_factors), dtype=object)

                    for j in range(num_samples_total):

                        perm = perms[j]

                        for i in range(num_factors):
                            perm_index = perm[i]
                            samples[j, i] = sweep[i][-1][perm_index]

                elif sweep_params['type'] == 'range':
                    num_samples_total = sweep_params['num_samples']
                    samples = np.zeros((num_samples_total, num_factors), dtype=object)

                    # pick each sample randomly (uniform dist) from within specified range
                    for i in range(num_factors):
                        a, b = tuple(sweep[i][-1])  # min_val, max_val
                        samples[:, i] = a + (b-a) * np.random.random((num_samples_total,))

                elif sweep_params['type'] == 'values':
                    num_samples_total = sweep_params['num_samples']
                    samples = np.zeros((num_samples_total, num_factors), dtype=object)

                    # pick each sample index randomly (i.e. from list of values specified)
                    for i in range(num_factors):
                        vals = np.array(sweep[i][-1])
                        samples[:, i] = vals[np.random.randint(len(vals), size=(num_samples_total,))]

                else:
                    raise Exception('Type of sweep is unrecognized.')

                new_sweep = [x[:] for x in sweep]

                # store samples
                for i in range(num_factors):
                    output['sweep']['pp'][i][-1] = list(samples[:, i])

                # now divide into clusters
                num_entries_per_cluster = int(num_samples_total / num_clusters)

                pos = list(np.linspace(0, num_samples_total, num_clusters, endpoint=False))
                for j in range(num_clusters):
                    for i in range(num_factors):
                        new_sweep[i][-1] = list(samples[int(pos[j]):int(pos[j]+num_entries_per_cluster), i])

                    new_params['sweep']['pp'] = new_sweep

                    # save
                    convert_numpy_to_lists(new_params)

                    # get output string
                    clust_kw.update({'cluster': j})  # we don't want to mutate cluster_kw, so use copy
                    output_str = get_output_filename(name, file_type, run_type, cluster_kw=clust_kw,
                                                     data_label=data_label)

                    with open(output_str, 'w') as out_file:
                        json.dump(new_params, out_file, indent=2)

                # also save HPC scripts (for applying functions to output and running simulations)
                if gen_hpc_scripts:
                    hpc_str = get_output_filename(name, 'hpc', run_type, cluster_kw=cluster_kw, data_label=data_label)
                    apply_str = get_output_filename(name, 'hpc', run_type, cluster_kw=cluster_kw, data_label=data_label,
                                                    details_str='apply')
                    hpc_contents = lc.get_hpc_sweep_script_contents(name, 'run_hpc.py', num_clusters)
                    apply_contents = lc.get_hpc_sweep_script_contents(name, 'apply_functions.py', num_clusters)

                    with open(hpc_str, 'w') as out_file:
                        out_file.write(hpc_contents)

                    with open(apply_str, 'w') as out_file:
                        out_file.write(apply_contents)

                # save output_str for all sweeps into one file
                output_str = get_output_filename(name, file_type, run_type, cluster_kw=cluster_kw,
                                                 data_label=data_label)

        convert_numpy_to_lists(output)

        with open(output_str, 'w') as out_file:
            json.dump(output, out_file, indent=2)

    elif file_type == 'data':
        # default, save to csv
        if data_ext == 'csv':
            if functions is not None:
                np.set_printoptions(threshold=np.inf)
                np.savetxt(output_str, output, fmt='%5s')
            else:
                np.savetxt(output_str, output, delimiter=c.COL_DELIMITER)

        # save to numpy binary
        if data_ext == 'npy':
            np.save(output_str, output)


def get_output_dir(name: str, file_type: str, run_type: str) -> str:

    """ Retrieve output directory using parameters specified

        Parameters
        ----------
        name
            Information label. Will be reflected in a corresponding directory or filename
        file_type
            Either 'params', 'data', or 'hpc'
        run_type
            Either 'sweep' or 'single'

        Returns
        ----------
        str
            Output directory as a string
    """

    output_dir = ''

    if run_type == 'single':
        if file_type == 'params':
            output_dir = c.SINGLE_PARAMS_DIR.format(name)
        elif file_type == 'debug':
            output_dir = c.SINGLE_DEBUG_DIR.format(name)
        elif file_type == 'data':
            output_dir = lc.SINGLE_DATA_DIR.format(name)
    if run_type == 'sweep':
        if file_type == 'params':
            output_dir = c.SWEEP_PARAMS_DIR.format(name)
        elif file_type == 'data':
            output_dir = lc.SWEEP_DATA_DIR.format(name)

    return output_dir


def get_output_filename(name: str, file_type: str, run_type: str, data_label: str = None,
                        cluster_kw: Dict = None, data_ext: str=c.DATA_EXT_DEFAULT, details_str: str = None) -> str:

    """ Retrieve output filename using parameters specified

        Parameters
        ----------
        name
            Information label. Will be reflected in a corresponding directory or filename
        file_type
            Either 'params', 'data', or 'hpc'
        run_type
            Either 'sweep' or 'single'
        data_label
            What label did we give the file?
        details_str
            Do we want to append a string to end of output filename? Include it here
        data_ext
            Extension to save as. Only applicable for data
        cluster_kw
            'cluster' (int): cluster index
            'num_clusters' (int): total number of clusters in sweep
            If we're not loading for a specific cluster, then 'cluster' kwarg not included


        Returns
        ----------
        str
            Output filename as a string
    """

    file_num = get_file_num(file_type, data_label=data_label, cluster_kw=cluster_kw, details_str=details_str)

    fn = None
    if file_type == 'hpc':
        if run_type == 'sweep':
            fn = lc.HPC_SWEEP_SCRIPT.format(name, file_num)

    if file_type == 'params':

        if run_type == 'single':
            fn = c.SINGLE_PARAMS_FILE.format(name, file_num)

        elif run_type == 'sweep':
            fn = c.SWEEP_PARAMS_FILE.format(name, file_num)

    if file_type == 'debug':
        if run_type == 'single':
            fn = c.SINGLE_DEBUG_FILE.format(name, file_num)

    if file_type == 'data':
        if run_type == 'sweep':
            fn = lc.SWEEP_DATA_FILE.format(name, file_num, data_ext)
        else:
            fn = lc.SINGLE_DATA_FILE.format(name, file_num, data_ext)

    return fn


def color_cycle(num_colors: int, cmap: str='Set1') -> np.ndarray:

    """ Get list of colors from colormap for plotting

        Parameters
        ----------
        num_colors : int
            How many colors to include in cycle?
        cmap : str, optional
            Name of color map to use

        Returns
        ----------
        numpy.ndarray
            List of colors as `ndarray` using `get_cmap` method of `pyplot`
    """

    return plt.get_cmap(cmap)(np.linspace(0, 1, num_colors))


def generate_mock_eco_inputs(num_years: int=1) -> Tuple[np.ndarray, Dict, Iterable[Dict]]:
    """ Create sample outputs of ecosystem run without running model
        1 year of output, one resource, one nutrient, no zooplankton.
        Primarily used for testing functions

        Returns
        ----------
        Tuple[Sized, Dict, Iterable[Dict]]
            Returns time array, dict of parameters and list of compartment dicts
    """

    compartments = [{'phy': 'all'}]
    params = {'bio': {'num_res': 1, 'num_phy': 1, 'num_zoo': 1}, 'dt_save': 1, 'num_years': num_years}
    t = np.linspace(0, int(num_years * c.NUM_DAYS_PER_YEAR - 1), int(num_years * c.NUM_DAYS_PER_YEAR))
    return t, params, compartments


def build_mock_eco_output(t, params):
    # just make sine waves
    def f(x):
        return np.sin(2 * x * np.pi / c.NUM_DAYS_PER_YEAR) + 5

    the_output = []
    for i in range(params.get('bio').get('num_compartments')):
        the_output.append(f(t))

    return np.array(the_output)


def eco_indices(index_key: str, bio: Dict = None, params: Dict = None, as_list: bool = True) -> Union[List, np.ndarray]:
    """ Returns corresponding indices of a compartment in ecosystem output

        Parameters
        ----------
        index_key
            Either 'phy', 'res', or 'zoo' for phytoplankton, resources, and zooplankton
        bio
            Dict of BIO params. We only need 'num_phy', 'num_res', and 'num_zoo'
        params
            Dict of ALL params. If specified, bio dict will be extracted using params.get('bio')
        as_list
            Return indices as a list (True) or as numpy array (False)

        Returns
        ----------
        range
            Returns range of indices corresponding to all members of a compartment in ecosystem output
    """

    if params:
        bio = params.get('bio')

    num_phy = bio.get('num_phy')
    num_res = bio.get('num_res')
    num_zoo = bio.get('num_zoo')

    out_list = None
    if index_key == 'phy':
        out_list = list(range(num_phy))
    elif index_key == 'res':
        out_list = list(range(num_phy, num_phy + num_res))
    elif index_key == 'zoo':
        out_list = list(range(num_phy + num_res, num_phy + num_res + num_zoo))

    return out_list if as_list else np.array(out_list)


def get_last_n_days(model_output: np.ndarray, num_days_end: int) -> np.ndarray:
    """ Returns last n years of ecosystem output

        Parameters
        ----------
        model_output
            num_compartments x num_days output of ecosystem model. May be 1D if only 1 compartment
        num_days_end
            Number of days from the end of time series we want to retrieve

        Returns
        ----------
        np.ndarray
            Returns model_output, but for the last n years
    """

    num_days = np.shape(model_output)[-1]
    ind_end = range(num_days - num_days_end, num_days)

    if len(np.shape(model_output)) == 2:
        return np.array(model_output)[:, ind_end]

    return np.array(model_output)[ind_end]


# get last n years of matrix/vector
def get_last_n_years(model_output: np.ndarray, n: int) -> np.ndarray:
    """ Returns last n years of ecosystem output

        Parameters
        ----------
        model_output
            num_compartments x num_days output of ecosystem model. May be 1D if only 1 compartment
        n
            Number of years from the end of time series we want to retrieve

        Returns
        ----------
        np.ndarray
            Returns model_output, but for the last n years
    """

    num_days = np.shape(model_output)[-1]
    ind_end = range(int(num_days - n * c.NUM_DAYS_PER_YEAR), num_days)

    if len(np.shape(model_output)) == 2:
        return np.array(model_output)[:, ind_end]

    return np.array(model_output)[ind_end]

    # return get_last_n_days(model_output, int(n * c.NUM_DAYS_PER_YEAR))


def convert_lists_to_numpy(params: Dict):
    """ Converts lists in dict to numpy arrays.
        Useful when importing JSON files
        This will CHANGE params. Does not return a copy.

        Parameters
        ----------
        params
            Dict of parameters
    """

    for key in params:
        if isinstance(params[key], (np.ndarray, tuple, list)):
            try:
                if key != 'pp':
                    params.update({key: np.asarray(params[key])})
            except ValueError:  # if we get a value error, skip it
                pass
        elif isinstance(params[key], dict):  # apply this to inner dicts
            convert_lists_to_numpy(params[key])


def convert_numpy_to_lists(params: Dict):
    """ Converts numpy arrays to lists in dict.
        Useful when exporting to JSON
        This will CHANGE params. Does not return a copy.

        Parameters
        ----------
        params
            Dict of parameters
    """

    for key in params:
        if isinstance(params[key], np.ndarray):
            params.update({key: params[key].tolist()})
        elif isinstance(params[key], dict):  # apply this to inner dicts and lists
            convert_numpy_to_lists(params[key])
        elif isinstance(params[key], np.int):
            params.update({key: int(params[key])})
        elif isinstance(params[key], np.float):
            params.update({key: float(params[key])})


def get_file_num(file_type: str, data_label: str='time_series', cluster_kw: Dict = None, details_str: str = None) -> str:

    """ String of details about particular file we're trying to import/export.

        Parameters
        ----------
        file_type
            'data', 'params', or 'hpc'
        data_label
            What kind of data is this? Time series? Some function output?
        details_str
            We can choose our own label for output if we prefer (we use this for HPC function application script)
        cluster_kw
            'cluster' (int): cluster index
            'num_clusters' (int): total number of clusters in sweep
            If we're not loading for a specific cluster, then 'cluster' kwarg not included


        Returns
        ----------
        str
            Output string
    """

    input_list = list()

    if file_type == 'data':
        if not details_str:
            details_str = c.DATA

        input_list.append(data_label)

    if cluster_kw is not None and 'cluster' in cluster_kw:
        input_list.append('cluster_{}'.format(cluster_kw.get('cluster')))

    if details_str:
        input_list.append(details_str)

    if file_type == 'hpc':
        input_list.append(c.HPC)
    elif file_type == 'params':
        input_list.append(c.PARAMS)
    elif file_type == 'debug':
        input_list.append(c.DEBUG)

    return get_chained_args_string(input_list)


def get_chained_args_string(input_list: List) -> str:
    """ Chains together list of arguments. Typically used for building filenames.

        Parameters
        ----------
        input_list
            List of arguments to be chained together.

        Returns
        ----------
        str
            Chained argument string
    """

    the_string = ""

    if len(input_list) == 1:
        return input_list[0]

    for arg in input_list:

        if arg == input_list[0]:
            the_string = "{}_".format(arg)

        elif arg == input_list[-1]:
            the_string = "{}{}".format(the_string, arg)

        else:
            the_string = "{}{}_".format(the_string, arg)

    return the_string


# set up model initial conditions
def initial_conditions(params: Dict) -> np.ndarray:

    """ Returns matrix of initial conditions

        Parameters
        ----------
        params
            Dictionary of parameters

        Returns
        ----------
        numpy.ndarray
            Initial condition ecosystem matrix
    """

    num_compartments = params['bio']['num_compartments']

    eco_0 = np.zeros(num_compartments,)

    bio = params.get('bio')
    eco_0[eco_indices('phy', bio)] = params['phy_0']

    if isinstance(params['res_0'], (int, float)):
        eco_0[eco_indices('res', bio)] = params['res_0'] * bio['res_forcing_amps'] / bio['res_forcing_amps'][0]
    else:
        eco_0[eco_indices('res', bio)] = params['res_0']
    eco_0[eco_indices('zoo', bio)] = params['zoo_0'] if bio.get('include_zoo') else 0

    # set growth rates to zero if initial conditions are zero (i.e we want to turn species off)
    if isinstance(params['phy_0'], (np.ndarray, tuple, list)):
        dead_phy = np.where(params['phy_0'] == 0)[0]

        # we want to store the old values so we don't lose them.
        if len(dead_phy) > 0:
            params['bio']['phy_growth_rate_max_TEMP'] = copy.deepcopy(params['bio']['phy_growth_rate_max'])
            params['bio']['phy_growth_rate_max'][dead_phy] = 0.0

    if bio.get('include_zoo') and isinstance(params['zoo_0'], (np.ndarray, tuple, list)):
        dead_zoo = np.where(params['zoo_0'] == 0)[0]
        if len(dead_zoo) > 0:
            params['bio']['zoo_grazing_rate_max_TEMP'] = copy.deepcopy(params['bio']['zoo_grazing_rate_max'])
            params['bio']['zoo_grazing_rate_max'][dead_zoo] = 0.0

    # Reshape into a vector
    return eco_0


def short_name(input_string: str) -> str:
    """ Simple function, used for axes/legend labels when plotting. Just capitalize and take first letter.

        Parameters
        ----------
        input_string
            Input string

        Returns
        ----------
        str
            First letter of input string, capitalized
    """

    # capitalized, first letter
    return input_string.upper()[0]


def filter_df(df: pd.DataFrame, cond_func: Callable) -> Tuple[pd.DataFrame, Tuple]:
    """ Drop output under a common class in DataFrame if all entries under said class meet condition

        Parameters
        ----------
        df
            Input DataFrame
        cond_func
            Function of df that, if met, will drop all output under the corresponding label

        Returns
        ----------
        Tuple[pandas.DataFrame, Tuple]
            Returns the new DataFrame, along with a tuple of the new output labels
    """

    good_labels = list()

    # if no class, do nothing and just return

    if 'out_class' in df.columns:
        labels = np.unique(df['out_class'].values)

        for i in range(len(labels)):
            current = df[df['out_class'] == labels[i]]
            if cond_func(current):
                df = df.drop(current.index)
            else:
                good_labels.append(labels[i])

    return df, tuple(good_labels)


def restrict_ts(eco: np.ndarray, pp: Dict, compartments: Iterable[Dict] = None, num_years: int = None,
                num_days: int = None,
                kind: str='indiv') -> np.ndarray:

    """ Restrict full eco output time series by compartments and years

        Parameters
        ----------
        eco
            Input ecosystem
        pp
            Dict of eco parameters
        kind
            'indiv', 'shannon' or 'total'
        compartments
            List of compartment dictionaries. If none specified, will apply to all compartments
        num_years
            How many years to restrict time series to?
        num_days
            How many days?

        Returns
        ----------
        np.ndarray
            Restricted version of eco
    """

    if kind == 'shannon':
        phy_indices = None
        if compartments is not None:
            phy_indices = [x for x in compartments if 'phy' in x][0]['phy']
        return al.shannon_ent_rel(eco, pp, num_years=num_years, num_days=num_days, phy_indices=phy_indices)

    out = np.array(eco)
    if num_years is not None:
        out = get_last_n_years(out, num_years)

    if num_days is not None:
        out = get_last_n_days(out, num_days)

    out = out[all_compartment_indices(pp, compartments=compartments), :]

    if kind == 'indiv':
        return out

    return np.sum(out, axis=0)


def keys_from_dict_list(compartments: Iterable[Dict]) -> List:
    """ Take list of dictionary entries and return just the keys in a list

        Parameters
        ----------
        compartments
            List of dictionaries, whose keys are either 'phy', 'res', or 'zoo'

        Returns
        ----------
        List
            A list of just the keys
    """

    keys = []
    for i in range(0, len(compartments)):
        keys += list(compartments[i].keys())

    return keys


def all_eco_compartments(params: Dict) -> Iterable[Dict]:
    """ Get a list of dictionaries, where the key is a class of compartments, and the value is a list from 0 to the
        number of compartments in that class

        Parameters
        ----------
        params
            Dict of params

        Returns
        ----------
        Iterable[Dict]
            The list of dictionaries
    """

    bio = params.get('bio')
    dict_list = [{'phy': list(range(0, bio['num_phy']))},
                 {'res': list(range(0, bio['num_res']))},
                 {'zoo': list(range(0, bio['num_zoo']))}]

    return dict_list


# build list of legend labels from compartments
def get_name_list(params: Dict, compartments: Iterable[Dict] = None, for_plot=True) -> List:

    """ Get a list of the short names of all compartments, for plotting (e.g. ['P1', 'P2', 'R1'])

        Parameters
        ----------
        params
            Dict of params
        compartments
            List of compartment dictionaries. If none specified, returns all compartment names
        for_plot
            If for plotting, use subscripts

        Returns
        ----------
        List
            The list of the short names of all compartments
    """

    key_str = r'{}$_{}$' if for_plot else '{}{}'
    return compartments_map(params, lambda key, index: key_str.format(short_name(key), index + 1),
                            compartments=compartments)


def compartments_map(params: Dict, func: Callable, compartments: Iterable[Dict] = None) -> List:
    """ takes a lambda function of key, index, returns list

        Parameters
        ----------
        params
            Dict of params
        func
            Function to call on all compartments
        compartments
            List of compartment dictionaries

        Returns
        ----------
        List
            The list of outputs when `func` applied to each compartment
    """

    if compartments is None:
        compartments = all_eco_compartments(params)

    the_list = list()

    for list_dict in compartments:
        key = list(list_dict.keys())[0]
        entry = list_dict[key]
        if (isinstance(entry, str) and entry == 'all') or entry is None:
            indices = list(range(params['bio']['num_{}'.format(key)]))
        else:
            indices = list_dict[key]
        for i in range(0, len(indices)):
            index = indices[i]
            the_list.append(func(key, index))

    return the_list


def num_compartments_selected(params: Dict, compartments: Iterable[Dict] = None) -> int:
    """Returns all the indices of the specified compartments in the ecosystem

        Parameters
        ----------
        params
            Dict of params
        compartments
            List of compartment dictionaries. If none specified, will apply function to all compartments

        Returns
        ----------
        int
            Number of compartments selected, using compartments dict
    """

    return len(all_compartment_indices(params, compartments=compartments))


def all_compartment_indices(params: Dict, compartments: Iterable[Dict] = None) -> List:
    """Returns all the indices of the specified compartments in the ecosystem

        Parameters
        ----------
        params
            Dict of params
        compartments
            List of compartment dictionaries. If none specified, will apply function to all compartments

        Returns
        ----------
        List
            The list of indices specified by compartments
    """

    def chosen_index(key, index):  # get indices specified in compartment list_dict
        return eco_indices(key, params['bio'])[index]

    # get list of indices specified in compartments
    return compartments_map(params, chosen_index, compartments=compartments)

# Convenience functions for calculating ml index

# Is `s` list-like? Useful for vectorization
def is_listlike(s):
    return isinstance(s, (set, list, tuple, np.ndarray, pd.Series, pd.Index))


def year_wrap(var):
    return var % c.NUM_DAYS_PER_YEAR


# Returns a bool: True if we restart the year
def time_wraps(var):
    return year_wrap(var) != var


def get_ml_index(t: float, ramp_lengths: np.ndarray = c.ML_RAMP_LENGTHS_DEFAULT,
                 ramp_times: np.ndarray = c.ML_RAMP_TIMES_DEFAULT) -> int:

    """Returns current index of turnover rate

        Parameters
        ----------
        t
            current time
        ramp_lengths
            how long do we ramp up and down?
        ramp_times
            at what times (from 0 to c.NUM_DAYS_PER_YEAR) to we start ramping?


        Returns
        ----------
        int
            The turnover index we're currently on
    """

    t_mod = year_wrap(t)

    # MAXIMUM
    # If we're between the end of the second ramp and the start of the first
    max_bool = year_wrap(int(ramp_times[1] + ramp_lengths[1])) <= t_mod <= ramp_times[0]
    # Same as above, in the case that second ramp doesn't wrap into the new year
    max_bool2 = (t_mod <= ramp_times[0]) or (t_mod >= ramp_times[1] + ramp_lengths[1])
    max_bool2 = max_bool2 and (not time_wraps(ramp_times[1] + ramp_lengths[1]))
    if max_bool or max_bool2:
        return 0

    # SHOAL (concentrates predators)
    if ramp_times[0] <= t_mod <= ramp_times[0] + ramp_lengths[0]:
        return 1

    # MINIMUM
    if ramp_times[0] + ramp_lengths[0] <= t_mod <= ramp_times[1]:
        return 2

    # DEEPEN (dilutes nutrient and phyto)
    return 3


def ml_profile(low_val: float = 0, high_val: float = 1, phase_shift=0,
               mixed_layer_ramp_lengths: np.ndarray = c.ML_RAMP_LENGTHS_DEFAULT,
               mixed_layer_ramp_times: np.ndarray = c.ML_RAMP_TIMES_DEFAULT,
               t: Union[float, np.ndarray] = None):

    if t is None:
        t = np.linspace(0, c.NUM_DAYS_PER_YEAR - 1, int(c.NUM_DAYS_PER_YEAR))
    if is_listlike(t):
        f = partial(ml_profile, low_val=low_val, high_val=high_val, phase_shift=phase_shift,
                    mixed_layer_ramp_lengths=mixed_layer_ramp_lengths,
                    mixed_layer_ramp_times=mixed_layer_ramp_times)
        return np.array([f(t=x) for x in t])

    extrema = np.array([low_val, high_val])

    def year_wrap(var):
        return var % c.NUM_DAYS_PER_YEAR

    t_mod = year_wrap(t + phase_shift)

    index = get_ml_index(t=t+phase_shift, ramp_lengths=mixed_layer_ramp_lengths, ramp_times=mixed_layer_ramp_times)

    ramp_lengths = mixed_layer_ramp_lengths
    ramp_times = mixed_layer_ramp_times

    if index == 0:
        return extrema[1]

    if index == 1:
        return extrema[1] + (t_mod - ramp_times[0]) / ramp_lengths[0] * \
               (extrema[0] - extrema[1])

    if index == 2:
        return extrema[0]

    # otherwise, index == 3
    if year_wrap(int(ramp_times[1])) > t_mod:
        return extrema[0] + (t_mod + c.NUM_DAYS_PER_YEAR - ramp_times[1]) / ramp_lengths[1] \
               * (extrema[1] - extrema[0])

    return extrema[0] + (t_mod - ramp_times[1]) / ramp_lengths[1] * (extrema[1] - extrema[0])


# Growth rate factor from mixed layer profile
def light_profile(low_val=0, high_val=1, **kwargs):
    light = np.exp(-ml_profile(low_val=0, high_val=1, **kwargs))
    low_light_val = np.exp(-1)
    high_light_val = 1
    # light = 1 - ml_profile(low_val=0, high_val=1, **kwargs)
    # low_light_val, high_light_val = low_val, high_val
    return low_val + (high_val - low_val) * (light - low_light_val) / (high_light_val - low_light_val)


def get_unique_vals(eco: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Returns ecosystem and time at unique time points

        Parameters
        ----------
        eco
            ecosystem
        t
            list of times


        Returns
        ----------
        Tuple[numpy.ndarray, numpy.ndarray]
            Eco and time vectors in a tuple
    """
    eco_new = list()
    t_new = np.unique(t)
    for i in range(int(t[-1]+1)):
        index = len(t) - 1 - t[::-1].index(t[i])
        eco_new.append(eco[:, index])
    return np.transpose(eco_new), np.array(t_new)


####################################

# NOISE

# filter_type : 'highpass', 'bandpass', or 'lowpass'
def butter_gen(cutoff, filter_type='bandpass', order=5):
    sos = butter(order, cutoff, btype=filter_type, output='sos')
    return sos

def butter_filter(data, cutoff, filter_type='bandpass', order=5):
    sos = butter_gen(cutoff, filter_type=filter_type, order=order)
    y = sosfilt(sos, data)
    return y

# If time domain, use Butterworth filter. Otherwise use Fourier cutoffs
def generate_noise(t_final: float, t0: float = 0, amp: float = 1.0, cutoff_freq: Union[float, List] = None,
                   filter_type: str = None, uniform: bool = False, additive: bool = False, seed: bool = None,
                   domain: str ='frequency') -> np.ndarray:
    """creates a white noise filtered at the chosen frequencies

        Parameters
        ----------
        t_final : last timestep of the run (in days)
        t0 : first timestep (in days, =0 to start on 01/01)
        amp : amplitude of the noise (1.0 by default)
        cutoff_freq : either cutoff frequency (lowpass/highpass) or list (bandpass)
        filter_type : 'highpass', 'bandpass', 'lowpass'. If not specified, just white noise.
            if 'bandpass', then freq is [left, right]
        uniform : uniform distribution? (gaussian by default)
        additive : Additive noise?
        seed: Seed random number generator for reproducible results
        domain: Filter white noise in frequency or time domain?

        Returns
        ----------
        np.ndarray
            Vector of noise, filtered accordingly

    """

    if additive is None:
        additive = False

    time = np.arange(t0, t_final + 1, 1)

    print(len(time))

    rng = default_rng(seed=seed)

    if uniform:
        if additive:
            white_noise = amp * rng.uniform(-1, 1, time.shape[0])
        else:
            white_noise = rng.uniform(1-amp, 1+amp, time.shape[0])
    else:
        if additive:
            white_noise = amp * rng.normal(0, 1, time.shape[0])
        else:
            white_noise = 1 + amp * rng.normal(0, 1, time.shape[0])

    if domain == 'time':
        return butter_filter(white_noise, cutoff_freq, filter_type=filter_type)
    else:
        noise_freq = np.fft.rfftfreq(white_noise.size, d=time[1] - time[0])
        f_noise = np.fft.rfft(white_noise)

        cut_f_noise = f_noise.copy()

        if filter_type == 'bandpass':
            cut_f_noise[(noise_freq < cutoff_freq[0])] = 0
            cut_f_noise[(noise_freq > cutoff_freq[1])] = 0
        if filter_type == 'lowpass':
            cut_f_noise[(noise_freq > cutoff_freq)] = 0
        if filter_type == 'highpass':
            cut_f_noise[(noise_freq < cutoff_freq)] = 0

        return np.fft.irfft(cut_f_noise, n=len(time))  # inverse transform

#####################################################


# pre-run setup
def pop_before_integrating(pp: Dict):

    """ Get rid of values that aren't passed to integrator. Return key with value

        Parameters
        ----------
        pp
            Dictionary of parameters
    """

    popped_vals = dict()

    for key in c.KEYS_TO_POP:
        popped_vals[key] = pp['bio'].pop(key, None)

    if 'sweep' in pp:
        scale_keys = [s[0] for s in pp['sweep']['pp'] if s[0].endswith('_scale')]

        for key in scale_keys:
            actual_name = '_'.join(key.split('_')[:-1])
            orig_key = '{}_orig'.format(actual_name)
            popped_vals[orig_key] = pp['bio'].pop(orig_key, None)

    return popped_vals


def additional_bio_setup(pp: Dict, pop_values: bool=False, safe_to_save: bool=False) -> Optional[Dict]:
    """Take user-input and properly format bio dictionary for integration.

        Parameters
        ----------
        pp
            Dictionary of parameters
        pop_values
            If True, Pop out values that won't be passed to integrator. Returns a dict
            If False, returns None and doesn't pop out values.
        safe_to_save
            Will we be saving this dictionary after we return it?

        Returns
        ----------
        Optional[Dict]
            Dictionary of keys popped during pre-run setup, if pop_values=True. Otherwise no output

    """

    bb = pp.get('bio')
    num_res = bb.get('num_res')
    num_phy = bb.get('num_phy')
    num_zoo = bb.get('num_zoo')

    # if num_years not specified, compute from num_days
    if not pp.get('num_years') and pp.get('num_days'):
        pp['num_years'] = int(float(pp['num_days']) / c.NUM_DAYS_PER_YEAR)
    else:
        pp['num_days'] = int(pp['num_years'] * c.NUM_DAYS_PER_YEAR)

    # Make turnover series a function
    # Only do this if we're not saving
    """
    if bb.get('turnover_series') is not None and not safe_to_save:
        t = np.linspace(pp['t0'], pp['t_final'] + 1, pp['t_final'] + 2)
        pp['bio']['turnover_series'] = interp1d(t, pp['bio']['turnover_series'])

    if bb.get('light_series') is not None and not safe_to_save:
        t = np.linspace(pp['t0'], pp['t_final'] + 1, pp['t_final'] + 2)
        pp['bio']['light_series'] = interp1d(t, pp['bio']['light_series'])
    """

    if bb.get('noise_sd') is not None:
        amp = bb.get('noise_sd')
        freq = bb.get('noise_freq')
        noise_filter = bb.get('noise_filter')
        uniform = bb.get('noise_uniform')
        noise_additive = bb.get('noise_additive')

        # compute noise
        pp['bio']['noise'] = generate_noise(pp['t_final'], pp['t0'], amp, cutoff_freq=freq,
                                            filter_type=noise_filter, uniform=uniform, additive=noise_additive)

        for key in ['noise_freq', 'noise_filter', 'noise_uniform']:
            if key in bb:
                del bb[key]
    else:
        pp['bio']['noise'] = None

    res_phy_stoich_ratio = bb.get('res_phy_stoich_ratio')

    res_phy_remin_frac = np.array(res_phy_stoich_ratio)

    pp['bio']['res_phy_remin_frac'] = res_phy_remin_frac

    ######################################################################

    # Ensure phytoplankton grouping is consistent
    if bb.get('include_zoo'):
        pref_shape = np.shape(bb.get('zoo_prey_pref'))
        assert pref_shape[0] == bb.get('num_zoo') + bb.get('num_phy')
        assert pref_shape[-1] == bb.get('num_zoo')

    if bb.get('res_forcing_amps') is None and bb.get('res_forcing_scale') is not None:
        pp['bio']['res_forcing_amps'] = get_res_amps_from_scale(bb.get('res_forcing_scale'),
                                                                res_phy_stoich_ratio=res_phy_stoich_ratio)

    # pop ALL parameters we won't need for integration
    if pop_values:
        return pop_before_integrating(pp)


def split_pandas_vectors(columns: List, df: pd.DataFrame) -> pd.DataFrame:

    # create new dict with all columns.
    # We will return DataFrame using pd.DataFrame.from_dict(...)
    data = dict()
    col_names = list(df)
    for col_name in col_names:
        data[col_name] = list()

    # now go down and add vals
    for i in range(df.shape[0]):
        for j in range(len(col_names)):
            col_name = col_names[j]
            for k in range(len(df[columns[0]][0])):
                if col_name in columns:
                    data[col_name].append(df[col_name][i][k])
                else:
                    data[col_name].append(df[col_name][i])

    # print(data)
    return pd.DataFrame.from_dict(data)


def debug_dict_setup(bio):
    debug_dict = dict()

    num_res = bio.get('num_res')
    num_phy = bio.get('num_phy')
    num_zoo = bio.get('num_zoo')

    debug_dict['res_uptake'] = [np.full((num_res,), c.NAN_VALUE)]
    debug_dict['res_forcing'] = [np.full((num_res,), c.NAN_VALUE)]
    debug_dict['res_phymort'] = [np.full((num_res,), c.NAN_VALUE)]

    debug_dict['res_zoo_remin_frac'] = [np.full((num_res, num_zoo), c.NAN_VALUE)]

    debug_dict['phy_growth'] = [np.full((num_phy,), c.NAN_VALUE)]
    debug_dict['phy_mort'] = [np.full((num_phy,), c.NAN_VALUE)]
    debug_dict['phy_turnover'] = [np.full((num_phy,), c.NAN_VALUE)]

    # make sure we add time
    debug_dict['t'] = [np.full((1,), c.NAN_VALUE)]

    if bio.get('include_zoo'):
        debug_dict['res_zoomort'] = [np.full((num_res,), c.NAN_VALUE)]
        debug_dict['res_sloppy'] = [np.full((num_res,), c.NAN_VALUE)]
        debug_dict['phy_zoomort'] = [np.full((num_phy,), c.NAN_VALUE)]
        debug_dict['zoo_growth'] = [np.full((num_zoo,), c.NAN_VALUE)]
        debug_dict['zoo_mort'] = [np.full((num_zoo,), c.NAN_VALUE)]

    return debug_dict


def get_res_amps_from_scale(scale, res_phy_stoich_ratio=None, makeup_ratio=None):
    num_res = res_phy_stoich_ratio.shape[0]
    amps = scale * np.ones(num_res,)

    conversion = np.zeros((num_res,))

    for j in range(num_res):
        conversion[j] = np.mean(res_phy_stoich_ratio[j, :]) / np.mean(res_phy_stoich_ratio[0, :])

    amps = np.multiply(amps, conversion)  # in nutrient units

    return amps
