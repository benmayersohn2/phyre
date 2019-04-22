"""
Apply a function to sweep results after you've already run the sweep.
This will only work if you've applied the time_series function in `ts_funs.py` and stored it when running the sweep
(see `run_hpc.py`)

For the example given:

python apply_functions.py example -nc 2

"""

from phyre.analysis import analysis as al
from phyre import ts_funs as f
from argparse import ArgumentParser

num_years = 80

functions = [('species_richness', f.species_richness, {'num_years': num_years, 'mean_amp_thresh': 1})]


def run(args):
    runs = args.params_name.split(',')

    for name in runs:
        al.apply_functions(name, functions, cluster_kw=args.cluster_kw, data_ext=args.data_ext)


def main():
    parser = ArgumentParser()
    parser.add_argument("params_name", help="name of parameter set")
    parser.add_argument("-nc", "--num_clusters", help="total number of clusters in sweep", type=int)
    parser.add_argument("-c", "--clusters", help="cluster indices to run; specify a comma-separated list.", type=str)
    parser.add_argument("-d", "--data_ext", help="data extension to use", type=str, default='npy')

    args = parser.parse_args()

    if args.num_clusters is not None:
        args.cluster_kw = {'num_clusters': args.num_clusters}

        if args.clusters is not None:
            clusters = [int(x.strip()) for x in args.clusters.split(',')]
            for cluster in clusters:
                args.cluster_kw.update(**{'cluster': cluster})
                run(args)
        else:
            for i in range(args.num_clusters):
                args.cluster_kw.update(**{'cluster': i})
                run(args)
    else:
        args.cluster_kw = None
        run(args)


if __name__ == '__main__':
    main()
