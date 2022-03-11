"""
run_hpc.py: Takes in params_name, cluster number, and total number of clusters as inputs.

This is run as a program from the command line:
python run_hpc.py example -nc 2

It can also be run as an array job using SLURM (see local_constants.py)
"""

from phyre.model import sweep
from phyre import ts_funs as f

from argparse import ArgumentParser

num_years = 10

functions = [('time_series', f.time_series, None),  # store ALL years in time series. Always include this
             ('avg_phy_total', f.average_value, {'num_years': num_years, 'kind': 'total',
                                                 'compartments': [{'phy': 'all'}]})
             ]


def run(args):
    sweep.run(args.params_name, functions=functions, method=args.method, cluster_kw=args.cluster_kw,
              data_ext=args.data_ext)


def main():
    parser = ArgumentParser()
    parser.add_argument("params_name", help="name of parameter set")
    parser.add_argument("-nc", "--num_clusters", help="total number of clusters in sweep", type=int)
    parser.add_argument("-c", "--clusters", help="cluster indices to run; specify a comma-separated list.", type=str)
    parser.add_argument("-m", "--method", help="integration method", type=str, default='odeint')
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
