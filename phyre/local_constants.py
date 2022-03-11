"""
local_constants.py: Local constants that are specific to each machine...
"""

import os
from phyre.constants import PROJECT_ROOT

USERNAME = 'name'
EMAIL_ADDRESS = 'email@email.com'

# data root directory. As an example, you could create a data directory as a subdirectory of the project root
# if you'd rather your data be on a 'scratch' directory on an HPC, for example, you can choose that instead
DATA_ROOT = os.path.join(PROJECT_ROOT, 'data')

###########################################

# generate a bash script that sets up an array of runs
# we fill in the number of runs (i.e. array 1-{})

# where to store script?
HPC_SWEEP_SCRIPT = os.path.join(PROJECT_ROOT, '{}_{}.s')

# what is in script?
HPC_SWEEP_SCRIPT_CONTENTS = (
    '''#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --time=10:00:00
#SBATCH --mem=5GB
#SBATCH --job-name=phyre
#SBATCH --mail-type=END
#SBATCH --mail-user={}
#SBATCH --output={}_%a.out
#SBATCH --array=0-{}

arr=($(seq 0 {}))

module purge

SRCDIR={}
RUNDIR={}
mkdir -p $RUNDIR

SCRIPTNAME="{}"
PARAMSNAME="{}"

cd $RUNDIR
module load anaconda3/4.3.1
python $SRCDIR/$SCRIPTNAME $PARAMSNAME -nc={} -c="${{arr[SLURM_ARRAY_TASK_ID]}}"
''')

###################################

# now we create functions to pass arguments to HPC_SWEEP_SCRIPT_CONTENTS
# this simply fills in the '{}' entries in the corresponding string above with the desired information
# Modify this as needed.

# IMPORTANT: you shouldn't need to change the arguments to the function - just the form of the return statement.


def get_hpc_sweep_script_contents(params_name, script_name, num_clusters):
    return HPC_SWEEP_SCRIPT_CONTENTS.format(EMAIL_ADDRESS, params_name, num_clusters-1, num_clusters-1,
                                            PROJECT_ROOT, DATA_ROOT,
                                            script_name, params_name, num_clusters)


###################################

# LEAVE THESE ALONE

SINGLE_DATA_DIR = os.path.join(DATA_ROOT, 'single/{}')
SINGLE_DATA_FILE = os.path.join(SINGLE_DATA_DIR, '{}.{}')

SWEEP_DATA_DIR = os.path.join(DATA_ROOT, 'sweep/{}/')
SWEEP_DATA_FILE = os.path.join(SWEEP_DATA_DIR, '{}.{}')
