#!/bin/bash
#SBATCH --job-name="ALG1B_MNIST_MLP_20g_PRUNE"
#SBATCH --output="ALG1B_MNIST_MLP_20g_%j_PRUNE.out"
#SBATCH --tasks-per-node=1
#SBATCH -N 2               # number of nodes requested
#SBATCH -n 2               # total number of mpi tasks requested
#SBATCH -p gtx             # queue (partition) -- normal, development, etc.
#SBATCH --gres=gpu:1
#SBATCH -t 01:00:00        # run time (hh:mm:ss) - 1.5 hours

cd /work/07080/salimeh/maverick2/MINT/
source setup.sh

module load launcher_gpu
 
export LAUNCHER_WORKDIR=$LAUNCHER_DIR
export LAUNCHER_PLUGIN_DIR=$LAUNCHER_DIR/plugins
export LAUNCHER_RMI=SLURM
export LAUNCHER_JOB_FILE="/work/07080/salimeh/maverick2/MINT/MINT/mlp/slurm/prune/alg_1b/tacc/ALG1B_20g.slurm"
 
#$LAUNCHER_DIR/paramrun
#export LAUNCHER_JOB_FILE=Set-Your-Launcher-Job-File-here


${LAUNCHER_DIR}/paramrun