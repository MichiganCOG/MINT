#!/bin/bash
#SBATCH --job-name="ALG1B_MNIST_MLP_20g_RETRAIN"
#SBATCH --output="ALG1B_MNIST_MLP_20g_%j_RETRAIN.out"
#SBATCH -N 4              # number of nodes requested
#SBATCH -n 9              # total number of mpi tasks requested
#SBATCH -p gtx    # queue (partition) -- normal, development, etc.
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-job=14
#SBATCH -t 0:20:00         # run time (hh:mm:ss) - 1.5 hours

cd /work/07080/salimeh/maverick2/MINT/
source setup.sh

module load launcher_gpu

 
export LAUNCHER_WORKDIR=$LAUNCHER_DIR
export LAUNCHER_PLUGIN_DIR=$LAUNCHER_DIR/plugins
export LAUNCHER_RMI=SLURM
export LAUNCHER_JOB_FILE="/work/07080/salimeh/maverick2/MINT/MINT/mlp/slurm/retrain/alg_1b/step_1/ul_none/tacc/ALG1B_20g.slurm"
 
#$LAUNCHER_DIR/paramrun
#export LAUNCHER_JOB_FILE=Set-Your-Launcher-Job-File-here


${LAUNCHER_DIR}/paramrun
