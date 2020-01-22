#!/bin/bash
#SBATCH --job-name="CIFAR10_RESNET56_A_ALT_BATCH_PRUNE_15g_1b_Part5"
#SBATCH --output="CIFAR10_RESNET56_A_ALT_BATCH_PRUNE_15g_1b_%j_Part5.out"
#SBATCH -N 2               # number of nodes requested
#SBATCH -n 10              # total number of mpi tasks requested
#SBATCH -p gtx    # queue (partition) -- normal, development, etc.
#SBATCH --gres=gpu:1
#SBATCH -t 24:00:00         # run time (hh:mm:ss) - 1.5 hours

cd /work/07080/salimeh/maverick2/MINT/
source setup.sh

module load launcher_gpu

 
export LAUNCHER_WORKDIR=$LAUNCHER_DIR
export LAUNCHER_PLUGIN_DIR=$LAUNCHER_DIR/plugins
export LAUNCHER_RMI=SLURM
export LAUNCHER_JOB_FILE="/work/07080/salimeh/maverick2/MINT/MINT/slurm/resnet56_a/CIFAR10/prune/alg_1b/step_1/CIFAR10_RESNET56_A_ALT_15g_Part5.slurm"
 
#$LAUNCHER_DIR/paramrun
#export LAUNCHER_JOB_FILE=Set-Your-Launcher-Job-File-here


${LAUNCHER_DIR}/paramrun
