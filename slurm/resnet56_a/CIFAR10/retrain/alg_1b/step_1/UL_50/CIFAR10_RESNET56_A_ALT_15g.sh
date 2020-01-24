#!/bin/bash
#SBATCH --job-name="CIFAR10_RESNET56_A_ALT_BATCH_RETRAIN_15g_1b"
#SBATCH --output="CIFAR10_RESNET56_A_ALT_BATCH_RETRAIN_15g_1b_%j.out"
#SBATCH -N 2               # number of nodes requested
#SBATCH -n 4              # total number of mpi tasks requested
#SBATCH --cpus-per-task=4
#SBATCH -p gtx    # queue (partition) -- normal, development, etc.
#SBATCH --gres=gpu:1
#SBATCH -t 15:00:00         # run time (hh:mm:ss) - 1.5 hours

cd /work/07080/salimeh/maverick2/MINT/
source setup.sh

module load launcher_gpu

 
export LAUNCHER_WORKDIR=$LAUNCHER_DIR
export LAUNCHER_PLUGIN_DIR=$LAUNCHER_DIR/plugins
export LAUNCHER_RMI=SLURM
export LAUNCHER_JOB_FILE="/work/07080/salimeh/maverick2/MINT/MINT/slurm/resnet56_a/CIFAR10/retrain/alg_1b/step_1/UL_50/CIFAR10_RESNET56_A_ALT_15g.slurm"
 
#$LAUNCHER_DIR/paramrun
#export LAUNCHER_JOB_FILE=Set-Your-Launcher-Job-File-here


${LAUNCHER_DIR}/paramrun
