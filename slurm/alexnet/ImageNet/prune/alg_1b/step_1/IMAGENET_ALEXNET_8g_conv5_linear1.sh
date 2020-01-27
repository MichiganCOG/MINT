#!/bin/bash
#SBATCH --job-name="IMAGENET_ALEXNET_BATCH_PRUNE_8g_1b_conv5_linear1"
#SBATCH --output="IMAGENET_ALEXNET_BATCH_PRUNE_8g_1b_%j_conv5_linear1.out"
#SBATCH -N 1               # number of nodes requested
#SBATCH -n 1              # total number of mpi tasks requested
#SBATCH -p gtx    # queue (partition) -- normal, development, etc.
#SBATCH --gres=gpu:1
#SBATCH -t 24:00:00         # run time (hh:mm:ss) - 1.5 hours

cd /work/07080/salimeh/maverick2/MINT/
source setup.sh

module load launcher_gpu

 
export LAUNCHER_WORKDIR=$LAUNCHER_DIR
export LAUNCHER_PLUGIN_DIR=$LAUNCHER_DIR/plugins
export LAUNCHER_RMI=SLURM
export LAUNCHER_JOB_FILE="/work/07080/salimeh/maverick2/MINT/MINT/slurm/alexnet/ImageNet/prune/alg_1b/step_1/IMAGENET_ALEXNET_8g_conv5_linear1.slurm"
 
#$LAUNCHER_DIR/paramrun
#export LAUNCHER_JOB_FILE=Set-Your-Launcher-Job-File-here


${LAUNCHER_DIR}/paramrun