#!/bin/bash
#SBATCH -J launcher            # job name
#SBATCH -o launcher.o%j        # output and error file name (%j expands to jobID)
#SBATCH -N 4               # number of nodes requested
#SBATCH -n 10              # total number of mpi tasks requested
#SBATCH -p gtx    # queue (partition) -- normal, development, etc.
#SBATCH --gres=gpu:1
#SBATCH -t 01:30:00         # run time (hh:mm:ss) - 1.5 hours

module load launcher_gpu
 
export LAUNCHER_WORKDIR=$LAUNCHER_DIR
export LAUNCHER_PLUGIN_DIR=$LAUNCHER_DIR/plugins
export LAUNCHER_RMI=SLURM
export LAUNCHER_JOB_FILE="/work/07080/salimeh/maverick2/MINT/MINT/slurm/dummy_launcher_gpu/LAUNCHER_GPU_SAMPLE.slurm"
 
#$LAUNCHER_DIR/paramrun
#export LAUNCHER_JOB_FILE=Set-Your-Launcher-Job-File-here

${LAUNCHER_DIR}/paramrun
~                                
