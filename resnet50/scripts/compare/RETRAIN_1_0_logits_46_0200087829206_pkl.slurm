#!/bin/bash
#SBATCH --job-name="RETRAIN_1_0_logits_46_0200087829206_pkl_COMPARE"
#SBATCH --output="RETRAIN_1_0_logits_46_0200087829206_pkl_COMPARE.out"
#SBATCH -N 1               # number of nodes requested
#SBATCH --partition=lgns27
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH -t 80:00:00         # run time (hh:mm:ss) - 1.5 hours

echo ${SLURM_JOB_GPUS}
export CUDA_VISIBLE_DEVICES="${SLURM_JOB_GPUS}"
echo "Start Job"

cd /z/home/madantrg/ViP 
source setup.sh 

cd /z/home/madantrg/Pruning/resnet50

python compare_RETRAIN_1_0_logits_46_0200087829206_pkl.py

echo "End Job"
