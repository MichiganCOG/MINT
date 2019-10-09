#!/bin/bash
#SBATCH --job-name="STL10_ALEXNET_BATCH"
#SBATCH -N 1
#SBATCH --exclude=choroid,lens,rhodopsin
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=330  # Time in Minutes
#SBATCH --output="STL10_ALEXNET_BATCH.out"

export CUDA_VISIBLE_DEVICES="${SLURM_JOB_GPUS}"
echo "start"
cd /z/home/madantrg/scratch_incremental_learning/
source setup.sh 
cd /z/home/madantrg/Pruning/
python train.py \
--Epoch 200 \
--Batch_size 128 \
--Lr 0.01 \
--Save_dir STL10_ALEXNET_BATCH \
--Dataset STL10 \
--Dims 10 \
--Expt_rerun 1 \
--Milestones 60 120 160 \
--Opt sgd \
--Weight_decay 0.0005 \
--Model alexnet \
--Gamma 0.2 \
--Nesterov \
--Device_ids 0 \
--Prune_step 1

echo "end"
