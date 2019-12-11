#!/bin/bash
#SBATCH --job-name="CIFAR10_RESNET56_BATCH"
#SBATCH -N 1
#SBATCH --nodelist=lgn2
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=330  # Time in Minutes
#SBATCH --output="CIFAR10_RESNET56_BATCH.out"

export CUDA_VISIBLE_DEVICES="${SLURM_JOB_GPUS}"
echo "start"
cd /z/home/madantrg/scratch_incremental_learning/
source setup.sh 
cd /z/home/madantrg/Pruning/
python train.py \
--Epoch 200 \
--Batch_size 128 \
--Lr 0.1 \
--Save_dir CIFAR10_RESNET56_BATCH \
--Dataset CIFAR10 \
--Dims 10 \
--Expt_rerun 1 \
--Milestones 100 150 \
--Opt sgd \
--Weight_decay 0.0001 \
--Model resnet \
--Gamma 0.1 \
--Device_ids 0

echo "end"
