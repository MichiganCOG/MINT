#!/bin/bash
#SBATCH --job-name="CIFAR10_RESNET56_A_300EPPOCH_BATCH"
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=330  # Time in Minutes
#SBATCH --output="CIFAR10_RESNET56_A_300EPOCH_BATCH.out"

export CUDA_VISIBLE_DEVICES="${SLURM_JOB_GPUS}"
echo "start"
cd /z/home/madantrg/scratch_incremental_learning/
source setup.sh 
cd /z/home/madantrg/Pruning/
python train_stage1.py \
--Epoch 300 \
--Batch_size 128 \
--Lr 0.01 \
--Save_dir CIFAR10_RESNET56_A_300EPOCH_BATCH \
--Dataset CIFAR10 \
--Dims 10 \
--Expt_rerun 1 \
--Milestones 150 225 \
--Opt sgd \
--Weight_decay 0.0002 \
--Model resnet_a \
--Gamma 0.1 \
--Device_ids 0

echo "end"
