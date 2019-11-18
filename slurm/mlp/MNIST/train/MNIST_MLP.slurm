#!/bin/bash
#SBATCH --job-name="MNIST_MLP_BATCH"
#SBATCH -N 1
#SBATCH --nodelist=lgn2
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=330  # Time in Minutes
#SBATCH --output="MNIST_MLP_BATCH.out"

export CUDA_VISIBLE_DEVICES="${SLURM_JOB_GPUS}"
echo "start"
cd /z/home/madantrg/scratch_incremental_learning/
source setup.sh 
cd /z/home/madantrg/Pruning/
python train_stage1.py \
--Epoch 30 \
--Batch_size 256 \
--Lr 0.1 \
--Save_dir MNIST_MLP_BATCH \
--Dataset MNIST \
--Dims 10 \
--Model mlp \
--Nesterov \
--Milestones 10 20 \
--Opt sgd \
--Device_ids 0

echo "end"
