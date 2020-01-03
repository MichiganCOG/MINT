#!/bin/bash
#SBATCH --job-name="CIFAR10_RESNET56_BATCH_15g_1b"
#SBATCH -N 1
#SBATCH --partition=lgns
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=7200  # Time in Minutes
#SBATCH --output="CIFAR10_RESNET56_BATCH_15g_1b.out"

export CUDA_VISIBLE_DEVICES="${SLURM_JOB_GPUS}"
echo "start"
cd /z/home/madantrg/scratch_incremental_learning/
source setup.sh 
cd /z/home/madantrg/Pruning/

python retrain_stage3.py \
--Epoch 200 \
--Batch_size 256 \
--Lr 0.01 \
--Dataset CIFAR10 \
--Model resnet \
--Milestones 100 150 \
--Weight_decay 0.0001 \
--Retrain /z/home/madantrg/Pruning/results/CIFAR10_RESNET56_BATCH/0/logits_best.pkl \
--Retrain_mask /z/home/madantrg/Pruning/results/CIFAR10_RESNET56_BATCH/0/I_parent_15g_1b.npy \
--Labels_file /z/home/madantrg/Pruning/results/CIFAR10_RESNET56_BATCH/0/Labels_15g_1b.npy \
--Labels_children_file /z/home/madantrg/Pruning/results/CIFAR10_RESNET56_BATCH/0/Labels_children_15g_1b.npy \
--parent_key conv39.weight conv40.weight conv41.weight conv42.weight conv43.weight conv44.weight conv45.weight conv46.weight conv47.weight conv48.weight conv49.weight conv50.weight conv51.weight conv52.weight conv53.weight \
--children_key conv40.weight conv41.weight conv42.weight conv43.weight conv44.weight conv45.weight conv46.weight conv47.weight conv48.weight conv49.weight conv50.weight conv51.weight conv52.weight conv53.weight conv54.weight \
--parent_clusters 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 \
--children_clusters 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 \
--upper_prune_limit 0.40 \
--upper_prune_per 0.8 \
--lower_prune_per 0.1 \
--prune_per_step 0.1

echo "end"
