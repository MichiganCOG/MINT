#!/bin/bash
#SBATCH --job-name="CIFAR10_VGG16_BN_BATCH_8g"
#SBATCH -N 1
#SBATCH --partition=lgns
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=7200  # Time in Minutes
#SBATCH --output="CIFAR10_VGG16_BN_BATCH_8g.out"

export CUDA_VISIBLE_DEVICES="${SLURM_JOB_GPUS}"
echo "start"
cd /z/home/madantrg/scratch_incremental_learning/
source setup.sh 
cd /z/home/madantrg/Pruning/

python retrain_stage3.py \
--Epoch 40 \
--Batch_size 128 \
--Lr 0.001 \
--Dataset CIFAR10 \
--Model vgg \
--Nesterov \
--Milestones 100 \
--Retrain /z/home/madantrg/Pruning/results/CIFAR10_VGG16_BN_BATCH/0/logits_best.pkl \
--Retrain_mask /z/home/madantrg/Pruning/results/CIFAR10_VGG16_BN_BATCH/0/I_parent_8g.npy \
--Labels_file /z/home/madantrg/Pruning/results/CIFAR10_VGG16_BN_BATCH/0/Labels_8g.npy \
--Labels_children_file /z/home/madantrg/Pruning/results/CIFAR10_VGG16_BN_BATCH/0/Labels_children_8g.npy \
--parent_key conv1.weight conv8.weight conv9.weight conv10.weight conv11.weight conv12.weight conv13.weight \
--children_key conv2.weight conv9.weight conv10.weight conv11.weight conv12.weight conv13.weight linear1.weight \
--parent_clusters 8 8 8 8 8 8 8 \
--children_clusters 8 8 8 8 8 8 8 \
--upper_prune_limit 0.85 \
--upper_prune_per 0.95 \
--lower_prune_per 0.10 \
--prune_per_step 0.01

echo "end"
