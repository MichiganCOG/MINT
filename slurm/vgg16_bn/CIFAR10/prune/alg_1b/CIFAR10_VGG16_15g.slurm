#!/bin/bash
#SBATCH --job-name="CIFAR10_VGG16_BN_BATCH_PRUNE_15g"
#SBATCH -N 1
#SBATCH --array=1-14
#SBATCH --partition=lgns
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --time=1500  # Time in Minutes
#SBATCH --output="CIFAR10_VGG16_BN_BATCH_PRUNE_15g_%A_%a.out"

export CUDA_VISIBLE_DEVICES="${SLURM_JOB_GPUS}"
echo "start"
cd /z/home/madantrg/scratch_incremental_learning/
source setup.sh 
cd /z/home/madantrg/Pruning/

python pruning_mi_global_1b.py \
--model vgg \
--dataset CIFAR10 \
--weights_dir /z/home/madantrg/Pruning/results/CIFAR10_VGG16_BN_BATCH/0/ \
--cores 20 \
--key_id $SLURM_ARRAY_TASK_ID \
--parent_clusters 15 \
--children_clusters 15 \
--name_postfix 15g_1b
 
echo "end"
