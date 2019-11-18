#!/bin/bash
#SBATCH --job-name="CIFAR10_VGG16_BN_BATCH_PRUNE_25g_c6_c7"
#SBATCH -N 1
#SBATCH --partition=lgns
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --time=2000  # Time in Minutes
#SBATCH --output="CIFAR10_VGG16_BN_BATCH_PRUNE_25g_c6_c7.out"

export CUDA_VISIBLE_DEVICES="${SLURM_JOB_GPUS}"
echo "start"
cd /z/home/madantrg/scratch_incremental_learning/
source setup.sh 
cd /z/home/madantrg/Pruning/

python pruning_mi_global_vgg.py \
--weights_dir /z/home/madantrg/Pruning/results/CIFAR10_VGG16_BN_BATCH_PARALLEL/0/ \
--cores 20 \
--parent_key conv6.weight \
--children_key conv7.weight \
--parent_clusters 25 \
--children_clusters 25 \
--name_postfix p_25g_c6_c7
 
echo "end"
