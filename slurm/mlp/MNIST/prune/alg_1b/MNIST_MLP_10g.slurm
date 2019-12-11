#!/bin/bash
#SBATCH --job-name="MNIST_MLP_BATCH_PRUNE_10g"
#SBATCH -N 1
#SBATCH --array=1-2
#SBATCH --partition=lgns
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --time=2000  # Time in Minutes
#SBATCH --output="MNIST_MLP_BATCH_PRUNE_10g_%A_%a.out"

export CUDA_VISIBLE_DEVICES="${SLURM_JOB_GPUS}"
echo "start"
cd /z/home/madantrg/scratch_incremental_learning/
source setup.sh 
cd /z/home/madantrg/Pruning/

python pruning_mi_global_1b.py \
--model mlp \
--dataset MNIST \
--weights_dir /z/home/madantrg/Pruning/results/MNIST_MLP_BATCH/0/ \
--cores 20 \
--key_id $SLURM_ARRAY_TASK_ID \
--parent_clusters 10 \
--children_clusters 10 \
--name_postfix 10g_1b
 
echo "end"
