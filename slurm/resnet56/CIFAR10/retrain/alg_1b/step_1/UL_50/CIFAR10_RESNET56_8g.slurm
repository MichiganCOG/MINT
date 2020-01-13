#!/bin/bash
#SBATCH --job-name="CIFAR10_RESNET56_BATCH_8g_1b"
#SBATCH -N 1
#SBATCH --nodelist=focus
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=7200  # Time in Minutes
#SBATCH --output="CIFAR10_RESNET56_BATCH_8g_1b.out"

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
--Retrain_mask /z/home/madantrg/Pruning/results/CIFAR10_RESNET56_BATCH/0/I_parent_8g_1b.npy \
--Labels_file /z/home/madantrg/Pruning/results/CIFAR10_RESNET56_BATCH/0/Labels_8g_1b.npy \
--Labels_children_file /z/home/madantrg/Pruning/results/CIFAR10_RESNET56_BATCH/0/Labels_children_8g_1b.npy \
--parent_key conv20.weight conv21.weight conv22.weight conv23.weight conv24.weight conv25.weight conv26.weight conv27.weight conv28.weight conv29.weight conv30.weight conv31.weight conv32.weight conv33.weight conv34.weight conv35.weight conv36.weight \
--children_key conv21.weight conv22.weight conv23.weight conv24.weight conv25.weight conv26.weight conv27.weight conv28.weight conv29.weight conv30.weight conv31.weight conv32.weight conv33.weight conv34.weight conv35.weight conv36.weight conv37.weight \
--parent_clusters 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 \
--children_clusters 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 \
--upper_prune_limit 0.50 \
--upper_prune_per 0.8 \
--lower_prune_per 0.1 \
--prune_per_step 0.1

echo "end"
