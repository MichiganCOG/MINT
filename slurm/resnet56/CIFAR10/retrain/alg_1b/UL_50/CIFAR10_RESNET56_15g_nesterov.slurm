#!/bin/bash
#SBATCH --job-name="CIFAR10_RESNET56_BATCH_15g_1b_nesterov"
#SBATCH -N 1
#SBATCH --nodelist=fovea
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=7200  # Time in Minutes
#SBATCH --output="CIFAR10_RESNET56_BATCH_15g_1b_nesterov.out"

export CUDA_VISIBLE_DEVICES="${SLURM_JOB_GPUS}"
echo "start"
cd /z/home/madantrg/scratch_incremental_learning/
source setup.sh 
cd /z/home/madantrg/Pruning/

python retrain_stage3.py \
--Epoch 40 \
--Batch_size 256 \
--Lr 0.01 \
--Dataset CIFAR10 \
--Model resnet \
--Nesterov \
--Milestones 100 \
--Retrain /z/home/madantrg/Pruning/results/CIFAR10_RESNET56_BATCH/0/logits_best.pkl \
--Retrain_mask /z/home/madantrg/Pruning/results/CIFAR10_RESNET56_BATCH/0/I_parent_15g_1b.npy \
--Labels_file /z/home/madantrg/Pruning/results/CIFAR10_RESNET56_BATCH/0/Labels_15g_1b.npy \
--Labels_children_file /z/home/madantrg/Pruning/results/CIFAR10_RESNET56_BATCH/0/Labels_children_15g_1b.npy \
--parent_key conv1.weight  conv2.weight  conv3.weight  conv4.weight  conv5.weight  conv6.weight  conv7.weight  conv8.weight  conv9.weight  conv10.weight conv11.weight conv12.weight conv13.weight conv14.weight conv20.weight conv21.weight conv22.weight conv23.weight conv24.weight conv25.weight conv26.weight conv27.weight conv28.weight conv29.weight conv30.weight conv31.weight conv32.weight conv35.weight conv38.weight conv39.weight conv40.weight conv41.weight conv42.weight conv43.weight conv44.weight conv45.weight conv46.weight conv47.weight conv48.weight conv49.weight conv50.weight conv51.weight conv52.weight \
--children_key conv2.weight  conv3.weight  conv4.weight  conv5.weight  conv6.weight  conv7.weight  conv8.weight  conv9.weight  conv10.weight conv11.weight conv12.weight conv13.weight conv14.weight conv15.weight conv21.weight conv22.weight conv23.weight conv24.weight conv25.weight conv26.weight conv27.weight conv28.weight conv29.weight conv30.weight conv31.weight conv32.weight conv33.weight conv36.weight conv39.weight conv40.weight conv41.weight conv42.weight conv43.weight conv44.weight conv45.weight conv46.weight conv47.weight conv48.weight conv49.weight conv50.weight conv51.weight conv52.weight conv53.weight \
--parent_clusters 15 15 15 15 15 15 15 15 15 15 15 15 15 15 32 32 32 32 32 32 32 32 32 32 32 32 32 32 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 \
--children_clusters 15 15 15 15 15 15 15 15 15 15 15 15 15 15 32 32 32 32 32 32 32 32 32 32 32 32 32 32 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 \
--upper_prune_limit 0.50 \
--upper_prune_per 0.20 \
--lower_prune_per 0.05 \
--prune_per_step 0.001

echo "end"
