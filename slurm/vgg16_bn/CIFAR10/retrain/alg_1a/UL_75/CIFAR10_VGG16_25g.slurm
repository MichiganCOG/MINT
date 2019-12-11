#!/bin/bash
#SBATCH --job-name="CIFAR10_VGG16_BN_BATCH_25g"
#SBATCH -N 1
#SBATCH --nodelist=cmos
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=7250  # Time in Minutes
#SBATCH --output="CIFAR10_VGG16_BN_BATCH_25g.out"

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
--Retrain_mask /z/home/madantrg/Pruning/results/CIFAR10_VGG16_BN_BATCH/0/I_parent_25g.npy \
--Labels_file /z/home/madantrg/Pruning/results/CIFAR10_VGG16_BN_BATCH/0/Labels_25g.npy \
--Labels_children_file /z/home/madantrg/Pruning/results/CIFAR10_VGG16_BN_BATCH/0/Labels_children_25g.npy \
--parent_key conv1.weight conv2.weight conv3.weight conv4.weight conv5.weight conv6.weight conv7.weight conv8.weight conv9.weight conv10.weight conv11.weight conv12.weight conv13.weight linear1.weight \
--children_key conv2.weight conv3.weight conv4.weight conv5.weight conv6.weight conv7.weight conv8.weight conv9.weight conv10.weight conv11.weight conv12.weight conv13.weight linear1.weight linear3.weight \
--parent_clusters 25 25 25 25 25 25 25 25 25 25 25 25 25 25 \
--children_clusters 25 25 25 25 25 25 25 25 25 25 25 25 25 10 \
--upper_prune_limit 0.75 \
--upper_prune_per 0.99 \
--lower_prune_per 0.90 \
--prune_per_step 0.001

echo "end"
