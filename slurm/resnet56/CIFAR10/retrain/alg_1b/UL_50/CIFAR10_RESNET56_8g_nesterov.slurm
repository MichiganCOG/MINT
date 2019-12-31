#!/bin/bash
#SBATCH --job-name="CIFAR10_RESNET56_BATCH_8g_1b_nesterov"
#SBATCH -N 1
#SBATCH --nodelist=fovea
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=7200  # Time in Minutes
#SBATCH --output="CIFAR10_RESNET56_BATCH_8g_1b_neterov.out"

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
--Nesterov \
--Milestones 100 150 \
--Weight_decay 0.0001 \
--Retrain /z/home/madantrg/Pruning/results/CIFAR10_RESNET56_BATCH/0/logits_best.pkl \
--Retrain_mask /z/home/madantrg/Pruning/results/CIFAR10_RESNET56_BATCH/0/I_parent_8g_1b.npy \
--Labels_file /z/home/madantrg/Pruning/results/CIFAR10_RESNET56_BATCH/0/Labels_8g_1b.npy \
--Labels_children_file /z/home/madantrg/Pruning/results/CIFAR10_RESNET56_BATCH/0/Labels_children_8g_1b.npy \
--parent_key conv1.weight  conv2.weight  conv3.weight  conv4.weight  conv5.weight  conv6.weight  conv7.weight  conv8.weight  conv9.weight  conv10.weight conv11.weight conv12.weight conv13.weight conv14.weight conv15.weight conv16.weight conv17.weight conv18.weight conv19.weight \
--children_key conv2.weight  conv3.weight  conv4.weight  conv5.weight  conv6.weight  conv7.weight  conv8.weight  conv9.weight  conv10.weight conv11.weight conv12.weight conv13.weight conv14.weight conv15.weight conv16.weight conv17.weight conv18.weight conv19.weight conv20.weight \
--parent_clusters 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 \
--children_clusters 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 16 \
--upper_prune_limit 0.50 \
--upper_prune_per 0.3 \
--lower_prune_per 0.2 \
--prune_per_step 0.001

echo "end"
