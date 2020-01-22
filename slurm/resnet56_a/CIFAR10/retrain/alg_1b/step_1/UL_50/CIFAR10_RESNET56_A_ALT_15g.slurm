#!/bin/bash
#SBATCH --job-name="CIFAR10_RESNET56_A_ALT_BATCH_15g_1b_step_1"
#SBATCH -N 1
#SBATCH --array=1-9
#SBATCH --partition=lgns
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=1200  # Time in Minutes
#SBATCH --output="CIFAR10_RESNET56_A_ALT_BATCH_15g_1b_%A_%a.out"

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
--Model resnet_a \
--Milestones 100 150 \
--Weight_decay 0.0001 \
--Retrain /z/home/madantrg/Pruning/results/CIFAR10_RESNET56_A_ALT_BATCH/0/logits_best.pkl \
--Retrain_mask /z/home/madantrg/Pruning/results/CIFAR10_RESNET56_A_ALT_BATCH/0/I_parent_15g_1b.npy \
--Labels_file /z/home/madantrg/Pruning/results/CIFAR10_RESNET56_A_ALT_BATCH/0/Labels_15g_1b.npy \
--Labels_children_file /z/home/madantrg/Pruning/results/CIFAR10_RESNET56_A_ALT_BATCH/0/Labels_children_15g_1b.npy \
--parent_key conv1.weight conv2.weight conv3.weight conv4.weight conv5.weight conv6.weight conv7.weight conv8.weight conv9.weight conv10.weight \
conv11.weight conv12.weight conv13.weight conv14.weight conv16.weight conv17.weight conv18.weight conv20.weight \
conv21.weight conv22.weight conv23.weight conv24.weight conv25.weight conv26.weight conv27.weight conv28.weight conv29.weight conv30.weight \
conv31.weight conv32.weight conv33.weight conv34.weight conv35.weight conv36.weight conv38.weight conv39.weight conv40.weight \
conv41.weight conv42.weight conv43.weight conv44.weight conv45.weight conv46.weight conv47.weight conv48.weight conv49.weight conv50.weight \
conv51.weight conv52.weight conv54.weight \
--children_key  conv2.weight conv3.weight conv4.weight conv5.weight conv6.weight conv7.weight conv8.weight conv9.weight conv10.weight \
conv11.weight conv12.weight conv13.weight conv14.weight conv15.weight conv17.weight conv18.weight conv19.weight \
conv21.weight conv22.weight conv23.weight conv24.weight conv25.weight conv26.weight conv27.weight conv28.weight conv29.weight conv30.weight \
conv31.weight conv32.weight conv33.weight conv34.weight conv35.weight conv36.weight conv37.weight conv39.weight conv40.weight \
conv41.weight conv42.weight conv43.weight conv44.weight conv45.weight conv46.weight conv47.weight conv48.weight conv49.weight conv50.weight \
conv51.weight conv52.weight conv53.weight conv55.weight \
--parent_clusters 15 15 15 15 15 15 15 15 15 15 \
15 15 15 15 15 15 15 32 \
32 32 32 32 32 32 32 32 32 32 \
32 32 32 32 32 32 64 64 64 \
64 64 64 64 64 64 64 64 64 64 \
64 64 64 \
--children_clusters 15 15 15 15 15 15 15 15 15 \
15 15 15 15 15 15 15 15 \
32 32 32 32 32 32 32 32 32 32 \
32 32 32 32 32 32 32 64 64 \
64 64 64 64 64 64 64 64 64 64 \
64 64 64 64 \
--upper_prune_limit 0.50 \
--upper_prune_per 0.1 \
--lower_prune_per 0.01 \
--prune_per_step 0.01 \
--Save_dir CIFAR10_RESNET56_A_ALT_RETRAIN_1_BATCH \
--key_id $SLURM_ARRAY_TASK_ID

echo "end"
