#!/bin/bash
#SBATCH --job-name="IMAGENET_ALEXNET_BATCH_15g_1b_step_1_4gpu"
#SBATCH -N 1
#SBATCH --array=1-10
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4
#SBATCH --time=3000  # Time in Minutes
#SBATCH --output="IMAGENET_ALEXNET_BATCH_15g_1b_%A_%a_4gpu.out"

export CUDA_VISIBLE_DEVICES="${SLURM_JOB_GPUS}"

cd /z/home/madantrg/ViP/ 
source setup.sh 
cd /z/home/madantrg/Pruning
python retrain_stage3_speed_trial.py \
--Epoch 20 \
--Batch_size 1024 \
--Lr 0.0005 \
--Dataset IMAGENET \
--Model alexnet \
--Milestones 100 \
--Weight_decay 0.0001 \
--Retrain /z/home/madantrg/Pruning/results/IMAGENET_ALEXNET_BATCH/0/logits_best.pkl \
--Retrain_mask /z/home/madantrg/Pruning/results/IMAGENET_ALEXNET_BATCH/0/I_parent_8g_1b.npy \
--Labels_file /z/home/madantrg/Pruning/results/IMAGENET_ALEXNET_BATCH/0/Labels_8g_1b.npy \
--Labels_children_file /z/home/madantrg/Pruning/results/IMAGENET_ALEXNET_BATCH/0/Labels_children_8g_1b.npy \
--parent_key conv1.weight conv2.weight conv3.weight conv4.weight conv5.weight linear1.weight linear2.weight \
--children_key conv2.weight conv3.weight conv4.weight conv5.weight linear1.weight linear2.weight linear3.weight \
--parent_clusters 8 8 8 8 8 8 8 \
--children_clusters 8 8 8 8 8 8 10 \
--upper_prune_limit 0.50 \
--upper_prune_per 1.0 \
--lower_prune_per 0.1 \
--prune_per_step 0.1 \
--Save_dir /z/home/madantrg/Pruning/results/IMAGENET_ALEXNET_RETRAIN_1_BATCH \
--key_id $SLURM_ARRAY_TASK_ID \
--Dims 1000 \
--Device_ids 0 1 2 3
