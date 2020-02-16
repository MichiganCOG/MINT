#!/bin/bash
#SBATCH --job-name="ALG1B_CIFAR10_VGG16_BN_FULL_FULL_64g_650s_650s_RETRAIN"
#SBATCH -N 1
#SBATCH --array=1-30
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --time=05:00:00  # Time in Minutes
#SBATCH --output="ALG1B_CIFAR10_VGG16_BN_FULL_FULL_64g_650s_650s_%A_%a_RETRAIN.out"

export CUDA_VISIBLE_DEVICES="${SLURM_JOB_GPUS}"

echo "Start Job"

cd /z/home/madantrg/ViP/
source setup.sh 

cd /z/home/madantrg/Pruning/vgg16_full/

python retrain_stage3.py \
--Epoch 300 \
--Batch_size 128 \
--Lr 0.1 \
--Dataset CIFAR10 \
--Dims 10 \
--Expt_rerun 1 \
--Milestones 90 180 260 \
--Opt sgd \
--Weight_decay 0.0005 \
--Model vgg \
--Gamma 0.2 \
--Nesterov \
--Device_ids 0 \
--Retrain BASELINE_CIFAR10_VGG16_BN_FULL/0/logits_best.pkl \
--Retrain_mask BASELINE_CIFAR10_VGG16_BN_FULL/0/I_parent_64g_650s_1b.npy \
--Labels_file BASELINE_CIFAR10_VGG16_BN_FULL/0/Labels_64g_650s_1b.npy \
--Labels_children_file BASELINE_CIFAR10_VGG16_BN_FULL/0/Labels_children_64g_650s_1b.npy \
--parent_key   conv1.weight conv7.weight conv8.weight conv9.weight conv10.weight conv11.weight conv12.weight conv13.weight \
--children_key conv2.weight conv8.weight conv9.weight conv10.weight conv11.weight conv12.weight conv13.weight linear1.weight \
--parent_clusters   64 64 64 64 64 64 64 64 \
--children_clusters 64 64 64 64 64 64 64 64 \
--upper_prune_limit 0.8 \
--upper_prune_per 0.85 \
--lower_prune_per 0.82 \
--prune_per_step 0.001 \
--Save_dir BASELINE_CIFAR10_VGG16_BN_FULL_RETRAIN_1 \
--key_id $SLURM_ARRAY_TASK_ID

echo "end"
