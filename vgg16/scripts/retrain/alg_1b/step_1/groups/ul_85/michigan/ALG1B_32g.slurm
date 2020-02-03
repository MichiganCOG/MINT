#!/bin/bash
#SBATCH --job-name="ALG1B_CIFAR10_VGG16_BN_32g_RETRAIN"
#SBATCH -N 1
#SBATCH --nodelist=fovea
#SBATCH --array=30-45
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00  # Time in Minutes
#SBATCH --output="ALG1B_CIFAR10_VGG16_BN_32g_RETRAIN.out"

export CUDA_VISIBLE_DEVICES="${SLURM_JOB_GPUS}"

echo "Start Job"

cd /z/home/madantrg/ViP/
source setup.sh 

cd /z/home/madantrg/Pruning/vgg16/

python retrain_stage3.py \
--Epoch 40 \
--Batch_size 128 \
--Lr 0.001 \
--Dataset CIFAR10 \
--Dims 10 \
--Expt_rerun 1 \
--Milestones 100 \
--Opt sgd \
--Weight_decay 0.0001 \
--Model vgg \
--Gamma 0.1 \
--Nesterov \
--Device_ids 0 \
--Retrain BASELINE_CIFAR10_VGG16_BN/0/logits_best.pkl \
--Retrain_mask BASELINE_CIFAR10_VGG16_BN/0/I_parent_32g_1b.npy \
--Labels_file BASELINE_CIFAR10_VGG16_BN/0/Labels_32g_1b.npy \
--Labels_children_file BASELINE_CIFAR10_VGG16_BN/0/Labels_children_32g_1b.npy \
--parent_key conv1.weight conv7.weight conv8.weight conv9.weight conv10.weight conv11.weight conv12.weight conv13.weight \
--children_key conv2.weight conv8.weight conv9.weight conv10.weight conv11.weight conv12.weight conv13.weight linear1.weight \
--parent_clusters 32 32 32 32 32 32 32 32 \
--children_clusters 32 32 32 32 32 32 32 32 \
--upper_prune_limit 0.85 \
--upper_prune_per 0.95 \
--lower_prune_per 0.1 \
--prune_per_step 0.012 \
--Save_dir BASELINE_CIFAR10_VGG16_BN_RETRAIN_1 \
--key_id $SLURM_ARRAY_TASK_ID

echo "end"
