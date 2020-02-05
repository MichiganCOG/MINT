#!/bin/bash
#SBATCH --job-name="ALG1B_CIFAR10_VGG16_BN_8g_PRUNE"
#SBATCH --array=1-14
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00  # Time in Minutes
#SBATCH --output="ALG1B_CIFAR10_VGG16_BN_8g_%A_%a_PRUNE.out"

export CUDA_VISIBLE_DEVICES="${SLURM_JOB_GPUS}"
echo "Start Job"

cd /z/home/madantrg/ViP/
source setup.sh 

cd /z/home/madantrg/Pruning/vgg16

python pruning_1b.py \
--model vgg \
--dataset CIFAR10 \
--weights_dir BASELINE_CIFAR10_VGG16_BN/0/ \
--cores 10 \
--dims 10 \
--key_id $SLURM_ARRAY_TASK_ID \
--parent_clusters 8 \
--children_clusters 8 \
--name_postfix 8g_1b


echo "end"
