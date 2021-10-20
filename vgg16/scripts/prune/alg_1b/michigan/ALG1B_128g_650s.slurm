#!/bin/bash
#SBATCH --job-name="BASELINE_CIFAR10_VGG16_BN_128g_200s_PRUNE"
#SBATCH --constraint=pascal
#SBATCH --array=4-6
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --time=1000:00:00  # Time in Minutes
#SBATCH --output="BASELINE_CIFAR10_VGG16_BN_128g_200s_%A_%a_PRUNE.out"

export CUDA_VISIBLE_DEVICES="${SLURM_JOB_GPUS}"
echo "Start Job"

cd /z/home/madantrg/ViP/
source setup.sh 

cd /z/home/madantrg/MINT/vgg16/

python pruning_1b.py \
--model vgg \
--dataset CIFAR10 \
--weights_dir BASELINE_CIFAR10_VGG16_BN/0/ \
--cores 10 \
--dims 10 \
--key_id $SLURM_ARRAY_TASK_ID \
--samples_per_class 650 \
--parent_clusters 128 \
--children_clusters 128 \
--name_postfix 128g_650s_1b


echo "end"
