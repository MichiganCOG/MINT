#!/bin/bash
#SBATCH --job-name="CIFAR10_VGG16_BN_BATCH_PRUNE_7l_8g_500"
#SBATCH -N 1
#SBATCH --partition=lgns
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --time=1500  # Time in Minutes
#SBATCH --output="CIFAR10_VGG16_BN_BATCH_PRUNE_7l_8g_500.out"

export CUDA_VISIBLE_DEVICES="${SLURM_JOB_GPUS}"
echo "start"
cd /z/home/madantrg/scratch_incremental_learning/
source setup.sh 
cd /z/home/madantrg/Pruning/

python pruning_mi_global_vgg.py \
--weights_dir /z/home/madantrg/Pruning/results/CIFAR10_VGG16_BN_BATCH/0/ \
--cores 20 \
--parent_key conv1.weight conv2.weight conv3.weight conv4.weight conv5.weight conv6.weight conv7.weight conv8.weight conv9.weight conv10.weight conv11.weight conv12.weight conv13.weight linear1.weight \
--children_key conv2.weight conv3.weight conv4.weight conv5.weight conv6.weight conv7.weight conv8.weight conv9.weight conv10.weight conv11.weight conv12.weight conv13.weight linear1.weight linear3.weight \
--parent_clusters 8 8 8 8 8 8 8 8 8 8 8 8 8 8 \
--children_clusters 8 8 8 8 8 8 8 8 8 8 8 8 8 10 \
--name_postfix 7l_8g_500 \
--samples_per_class 500
 
echo "end"
