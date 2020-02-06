#!/bin/bash
#SBATCH --job-name="ALG1B_MNIST_MLP_100g_500s_PRUNE"
#SBATCH --partition=lgns
#SBATCH --array=1-2
#SBATCH --cpus-per-task=16
#SBATCH --time=20:00:00
#SBATCH --output="ALG1B_MNIST_MLP_100g_500s_%A_%a_RETRAIN.out"

export CUDA_VISIBLE_DEVICES="${SLURM_JOB_GPUS}"
echo "Start Job"

cd /z/home/madantrg/ViP 
source setup.sh 

cd /z/home/madantrg/Pruning/mlp/

python pruning_1b.py \
--model mlp \
--dataset MNIST \
--weights_dir BASELINE_MNIST_MLP/0/ \
--cores 16 \
--dims 10 \
--key_id $SLURM_ARRAY_TASK_ID \
--samples_per_class 500 \
--parent_clusters 100 \
--children_clusters 100 \
--name_postfix 100g_500s_1b


echo "end"
