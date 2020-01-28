#!/bin/bash
#SBATCH --job-name="ALG1B_MNIST_MLP_10g_PRUNE"
#SBATCH -N 1
#SBATCH --array=1-2
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --time=2000  # Time in Minutes
#SBATCH --output="ALG1B_MNIST_MLP_10g_%A_%a_PRUNE.out"

export CUDA_VISIBLE_DEVICES="${SLURM_JOB_GPUS}"

echo "Start Job"

cd /z/home/madantrg/ViP/
source setup.sh 

cd /z/home/madantrg/Pruning/mlp/

python pruning_1b.py \
--model mlp \
--dataset MNIST \
--weights_dir /z/home/madantrg/Pruning/mlp/results/MNIST_MLP_BATCH/0/ \
--cores 20 \
--dims 10 \
--key_id $SLURM_ARRAY_TASK_ID \
--samples_per_class 250 \
--parent_clusters 10 \
--children_clusters 10 \
--name_postfix 10g_1b
 
echo "end"
