#!/bin/bash
#SBATCH --job-name="ALG1B_MNIST_MLP_550s_PRUNE"
#SBATCH --array=1-2
#SBATCH --nodelist=lgn1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --output="ALG1B_MNIST_MLP_550s_%A_%a_PRUNE.out"

export CUDA_VISIBLE_DEVICES="${SLURM_JOB_GPUS}"
echo "Start Job"

cd /z/home/madantrg/ViP/ 
source setup.sh 

cd /z/home/madantrg/Pruning/mlp

python pruning_1b.py \
--model mlp \
--dataset MNIST \
--weights_dir BASELINE_MNIST_MLP/0/ \
--cores 16 \
--dims 10 \
--key_id $SLURM_ARRAY_TASK_ID \
--samples_per_class 550 \
--parent_clusters 20 \
--children_clusters 20 \
--name_postfix 550s_1b


echo "end"
