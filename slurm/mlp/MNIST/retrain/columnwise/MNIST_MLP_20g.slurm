#!/bin/bash
#SBATCH --job-name="MNIST_MLP_BATCH_20g"
#SBATCH -N 1
#SBATCH --nodelist=fovea
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=7200  # Time in Minutes
#SBATCH --output="MNIST_MLP_BATCH_20g.out"

export CUDA_VISIBLE_DEVICES="${SLURM_JOB_GPUS}"
echo "start"
cd /z/home/madantrg/scratch_incremental_learning/
source setup.sh 
cd /z/home/madantrg/Pruning/

python retrain_stage4.py \
--Epoch 30 \
--Batch_size 256 \
--Lr 0.1 \
--Dataset MNIST \
--Model mlp \
--Nesterov \
--Milestones 10 20 \
--Retrain /z/home/madantrg/Pruning/results/MNIST_MLP_BATCH/0/logits_best.pkl \
--Retrain_mask /z/home/madantrg/Pruning/results/MNIST_MLP_BATCH/0/I_parent_20g.npy \
--Labels_file /z/home/madantrg/Pruning/results/MNIST_MLP_BATCH/0/Labels_20g.npy \
--Labels_children_file /z/home/madantrg/Pruning/results/MNIST_MLP_BATCH/0/Labels_children_20g.npy \
--parent_key fc1.weight fc2.weight \
--children_key fc2.weight fc3.weight \
--parent_clusters 20 20 \
--children_clusters 20 10 \
--upper_prune_limit 1.1 \
--upper_prune_per 0.9 \
--lower_prune_per 0.7 \
--prune_per_step 0.05

echo "end"
