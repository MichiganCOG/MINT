#!/bin/bash
#SBATCH --job-name="ALG1B_MNIST_MLP_10g_RETRAIN"
#SBATCH -N 1
#SBATCH --array=30-71
#SBATCH --nodelist=fovea
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --time=30  # Time in Minutes
#SBATCH --output="ALG1B_MNIST_MLP_10g_%A_%a_RETRAIN.out"

export CUDA_VISIBLE_DEVICES="${SLURM_JOB_GPUS}"

echo "Start Job"

cd /z/home/madantrg/ViP/
source setup.sh 

cd /z/home/madantrg/Pruning/mlp/

python retrain_stage3.py \
--Epoch 30 \
--Batch_size 256 \
--Lr 0.1 \
--Dataset MNIST \
--Dims 10 \
--Expt_rerun 1 \
--Milestones 10 20 \
--Opt sgd \
--Weight_decay 0.0001 \
--Model mlp \
--Gamma 0.1 \
--Nesterov \
--Device_ids 0 \
--Retrain BASELINE_MNIST_MLP/0/logits_best.pkl \
--Retrain_mask BASELINE_MNIST_MLP/0/I_parent_10g_1b.npy \
--Labels_file BASELINE_MNIST_MLP/0/Labels_10g_1b.npy \
--Labels_children_file BASELINE_MNIST_MLP/0/Labels_children_10g_1b.npy \
--parent_key   fc1.weight fc2.weight \
--children_key fc2.weight fc3.weight \
--parent_clusters   10 10 \
--children_clusters 10 10 \
--upper_prune_limit 0.8 \
--upper_prune_per 0.99 \
--lower_prune_per 0.1 \
--prune_per_step 0.012 \
--Save_dir BASELINE_MNIST_MLP_RETRAIN_1 \
--key_id $SLURM_ARRAY_TASK_ID

echo "end"
