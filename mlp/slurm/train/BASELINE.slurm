#!/bin/bash
#SBATCH --job-name="BASELINE_MNIST_MLP"
#SBATCH -N 1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=330  # Time in Minutes
#SBATCH --output="BASELINE_MNIST_MLP.out"

export CUDA_VISIBLE_DEVICES="${SLURM_JOB_GPUS}"
echo "Start Job"

cd /z/home/madantrg/ViP/
source setup.sh 

cd /z/home/madantrg/Pruning/mlp/

python train_stage1.py \
--Epoch 30 \
--Batch_size 256 \
--Lr 0.1 \
--Save_dir BASELINE_MNIST_MLP \
--Dataset MNIST \
--Dims 10 \
--Expt_rerun 1 \
--Milestones 10 20 \
--Opt sgd \
--Weight_decay 0.0001 \
--Model mlp \
--Gamma 0.1 \
--Nesterov \
--Device_ids 0

echo "End Job"
