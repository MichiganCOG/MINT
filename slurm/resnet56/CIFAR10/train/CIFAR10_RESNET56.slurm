#!/bin/bash
#SBATCH --job-name="CIFAR10_RESNET56_BATCH"
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --partition=gtx
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=330  # Time in Minutes
#SBATCH --output="CIFAR10_RESNET56_BATCH.out"

export CUDA_VISIBLE_DEVICES="${SLURM_JOB_GPUS}"
echo "start"
cd /work/07080/salimeh/maverick2/MINT/ 
source setup.sh 
cd MINT
python train_stage1.py \
--Epoch 300 \
--Batch_size 128 \
--Lr 0.01 \
--Save_dir CIFAR10_RESNET56_BATCH \
--Dataset CIFAR10 \
--Dims 10 \
--Expt_rerun 1 \
--Milestones 150 225 \
--Opt sgd \
--Weight_decay 0.0002 \
--Model resnet \
--Gamma 0.1 \
--Device_ids 0

echo "end"
