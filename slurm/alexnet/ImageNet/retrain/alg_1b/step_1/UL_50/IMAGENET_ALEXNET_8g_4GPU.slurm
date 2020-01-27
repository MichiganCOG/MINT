#!/bin/bash
#SBATCH --job-name="IMAGENET_ALEXNET_BATCH_RETRAIN_8g_1b_4GPU"
#SBATCH --output="IMAGENET_ALEXNET_BATCH_RETRAIN_8g_1b_%j_4GPU.out"
#SBATCH -N 1               # number of nodes requested
#SBATCH -n 1              # total number of mpi tasks requested
#SBATCH -p gtx    # queue (partition) -- normal, development, etc.
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH -t 24:00:00         # run time (hh:mm:ss) - 1.5 hours

export CUDA_VISIBLE_DEVICES="${SLURM_JOB_GPUS}"
cd /work/07080/salimeh/maverick2/MINT/
source setup.sh
cd MINT
python retrain_stage3_speed_trial.py \
--Epoch 20 \
--Batch_size 1024 \
--Lr 0.0005 \
--Dataset IMAGENET \
--Model alexnet \
--Milestones 100 \
--Weight_decay 0.0004 \
--Retrain /work/07080/salimeh/maverick2/MINT/MINT/results/IMAGENET_ALEXNET_BATCH/0/logits_best.pkl \
--Retrain_mask /work/07080/salimeh/maverick2/MINT/MINT/results/IMAGENET_ALEXNET_BATCH/0/I_parent_8g_1b.npy \
--Labels_file /work/07080/salimeh/maverick2/MINT/MINT/results/IMAGENET_ALEXNET_BATCH/0/Labels_8g_1b.npy \
--Labels_children_file /work/07080/salimeh/maverick2/MINT/MINT/results/IMAGENET_ALEXNET_BATCH/0/Labels_children_8g_1b.npy \
--parent_key conv1.weight conv2.weight conv3.weight conv4.weight conv5.weight linear1.weight linear2.weight \
--children_key conv2.weight conv3.weight conv4.weight conv5.weight linear1.weight linear2.weight linear3.weight \
--parent_clusters 8 8 8 8 8 8 8 \
--children_clusters 8 8 8 8 8 8 10 \
--upper_prune_limit 0.50 \
--upper_prune_per 1.0 \
--lower_prune_per 0.1 \
--prune_per_step 0.1 \
--Save_dir /work/07080/salimeh/maverick2/MINT/MINT/results/IMAGENET_ALEXNET_RETRAIN_1_BATCH \
--key_id 2 \
--Dims 1000 \
--Device_ids 0 1 2 3
