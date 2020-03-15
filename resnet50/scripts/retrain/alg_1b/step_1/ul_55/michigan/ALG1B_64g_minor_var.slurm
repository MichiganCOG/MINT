#!/bin/bash
#SBATCH --job-name="ALG1B_IMAGENET2012_RESNET50_64g_RETRAIN_MINOR_VAR"
#SBATCH --output="ALG1B_IMAGENET2012_RESNET50_64g_RETRAIN_64g_%A_%a_MINOR_VAR.out"
#SBATCH -N 1               # number of nodes requested
#SBATCH --partition=lgns27
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH -t 80:00:00         # run time (hh:mm:ss) - 1.5 hours
#SBATCH --array=1
echo ${SLURM_JOB_GPUS}
export CUDA_VISIBLE_DEVICES="${SLURM_JOB_GPUS}"
echo "Start Job"

cd /z/home/madantrg/ViP 
source setup.sh 

cd /z/home/madantrg/Pruning/resnet50

python retrain_stage3.py \
--Epoch 30 \
--Batch_size 64 \
--Lr 0.0001 \
--Dataset IMAGENET \
--Dims 1000 \
--Model resnet \
--Milestones 40 \
--Weight_decay 0.0001 \
--Gamma 0.1 \
--Device_ids 0 \
--Retrain BASELINE_IMAGENET2012_RESNET50_RETRAIN_1/0/logits_53.2144243945449.pkl \
--Retrain_mask BASELINE_IMAGENET2012_RESNET50/0/I_parent_64g_5s_1b.npy \
--Labels_file BASELINE_IMAGENET2012_RESNET50/0/Labels_64g_5s_1b.npy \
--Labels_children_file BASELINE_IMAGENET2012_RESNET50/0/Labels_children_64g_5s_1b.npy \
--parent_key conv1.weight conv2.weight conv3.weight conv4.weight conv5.weight conv6.weight conv7.weight conv8.weight conv9.weight conv10.weight \
conv11.weight conv12.weight conv13.weight conv14.weight conv15.weight conv16.weight conv17.weight conv18.weight conv19.weight conv20.weight \
conv21.weight conv22.weight conv23.weight conv24.weight conv25.weight conv26.weight conv27.weight conv28.weight conv29.weight conv30.weight \
conv31.weight conv32.weight conv33.weight conv34.weight conv35.weight conv36.weight conv37.weight conv38.weight conv39.weight conv40.weight \
conv41.weight conv42.weight conv43.weight conv44.weight conv45.weight conv46.weight conv47.weight conv48.weight conv49.weight conv50.weight \
conv51.weight conv52.weight \
--children_key  conv2.weight conv3.weight conv4.weight conv5.weight conv6.weight conv7.weight conv8.weight conv9.weight conv10.weight \
conv11.weight conv12.weight conv13.weight conv14.weight conv15.weight conv16.weight conv17.weight conv18.weight conv19.weight conv20.weight \
conv21.weight conv22.weight conv23.weight conv24.weight conv25.weight conv26.weight conv27.weight conv28.weight conv29.weight conv30.weight \
conv31.weight conv32.weight conv33.weight conv34.weight conv35.weight conv36.weight conv37.weight conv38.weight conv39.weight conv40.weight \
conv41.weight conv42.weight conv43.weight conv44.weight conv45.weight conv46.weight conv47.weight conv48.weight conv49.weight conv50.weight \
conv51.weight conv52.weight conv53.weight \
--parent_clusters 64 64 64 64 64 64 64 64 64 64 \
64 64 64 64 64 64 64 64 64 64 \
64 64 64 64 64 64 64 64 64 64 \
64 64 64 64 64 64 64 64 64 64 \
64 64 64 64 64 64 64 64 64 64 \
64 64 \
--children_clusters 64 64 64 64 64 64 64 64 64 64 \
64 64 64 64 64 64 64 64 64 64 \
64 64 64 64 64 64 64 64 64 64 \
64 64 64 64 64 64 64 64 64 64 \
64 64 64 64 64 64 64 64 64 64 \
64 64 \
--upper_prune_limit 0.55 \
--upper_prune_per 0.95 \
--lower_prune_per 0.10 \
--prune_per_step 0.01 \
--Save_dir BASELINE_IMAGENET2012_RESNET50_RETRAIN_4 \
--key_id $SLURM_ARRAY_TASK_ID 

echo "End Job"
