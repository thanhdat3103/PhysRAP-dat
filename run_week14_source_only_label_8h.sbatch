#!/bin/bash
#SBATCH --job-name=physrap_W14_label
#SBATCH --account=bddu-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20g
#SBATCH --time=08:00:00
#SBATCH --output=/work/hdd/bddu/dnguyen21/physrap_runs/week14_label_eval/slurm/%x-%j.out

set -euo pipefail

cd /u/dnguyen21/physrap_upstream
source ./physrap_env.sh

python trainer.py \
  --datasets VIPL \
  --vipl_fold 3 \
  --save_path /work/hdd/bddu/dnguyen21/physrap_runs/week14_label_eval/source_only \
  --gpu 0 \
  --img_size 128 \
  --batch_size 4 \
  --num_rppg 160 \
  --epochs 10 \
  --loss_weight "[1,1]" \
  --source_only_eval \
  --eval_gt_mode label
