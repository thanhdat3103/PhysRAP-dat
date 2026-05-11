#!/bin/bash
#SBATCH --job-name=physrap_W14_ttasrc
#SBATCH --account=bddu-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20g
#SBATCH --time=01:00:00
#SBATCH --output=/work/hdd/bddu/dnguyen21/physrap_runs/week14_diagnostics/slurm/%x-%j.out

set -euo pipefail

cd /u/dnguyen21/physrap_upstream
source ./physrap_env.sh
export PYTHONPATH=/u/dnguyen21/physrap_upstream:${PYTHONPATH:-}

CKPT=/work/hdd/bddu/dnguyen21/physrap_runs/week14_label_tta/TTA_full_label/0505_0436/ckpt/rppg_estimator_stu_epoch_9.pth

echo "===== JOB INFO ====="
echo "JOB_ID=${SLURM_JOB_ID}"
echo "HOSTNAME=$(hostname)"
echo "CKPT=${CKPT}"

python -u scripts/eval_source_ckpt_csv.py \
  --ckpt "${CKPT}" \
  --out_dir /work/hdd/bddu/dnguyen21/physrap_runs/week14_diagnostics/tta_full_source_ckpt \
  --datasets VIPL \
  --vipl_fold 3 \
  --gpu 0 \
  --img_size 128 \
  --batch_size 4 \
  --num_rppg 160 \
  --loss_weight "[1,1]"
