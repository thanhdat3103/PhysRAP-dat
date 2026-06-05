#!/bin/bash
#SBATCH --job-name=physrap_w18_ctta_psd_eval
#SBATCH --account=bddu-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --array=0-4%2
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --tasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -euo pipefail

module reset

cd /u/dnguyen21/physrap_upstream

source /work/hdd/bddu/dnguyen21/venvs/physrap_vipl_gpu/bin/activate

export PYTHONPATH=/u/dnguyen21/physrap_upstream:${PYTHONPATH:-}
export PHYSRAP_VIPL_DIR=/work/hdd/bddu/dnguyen21/datasets/VIPL-HR-V1/hdf5/vis
export PHYSRAP_VIPL_MANIFEST=/work/hdd/bddu/dnguyen21/datasets/VIPL-HR-V1/manifests/vipl_hr_vis_manifest.csv
export PHYSRAP_UBFC_DIR=/work/hdd/bddu/dnguyen21/datasets/UBFC-rPPG/hdf5_dataset2

FOLDS=(1 2 3 4 5)
FOLD=${FOLDS[$SLURM_ARRAY_TASK_ID]}

BASE=/work/hdd/bddu/dnguyen21/physrap_runs
RUNROOT=$BASE/week18_protocol_audit/ctta_identity_5fold_eval_gt_ecg_psd/fold${FOLD}
CONTINUE_LOG=source_fold${FOLD}_epoch9_identity_ecg_psd_seed
CKPT=$RUNROOT/$CONTINUE_LOG/ckpt/rppg_estimator_stu_dataset_1.pth

OUTROOT=$BASE/week18_protocol_audit/independent_eval_ctta_adapted_5fold_trainer_psd/fold${FOLD}
mkdir -p "$OUTROOT"

test -f "$CKPT"

echo "date=$(date)"
echo "host=$(hostname)"
echo "fold=$FOLD"
echo "ckpt=$CKPT"
echo "outroot=$OUTROOT"
echo "repo_commit=$(git rev-parse HEAD)"
echo "metric_impl=trainer_cal_psd_hr"

python scripts/eval_ckpt_trainer_psd_metric.py \
  --ckpt "$CKPT" \
  --out_dir "$OUTROOT" \
  --datasets UBFC \
  --vipl_fold "$FOLD" \
  --gpu 0 \
  --img_size 128 \
  --batch_size 1 \
  --num_rppg 160 \
  --model ResNet3D \
  --eval_gt_mode ecg_psd \
  --source_only_eval
