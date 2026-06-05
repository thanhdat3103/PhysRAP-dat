#!/bin/bash
#SBATCH --job-name=physrap_w18_eval_ctta_ecg
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
CKPT_DIR=$RUNROOT/$CONTINUE_LOG/ckpt

OUTROOT=$BASE/week18_protocol_audit/independent_eval_ctta_identity_adapted_5fold_ecg/fold${FOLD}
mkdir -p "$OUTROOT"

echo "date=$(date)"
echo "host=$(hostname)"
echo "fold=$FOLD"
echo "runroot=$RUNROOT"
echo "ckpt_dir=$CKPT_DIR"
echo "outroot=$OUTROOT"
echo "repo_commit=$(git rev-parse HEAD)"

echo "=== checkpoint candidates ==="
find "$CKPT_DIR" -maxdepth 1 -type f -name "*.pth" -print -exec ls -lh {} \;

if [ -f "$CKPT_DIR/rppg_estimator_stu_dataset_1.pth" ]; then
  CKPT="$CKPT_DIR/rppg_estimator_stu_dataset_1.pth"
elif [ -f "$CKPT_DIR/rppg_estimator_student_dataset_1.pth" ]; then
  CKPT="$CKPT_DIR/rppg_estimator_student_dataset_1.pth"
else
  CKPT=$(find "$CKPT_DIR" -maxdepth 1 -type f -name "*stu*dataset_1*.pth" | sort | tail -n 1)
fi

if [ -z "${CKPT:-}" ] || [ ! -f "$CKPT" ]; then
  echo "ERROR: could not find adapted student dataset_1 checkpoint"
  exit 1
fi

echo "selected_ckpt=$CKPT"

python scripts/eval_source_ckpt_csv_week16_gtmode.py \
  --ckpt "$CKPT" \
  --out_dir "$OUTROOT" \
  --datasets UBFC \
  --vipl_fold "$FOLD" \
  --gpu 0 \
  --img_size 128 \
  --batch_size 1 \
  --num_rppg 160 \
  --model ResNet3D \
  --eval_gt_mode ecg \
  --source_only_eval
