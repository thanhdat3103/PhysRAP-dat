#!/bin/bash
#SBATCH --job-name=physrap_w17_eval5fold
#SBATCH --account=bddu-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --array=0-3%2
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --tasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -euo pipefail

module reset

cd /u/dnguyen21/physrap_upstream

source /work/hdd/bddu/dnguyen21/venvs/physrap_vipl_gpu/bin/activate

export PYTHONPATH=/u/dnguyen21/physrap_upstream:${PYTHONPATH:-}
export PHYSRAP_VIPL_DIR=/work/hdd/bddu/dnguyen21/datasets/VIPL-HR-V1/hdf5/vis
export PHYSRAP_VIPL_MANIFEST=/work/hdd/bddu/dnguyen21/datasets/VIPL-HR-V1/manifests/vipl_hr_vis_manifest.csv

FOLDS=(1 2 4 5)
FOLD=${FOLDS[$SLURM_ARRAY_TASK_ID]}

BASE=/work/hdd/bddu/dnguyen21/physrap_runs/week17_5fold_source_training
CKPT_DIR=$(find "$BASE/fold${FOLD}" -type d -name ckpt | sort | tail -n 1)
OUTROOT="$BASE/eval_all_ckpts_label/fold${FOLD}"

mkdir -p "$OUTROOT"

echo "date=$(date)"
echo "host=$(hostname)"
echo "fold=$FOLD"
echo "ckpt_dir=$CKPT_DIR"
echo "outroot=$OUTROOT"
echo "repo_commit=$(git rev-parse HEAD)"

for E in 0 1 2 3 4 5 6 7 8 9; do
  CKPT="$CKPT_DIR/rppg_estimator_stu_epoch_${E}.pth"
  OUT="$OUTROOT/epoch_${E}"
  mkdir -p "$OUT"

  echo "===== fold ${FOLD}, epoch ${E} ====="
  echo "CKPT=$CKPT"

  python scripts/eval_source_ckpt_csv_week16_gtmode.py \
    --ckpt "$CKPT" \
    --out_dir "$OUT" \
    --datasets VIPL \
    --vipl_fold "$FOLD" \
    --gpu 0 \
    --img_size 128 \
    --batch_size 2 \
    --num_rppg 160 \
    --model ResNet3D \
    --eval_gt_mode label \
    --source_only_eval
done
