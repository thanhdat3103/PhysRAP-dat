#!/bin/bash
#SBATCH --job-name=physrap_w16_eval_all_label
#SBATCH --account=bddu-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --tasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --time=01:30:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

module reset

cd /u/dnguyen21/physrap_upstream
source /work/hdd/bddu/dnguyen21/venvs/physrap_vipl_gpu/bin/activate

export PYTHONPATH=/u/dnguyen21/physrap_upstream:$PYTHONPATH
export PHYSRAP_VIPL_DIR=/work/hdd/bddu/dnguyen21/datasets/VIPL-HR-V1/hdf5/vis
export PHYSRAP_VIPL_MANIFEST=/work/hdd/bddu/dnguyen21/datasets/VIPL-HR-V1/manifests/vipl_hr_vis_manifest.csv

CKPT_DIR=/work/hdd/bddu/dnguyen21/physrap_runs/week16_source_training/source_train_10ep_fullvipl_label/0526_1337/ckpt
OUTROOT=/work/hdd/bddu/dnguyen21/physrap_runs/week16_source_training/source_train_10ep_fullvipl_label/eval_all_ckpts_label
mkdir -p "$OUTROOT"

for E in 0 1 2 3 4 5 6 7 8 9; do
  CKPT="$CKPT_DIR/rppg_estimator_stu_epoch_${E}.pth"
  OUT="$OUTROOT/epoch_${E}"
  mkdir -p "$OUT"

  echo "===== Evaluating epoch ${E} ====="

  python scripts/eval_source_ckpt_csv_week16_gtmode.py \
    --ckpt "$CKPT" \
    --out_dir "$OUT" \
    --datasets VIPL \
    --vipl_fold 3 \
    --gpu 0 \
    --img_size 128 \
    --batch_size 2 \
    --num_rppg 160 \
    --model ResNet3D \
    --eval_gt_mode label \
    --source_only_eval
done
