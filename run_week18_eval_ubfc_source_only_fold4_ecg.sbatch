#!/bin/bash
#SBATCH --job-name=physrap_w18_ubfc_f4_ecg
#SBATCH --account=bddu-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --tasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

module reset

cd /u/dnguyen21/physrap_upstream

source /work/hdd/bddu/dnguyen21/venvs/physrap_vipl_gpu/bin/activate

export PYTHONPATH=/u/dnguyen21/physrap_upstream:${PYTHONPATH:-}
export PHYSRAP_VIPL_DIR=/work/hdd/bddu/dnguyen21/datasets/VIPL-HR-V1/hdf5/vis
export PHYSRAP_VIPL_MANIFEST=/work/hdd/bddu/dnguyen21/datasets/VIPL-HR-V1/manifests/vipl_hr_vis_manifest.csv
export PHYSRAP_UBFC_DIR=/work/hdd/bddu/dnguyen21/datasets/UBFC-rPPG/hdf5_dataset2

BASE=/work/hdd/bddu/dnguyen21/physrap_runs
OUTROOT=$BASE/week18_protocol_audit/source_only_ubfc_fold4_eval_gt_ecg
CKPT=$BASE/week17_5fold_source_training/fold4/0529_0234/ckpt/rppg_estimator_stu_epoch_9.pth

mkdir -p "$OUTROOT"

test -f "$CKPT"

echo "date=$(date)"
echo "host=$(hostname)"
echo "ckpt=$CKPT"
echo "outroot=$OUTROOT"
echo "repo_commit=$(git rev-parse HEAD)"
echo "eval_gt_mode=ecg"

python scripts/eval_source_ckpt_csv_week16_gtmode.py \
  --ckpt "$CKPT" \
  --out_dir "$OUTROOT" \
  --datasets UBFC \
  --vipl_fold 4 \
  --gpu 0 \
  --img_size 128 \
  --batch_size 1 \
  --num_rppg 160 \
  --model ResNet3D \
  --eval_gt_mode ecg \
  --source_only_eval
