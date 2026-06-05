#!/bin/bash
#SBATCH --job-name=physrap_w18_ubfc_src_ecg
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
OUTROOT=$BASE/week18_protocol_audit/source_only_ubfc_5fold_eval_gt_ecg/fold${FOLD}
mkdir -p "$OUTROOT"

if [ "$FOLD" = "1" ]; then
  CKPT=$BASE/week17_5fold_source_training/fold1/0529_0056/ckpt/rppg_estimator_stu_epoch_9.pth
elif [ "$FOLD" = "2" ]; then
  CKPT=$BASE/week17_5fold_source_training/fold2/0529_0056/ckpt/rppg_estimator_stu_epoch_9.pth
elif [ "$FOLD" = "3" ]; then
  CKPT=$BASE/week16_source_training/source_train_10ep_fullvipl_label/0526_1337/ckpt/rppg_estimator_stu_epoch_9.pth
elif [ "$FOLD" = "4" ]; then
  CKPT=$BASE/week17_5fold_source_training/fold4/0529_0234/ckpt/rppg_estimator_stu_epoch_9.pth
elif [ "$FOLD" = "5" ]; then
  CKPT=$BASE/week17_5fold_source_training/fold5/0529_0246/ckpt/rppg_estimator_stu_epoch_9.pth
else
  echo "bad fold=$FOLD"
  exit 1
fi

test -f "$CKPT"

echo "date=$(date)"
echo "host=$(hostname)"
echo "fold=$FOLD"
echo "ckpt=$CKPT"
echo "outroot=$OUTROOT"
echo "repo_commit=$(git rev-parse HEAD)"
echo "eval_gt_mode=ecg"

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
