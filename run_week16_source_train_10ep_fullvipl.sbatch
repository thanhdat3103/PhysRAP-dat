#!/bin/bash
#SBATCH --job-name=physrap_w16_src10
#SBATCH --account=bddu-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --tasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

module reset

cd /u/dnguyen21/physrap_upstream

source /work/hdd/bddu/dnguyen21/venvs/physrap_vipl_gpu/bin/activate

export PYTHONPATH=/u/dnguyen21/physrap_upstream:$PYTHONPATH
export PHYSRAP_VIPL_DIR=/work/hdd/bddu/dnguyen21/datasets/VIPL-HR-V1/hdf5/vis
export PHYSRAP_VIPL_MANIFEST=/work/hdd/bddu/dnguyen21/datasets/VIPL-HR-V1/manifests/vipl_hr_vis_manifest.csv

RUNROOT=/work/hdd/bddu/dnguyen21/physrap_runs/week16_source_training/source_train_10ep_fullvipl_label
mkdir -p "$RUNROOT"
mkdir -p logs

echo "date=$(date)"
echo "host=$(hostname)"
echo "repo_commit=$(git rev-parse HEAD)"
echo "PHYSRAP_VIPL_DIR=$PHYSRAP_VIPL_DIR"
echo "PHYSRAP_VIPL_MANIFEST=$PHYSRAP_VIPL_MANIFEST"

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

python trainer.py \
  --datasets VIPL \
  --vipl_fold 3 \
  --save_path "$RUNROOT" \
  --gpu 0 \
  --img_size 128 \
  --batch_size 2 \
  --num_rppg 160 \
  --epochs 10 \
  --eval_gt_mode label \
  --source_only_eval
