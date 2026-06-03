#!/bin/bash
#SBATCH --job-name=physrap_w17_ubfc_h5
#SBATCH --account=bddu-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
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

export PYTHONPATH=/u/dnguyen21/physrap_upstream:${PYTHONPATH:-}

UBFCROOT=/work/hdd/bddu/dnguyen21/datasets/UBFC-rPPG
OUTROOT=/work/hdd/bddu/dnguyen21/datasets/UBFC-rPPG/hdf5_dataset2
LOGDIR=/work/hdd/bddu/dnguyen21/physrap_runs/week17_ubfc_conversion

mkdir -p "$OUTROOT"
mkdir -p "$LOGDIR"

echo "date=$(date)"
echo "host=$(hostname)"
echo "repo_commit=$(git rev-parse HEAD)"
echo "UBFCROOT=$UBFCROOT"
echo "OUTROOT=$OUTROOT"
echo "ffmpeg=$(python - <<'PY'
import imageio_ffmpeg
print(imageio_ffmpeg.get_ffmpeg_exe())
PY
)"

python scripts/ubfc_dataset2_to_hdf5_ffmpeg.py \
  --ubfc_root "$UBFCROOT" \
  --out_root "$OUTROOT" \
  --img_size 128

echo "done date=$(date)"
