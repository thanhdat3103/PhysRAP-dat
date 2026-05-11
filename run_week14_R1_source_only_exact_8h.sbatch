#!/bin/bash
#SBATCH --job-name=physrap_W14_src
#SBATCH --account=bddu-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20g
#SBATCH --time=08:00:00
#SBATCH --output=/work/hdd/bddu/dnguyen21/physrap_runs/week14_source_only_exact/slurm/%x-%j.out

set -euo pipefail

cd /u/dnguyen21/physrap_upstream
source ./physrap_env.sh

RUN_ROOT=/work/hdd/bddu/dnguyen21/physrap_runs/week14_source_only_exact
SNAP_DIR=${RUN_ROOT}/snapshot_${SLURM_JOB_ID}
mkdir -p "${SNAP_DIR}"

echo "===== JOB INFO ====="
echo "JOB_ID=${SLURM_JOB_ID}"
echo "HOSTNAME=$(hostname)"
echo "DATE=$(date)"
echo "PWD=$(pwd)"

echo "===== CODE STATE ====="
git status
git rev-parse --short HEAD
git diff -- trainer.py utils/util.py datasets/rppg_datasets.py utils/engine.py > "${SNAP_DIR}/code_diff_at_submit.patch"
cp run_week14_R1_source_only_exact_8h.sbatch "${SNAP_DIR}/"

echo "===== PYTHON CHECK ====="
python -m py_compile trainer.py datasets/rppg_datasets.py utils/engine.py utils/util.py

echo "===== START TRAINING: WEEK14 R1 SOURCE-ONLY EXACT WEEK12 SETTING ====="
python -u trainer.py \
  --datasets VIPL \
  --vipl_fold 3 \
  --save_path /work/hdd/bddu/dnguyen21/physrap_runs/week14_source_only_exact/R1_source_only \
  --gpu 0 \
  --img_size 128 \
  --batch_size 4 \
  --num_rppg 160 \
  --epochs 10 \
  --loss_weight "[1,1]" \
  --source_only_eval

echo "===== FINISHED ====="
echo "DATE=$(date)"
