#!/bin/bash
#SBATCH --job-name=physrap_smoke_A
#SBATCH --account=bddu-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20g
#SBATCH --time=01:00:00
#SBATCH --output=/u/dnguyen21/physrap_upstream/slurm-%j.out

source /work/hdd/bddu/dnguyen21/venvs/physrap_vipl_gpu/bin/activate
export PHYSRAP_VIPL_DIR=/work/hdd/bddu/dnguyen21/datasets/VIPL-HR-V1/hdf5/vis
export PHYSRAP_VIPL_MANIFEST=/work/hdd/bddu/dnguyen21/datasets/VIPL-HR-V1/manifests/vipl_hr_vis_manifest.csv

cd /u/dnguyen21/physrap_upstream
mkdir -p logs

python trainer.py \
  --datasets VIPL \
  --vipl_fold 3 \
  --save_path /work/hdd/bddu/dnguyen21/physrap_runs/week12_smoke_source_only \
  --gpu 0 \
  --img_size 128 \
  --batch_size 4 \
  --num_rppg 160 \
  --epochs 1 \
  --loss_weight "[1,1]" \
  --source_only_eval
