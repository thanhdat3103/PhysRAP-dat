#!/bin/bash
#SBATCH --job-name=physrap_final_B
#SBATCH --account=bddu-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20g
#SBATCH --time=04:00:00
#SBATCH --output=/u/dnguyen21/physrap_upstream/slurm-%j.out

source /work/hdd/bddu/dnguyen21/venvs/physrap_vipl_gpu/bin/activate
export PHYSRAP_VIPL_DIR=/work/hdd/bddu/dnguyen21/datasets/VIPL-HR-V1/hdf5/vis
export PHYSRAP_VIPL_MANIFEST=/work/hdd/bddu/dnguyen21/datasets/VIPL-HR-V1/manifests/vipl_hr_vis_manifest.csv

cd /u/dnguyen21/physrap_upstream
mkdir -p logs

python trainer.py \
  --datasets VIPL_VIPL \
  --vipl_fold 3 \
  --save_path /work/hdd/bddu/dnguyen21/physrap_runs/week12_final_full_tta \
  --gpu 0 \
  --img_size 128 \
  --batch_size 4 \
  --num_rppg 160 \
  --epochs 10 \
  --loss_weight "[1,1]" \
  --ablation_mode full \
  --tta_num_augs 10 \
  --tta_future_steps 4 \
  --tta_save_ratio 0.8 \
  --tta_related_save_ratio 0.2 \
  --tta_teacher_alpha 0.99 \
  --tta_aug_mode all
