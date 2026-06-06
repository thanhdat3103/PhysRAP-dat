#!/bin/bash
#SBATCH --job-name=physrap_w18_trace_5f
#SBATCH --account=bddu-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --array=0-4%2
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
export PHYSRAP_UBFC_DIR=/work/hdd/bddu/dnguyen21/datasets/UBFC-rPPG/hdf5_dataset2

mkdir -p logs

FOLDS=(1 2 3 4 5)
FOLD=${FOLDS[$SLURM_ARRAY_TASK_ID]}

BASE=/work/hdd/bddu/dnguyen21/physrap_runs
RUNROOT=$BASE/week18_protocol_audit/ctta_identity_online_trace_5fold/fold${FOLD}
CONTINUE_LOG=source_fold${FOLD}_epoch9_identity_ecg_psd_trace_seed

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

mkdir -p "$RUNROOT/$CONTINUE_LOG/ckpt"
mkdir -p "$RUNROOT/$CONTINUE_LOG/rppg"

cp -f "$CKPT" "$RUNROOT/$CONTINUE_LOG/ckpt/rppg_estimator_stu_dataset_0.pth"
cp -f "$CKPT" "$RUNROOT/$CONTINUE_LOG/ckpt/rppg_estimator_teacher_dataset_0.pth"

echo "date=$(date)"
echo "host=$(hostname)"
echo "fold=$FOLD"
echo "ckpt=$CKPT"
echo "runroot=$RUNROOT"
echo "continue_log=$CONTINUE_LOG"
echo "repo_commit=$(git rev-parse HEAD)"

export FOLD RUNROOT CONTINUE_LOG
touch "$RUNROOT/ctta_start_marker.txt"

python - <<'PY'
import os
from types import SimpleNamespace
from trainer import RppgEstimatorTrainer, set_seed

fold = int(os.environ["FOLD"])
runroot = os.environ["RUNROOT"]
continue_log = os.environ["CONTINUE_LOG"]

args = SimpleNamespace(
    datasets="VIPL_UBFC",
    vipl_fold=fold,
    gpu="0",
    img_size=128,
    batch_size=2,
    num_rppg=160,
    model="ResNet3D",
    dropout=0.2,
    loss='["np_loss", "ce_loss"]',
    loss_weight='[0,1]',
    optim="adam",
    lr=1e-4,
    beta1=0.9,
    beta2=0.999,
    weight_decay=5e-5,
    momentum=0.9,
    scheduler="step",
    step_size=50,
    gamma=0.1,
    save_path=runroot,
    save_mode="all",
    eval_step=1,
    epochs=10,
    echo_batches=500,
    tta_num_augs=10,
    tta_future_steps=1,
    tta_save_ratio=0.8,
    tta_related_save_ratio=0.2,
    tta_teacher_alpha=0.99,
    ablation_mode="full",
    tta_aug_mode="identity",
    eval_gt_mode="ecg_psd",
    source_only_eval=False,
)

set_seed(92)
trainer = RppgEstimatorTrainer(args)
trainer.train(start_dataset_idx=1, continue_log=continue_log)
PY
