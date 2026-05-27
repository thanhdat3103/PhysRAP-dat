import os
import csv
import argparse
import copy
import torch
import numpy as np
from tqdm import tqdm

from utils.engine import build_dataset, build_model
from utils.util import cal_psd_hr, pearson_correlation_coefficient


def bpm_from_signal(signal_tensor, frame_rate):
    signal_tensor = signal_tensor.view(-1)
    psd = cal_psd_hr(signal_tensor, frame_rate, return_type='psd')
    return float((psd.view(-1).max(0)[1].detach().cpu() + 40).item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--datasets', type=str, default='VIPL')
    parser.add_argument('--vipl_fold', type=int, default=3)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--img_size', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_rppg', type=int, default=160)
    parser.add_argument('--model', type=str, default='ResNet3D')
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--loss', type=str, default='["np_loss", "ce_loss"]')
    parser.add_argument('--loss_weight', type=str, default='[1,1]')
    parser.add_argument('--optim', type=str, default='adam')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--scheduler', type=str, default='step')
    parser.add_argument('--step_size', type=int, default=50)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--save_path', type=str, default='unused')
    parser.add_argument('--save_mode', type=str, default='all')
    parser.add_argument('--eval_step', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--echo_batches', type=int, default=500)
    parser.add_argument('--tta_num_augs', type=int, default=10)
    parser.add_argument('--tta_future_steps', type=int, default=4)
    parser.add_argument('--tta_save_ratio', type=float, default=0.8)
    parser.add_argument('--tta_related_save_ratio', type=float, default=0.2)
    parser.add_argument('--tta_teacher_alpha', type=float, default=0.99)
    parser.add_argument('--ablation_mode', type=str, default='full')
    parser.add_argument('--tta_aug_mode', type=str, default='all')
    parser.add_argument('--source_only_eval', action='store_true')
    parser.add_argument('--eval_gt_mode', choices=['ecg', 'label'], default='ecg')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    gpu_id = int(args.gpu.split(',')[0])
    device = torch.device(f'cuda:{gpu_id}')
    frame_rate = 30

    print("===== BUILD DATASET =====")
    val_loaders = build_dataset(args, mode='test_all', batch_size=1)
    val_loader = val_loaders[0]
    dataset_obj = getattr(val_loader, 'dataset', None)
    data_list = getattr(dataset_obj, 'data_list', None)

    print("===== EVALUATE CKPT =====")
    print("ckpt =", args.ckpt)
    print("out_dir =", args.out_dir)
    print("num_val =", len(val_loader))
    print("eval_gt_mode =", args.eval_gt_mode)

    model_cache = {}
    video_rows = []
    clip_rows = []

    with torch.no_grad():
        for row_idx, sample_batched in enumerate(tqdm(val_loader)):
            inputs = sample_batched['video'].to(device)
            ecg = sample_batched['ecg'].to(device)
            clip_average_HR = sample_batched.get('clip_avg_hr', None)
            label_gt_values = None
            if clip_average_HR is not None:
                label_gt_values = clip_average_HR.detach().cpu().numpy().reshape(-1).astype(float)

            sample_ref = ''
            if data_list is not None and row_idx < len(data_list):
                sample_ref = str(data_list[row_idx])

            num_clip = 3
            input_len = inputs.shape[2]
            input_len = input_len - input_len % (num_clip * 4)
            clip_len = input_len // num_clip

            inputs = inputs[:, :, :input_len, :, :]
            ecg = ecg[:, :input_len]

            if clip_len not in model_cache:
                new_args = copy.deepcopy(args)
                new_args.num_rppg = clip_len
                model = build_model(new_args).to(device)
                state = torch.load(args.ckpt, map_location=device)
                model.load_state_dict(state)
                model.eval()
                model_cache[clip_len] = model

            model = model_cache[clip_len]

            gt_clip_bpms = []
            pred_clip_bpms = []

            for clip_idx in range(num_clip):
                inputs_iter = inputs[:, :, clip_idx * clip_len:(clip_idx + 1) * clip_len, :, :]
                ecg_iter = ecg[:, clip_idx * clip_len:(clip_idx + 1) * clip_len]

                outputs = model({'input_clip': inputs_iter})
                rppg = outputs['rPPG']

                if args.eval_gt_mode == 'label':
                    if label_gt_values is None or label_gt_values.size == 0:
                        raise RuntimeError("eval_gt_mode=label but clip_avg_hr is missing")
                    if label_gt_values.size == 1:
                        gt_bpm = float(label_gt_values[0])
                    elif label_gt_values.size >= num_clip:
                        gt_bpm = float(label_gt_values[min(clip_idx, label_gt_values.size - 1)])
                    else:
                        gt_bpm = float(np.mean(label_gt_values))
                else:
                    gt_bpm = bpm_from_signal(ecg_iter[0], frame_rate)
                pred_bpm = bpm_from_signal(rppg[0], frame_rate)

                gt_clip_bpms.append(gt_bpm)
                pred_clip_bpms.append(pred_bpm)

                clip_rows.append({
                    'row_idx': row_idx,
                    'clip_idx': clip_idx,
                    'sample_ref': sample_ref,
                    'clip_len': clip_len,
                    'gt_bpm': gt_bpm,
                    'pred_bpm': pred_bpm,
                    'abs_error': abs(pred_bpm - gt_bpm),
                })

            video_gt = float(np.mean(gt_clip_bpms))
            video_pred = float(np.mean(pred_clip_bpms))
            video_abs_error = abs(video_pred - video_gt)

            clip_avg_mean = ''
            if clip_average_HR is not None:
                arr = clip_average_HR.detach().cpu().numpy().reshape(-1)
                if arr.size > 0:
                    clip_avg_mean = float(np.mean(arr))

            video_rows.append({
                'row_idx': row_idx,
                'sample_ref': sample_ref,
                'input_len': input_len,
                'clip_len': clip_len,
                'gt_bpm': video_gt,
                'pred_bpm': video_pred,
                'abs_error': video_abs_error,
                'clip_avg_hr_mean': clip_avg_mean,
                'gt_clip_0': gt_clip_bpms[0],
                'gt_clip_1': gt_clip_bpms[1],
                'gt_clip_2': gt_clip_bpms[2],
                'pred_clip_0': pred_clip_bpms[0],
                'pred_clip_1': pred_clip_bpms[1],
                'pred_clip_2': pred_clip_bpms[2],
            })

    video_csv = os.path.join(args.out_dir, 'source_video_predictions.csv')
    clip_csv = os.path.join(args.out_dir, 'source_clip_predictions.csv')

    with open(video_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(video_rows[0].keys()))
        writer.writeheader()
        writer.writerows(video_rows)

    with open(clip_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(clip_rows[0].keys()))
        writer.writeheader()
        writer.writerows(clip_rows)

    gt = np.array([r['gt_bpm'] for r in video_rows], dtype=float)
    pred = np.array([r['pred_bpm'] for r in video_rows], dtype=float)
    err = np.abs(pred - gt)

    mae = float(np.mean(err))
    rmse = float(np.sqrt(np.mean((pred - gt) ** 2)))
    sd = float(np.std(pred - gt))
    r_value = float(pearson_correlation_coefficient(gt, pred))

    pred_min_hits = int(np.sum(pred <= 40.5))
    pred_low_hits = int(np.sum(pred <= 45.0))
    high_error_20 = int(np.sum(err >= 20.0))
    high_error_30 = int(np.sum(err >= 30.0))

    summary_path = os.path.join(args.out_dir, 'source_eval_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f'eval_gt_mode={args.eval_gt_mode}\n')
        f.write(f'MAE={mae}\n')
        f.write(f'RMSE={rmse}\n')
        f.write(f'SD={sd}\n')
        f.write(f'R={r_value}\n')
        f.write(f'N={len(video_rows)}\n')
        f.write(f'pred_bpm_le_40_5={pred_min_hits}\n')
        f.write(f'pred_bpm_le_45={pred_low_hits}\n')
        f.write(f'abs_error_ge_20={high_error_20}\n')
        f.write(f'abs_error_ge_30={high_error_30}\n')
        f.write(f'video_csv={video_csv}\n')
        f.write(f'clip_csv={clip_csv}\n')

    print("===== SUMMARY =====")
    print(open(summary_path).read())


if __name__ == '__main__':
    main()
