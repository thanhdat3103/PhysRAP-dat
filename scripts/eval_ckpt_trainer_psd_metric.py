import argparse
import copy
import csv
import os

import numpy as np
import torch
from tqdm import tqdm

from utils.engine import build_dataset, build_model
from utils.util import cal_psd_hr, pearson_correlation_coefficient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--datasets", type=str, default="UBFC")
    parser.add_argument("--vipl_fold", type=int, default=1)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_rppg", type=int, default=160)
    parser.add_argument("--model", type=str, default="ResNet3D")
    parser.add_argument("--eval_gt_mode", choices=["ecg_psd", "label"], default="ecg_psd")
    parser.add_argument("--source_only_eval", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    gpu_id = int(args.gpu.split(",")[0])
    device = torch.device(f"cuda:{gpu_id}")
    frame_rate = 30

    print("===== BUILD DATASET =====")
    val_loaders = build_dataset(args, mode="test_all", batch_size=1)
    val_loader = val_loaders[0]
    dataset_obj = getattr(val_loader, "dataset", None)
    data_list = getattr(dataset_obj, "data_list", None)

    print("===== EVALUATE CKPT WITH TRAINER-STYLE PSD METRIC =====")
    print("ckpt =", args.ckpt)
    print("out_dir =", args.out_dir)
    print("num_val =", len(val_loader))
    print("eval_gt_mode =", args.eval_gt_mode)

    model_cache = {}
    video_rows = []
    clip_rows = []

    with torch.no_grad():
        for row_idx, sample_batched in enumerate(tqdm(val_loader)):
            inputs = sample_batched["video"].to(device)
            ecg = sample_batched["ecg"].to(device)
            clip_average_HR = sample_batched.get("clip_avg_hr", None)

            sample_ref = ""
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

            psd_gt_total = 0
            psd_pred_total = 0

            gt_clip_bpms = []
            pred_clip_bpms = []

            for clip_idx in range(num_clip):
                inputs_iter = inputs[:, :, clip_idx * clip_len:(clip_idx + 1) * clip_len, :, :]
                ecg_iter = ecg[:, clip_idx * clip_len:(clip_idx + 1) * clip_len]

                outputs = model({"input_clip": inputs_iter})
                rppg = outputs["rPPG"]

                psd_pred = cal_psd_hr(rppg[0], frame_rate, return_type="psd")
                pred_bpm = float((psd_pred.view(-1).max(0)[1].cpu() + 40).item())

                if args.eval_gt_mode == "label":
                    if clip_average_HR is None:
                        raise RuntimeError("eval_gt_mode=label but clip_avg_hr is missing")
                    gt_bpm = float(clip_average_HR.detach().view(-1).float().mean().cpu().item())
                else:
                    psd_gt = cal_psd_hr(ecg_iter, frame_rate, return_type="psd")
                    gt_bpm = float((psd_gt.view(-1).max(0)[1].cpu() + 40).item())

                psd_gt_total += gt_bpm
                psd_pred_total += pred_bpm

                gt_clip_bpms.append(gt_bpm)
                pred_clip_bpms.append(pred_bpm)

                clip_rows.append({
                    "row_idx": row_idx,
                    "clip_idx": clip_idx,
                    "sample_ref": sample_ref,
                    "gt_bpm": gt_bpm,
                    "pred_bpm": pred_bpm,
                    "abs_error": abs(pred_bpm - gt_bpm),
                })

            video_gt = float(psd_gt_total / num_clip)
            video_pred = float(psd_pred_total / num_clip)

            video_rows.append({
                "row_idx": row_idx,
                "sample_ref": sample_ref,
                "gt_bpm": video_gt,
                "pred_bpm": video_pred,
                "abs_error": abs(video_pred - video_gt),
                "clip_len": clip_len,
                "num_clip": num_clip,
            })

    video_csv = os.path.join(args.out_dir, "trainer_psd_video_predictions.csv")
    clip_csv = os.path.join(args.out_dir, "trainer_psd_clip_predictions.csv")

    with open(video_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(video_rows[0].keys()))
        writer.writeheader()
        writer.writerows(video_rows)

    with open(clip_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(clip_rows[0].keys()))
        writer.writeheader()
        writer.writerows(clip_rows)

    gt = np.array([r["gt_bpm"] for r in video_rows], dtype=float)
    pred = np.array([r["pred_bpm"] for r in video_rows], dtype=float)
    err = np.abs(pred - gt)

    mae = float(np.mean(err))
    rmse = float(np.sqrt(np.mean((pred - gt) ** 2)))
    sd = float(np.std(pred - gt))
    r_value = float(pearson_correlation_coefficient(gt, pred))

    summary_path = os.path.join(args.out_dir, "trainer_psd_eval_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"eval_gt_mode={args.eval_gt_mode}\n")
        f.write("metric_impl=trainer_cal_psd_hr\n")
        f.write(f"MAE={mae}\n")
        f.write(f"RMSE={rmse}\n")
        f.write(f"SD={sd}\n")
        f.write(f"R={r_value}\n")
        f.write(f"N={len(video_rows)}\n")
        f.write(f"video_csv={video_csv}\n")
        f.write(f"clip_csv={clip_csv}\n")

    print("===== SUMMARY =====")
    print(open(summary_path).read())


if __name__ == "__main__":
    main()
