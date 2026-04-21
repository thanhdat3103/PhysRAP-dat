import os
import csv
import cv2
import h5py
import argparse
import numpy as np


def load_single_column_csv(path: str) -> np.ndarray:
    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    if np.isscalar(data):
        data = np.array([data], dtype=np.float32)
    data = np.asarray(data).reshape(-1)
    if data.size == 0 or np.isnan(data).all():
        data = np.genfromtxt(path, skip_header=1)
        if np.isscalar(data):
            data = np.array([data], dtype=np.float32)
        data = np.asarray(data).reshape(-1)
    return data.astype(np.float32)


def read_video(video_path: str, out_h: int, out_w: int, target_fps: float = 30.0):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1e-6:
        fps = target_fps

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_CUBIC)
        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        raise RuntimeError(f"No frames read from {video_path}")

    video = np.stack(frames, axis=0)  # (T, H, W, C)

    if abs(fps - target_fps) > 1e-6:
        target_len = max(1, int(round(len(video) * target_fps / fps)))
        new_x = np.linspace(0, len(video) - 1, target_len)
        idx = np.clip(np.round(new_x).astype(int), 0, len(video) - 1)
        video = video[idx]

    return video.astype(np.uint8), float(fps)


def resample_signal(sig: np.ndarray, target_len: int) -> np.ndarray:
    if len(sig) == target_len:
        return sig.astype(np.float32)
    old_x = np.linspace(0, len(sig) - 1, len(sig))
    new_x = np.linspace(0, len(sig) - 1, target_len)
    out = np.interp(new_x, old_x, sig)
    return out.astype(np.float32)


def convert_one(row, output_root, h, w, target_fps):
    sample_id = row["sample_id"].strip()
    video_path = row["video_path"].strip()
    wave_path = row["wave_path"].strip()
    hr_path = row.get("hr_path", "").strip()

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"missing video: {video_path}")
    if not os.path.exists(wave_path):
        raise FileNotFoundError(f"missing wave: {wave_path}")

    video, src_fps = read_video(video_path, h, w, target_fps)
    wave = load_single_column_csv(wave_path)
    wave = resample_signal(wave, len(video))

    sample_out_dir = os.path.join(output_root, sample_id)
    os.makedirs(sample_out_dir, exist_ok=True)
    h5_path = os.path.join(sample_out_dir, "sample.hdf5")

    with h5py.File(h5_path, "w") as f:
        f.create_dataset("video_data", data=video.transpose(3, 0, 1, 2), compression="gzip")
        f.create_dataset("ecg_data", data=wave, compression="gzip")
        if hr_path and os.path.exists(hr_path):
            gt_hr = load_single_column_csv(hr_path)
            gt_hr = gt_hr[np.isfinite(gt_hr)]
            if gt_hr.size > 0:
                f.create_dataset("gt_hr", data=np.array(gt_hr.mean(), dtype=np.float32))

    return h5_path, video.shape[0], src_fps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--h", type=int, default=128)
    parser.add_argument("--w", type=int, default=128)
    parser.add_argument("--target_fps", type=float, default=30.0)
    parser.add_argument("--limit", type=int, default=0, help="0 means no limit")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--fail_csv", default="vipl_bulk_failures.csv")
    args = parser.parse_args()

    with open(args.input_csv, "r", newline="") as f:
        rows = list(csv.DictReader(f))

    if args.limit > 0:
        rows = rows[:args.limit]

    ok = 0
    skipped = 0
    failed = []

    for idx, row in enumerate(rows, start=1):
        sample_id = row["sample_id"].strip()
        out_h5 = os.path.join(args.output_root, sample_id, "sample.hdf5")

        if args.skip_existing and os.path.exists(out_h5):
            skipped += 1
            print(f"[{idx}/{len(rows)}] skip existing: {sample_id}")
            continue

        try:
            h5_path, num_frames, src_fps = convert_one(
                row=row,
                output_root=args.output_root,
                h=args.h,
                w=args.w,
                target_fps=args.target_fps,
            )
            ok += 1
            print(f"[{idx}/{len(rows)}] OK {sample_id} -> {h5_path} | frames={num_frames} src_fps={src_fps}")
        except Exception as e:
            failed.append({
                "sample_id": sample_id,
                "video_path": row.get("video_path", ""),
                "wave_path": row.get("wave_path", ""),
                "hr_path": row.get("hr_path", ""),
                "error": str(e),
            })
            print(f"[{idx}/{len(rows)}] FAIL {sample_id}: {e}")

    if failed:
        with open(args.fail_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=failed[0].keys())
            writer.writeheader()
            writer.writerows(failed)

    print("========== SUMMARY ==========")
    print("total rows   :", len(rows))
    print("ok           :", ok)
    print("skipped      :", skipped)
    print("failed       :", len(failed))
    if failed:
        print("fail csv     :", args.fail_csv)


if __name__ == "__main__":
    main()