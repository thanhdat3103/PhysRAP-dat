import os
import cv2
import h5py
import argparse
import numpy as np


def load_single_column_csv(path):
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


def read_video(video_path, out_h, out_w, target_fps=30.0):
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
        old_x = np.linspace(0, len(video) - 1, len(video))
        new_x = np.linspace(0, len(video) - 1, target_len)
        idx = np.clip(np.round(new_x).astype(int), 0, len(video) - 1)
        video = video[idx]

    return video.astype(np.uint8), float(fps)


def resample_signal(sig, target_len):
    if len(sig) == target_len:
        return sig.astype(np.float32)
    old_x = np.linspace(0, len(sig) - 1, len(sig))
    new_x = np.linspace(0, len(sig) - 1, target_len)
    out = np.interp(new_x, old_x, sig)
    return out.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--h", type=int, default=128)
    parser.add_argument("--w", type=int, default=128)
    parser.add_argument("--target_fps", type=float, default=30.0)
    args = parser.parse_args()

    video_path = os.path.join(args.input_dir, "video.avi")
    wave_path = os.path.join(args.input_dir, "wave.csv")
    hr_path = os.path.join(args.input_dir, "gt_HR.csv")

    video, src_fps = read_video(video_path, args.h, args.w, args.target_fps)
    wave = load_single_column_csv(wave_path)
    wave = resample_signal(wave, len(video))

    os.makedirs(args.output_dir, exist_ok=True)
    h5_path = os.path.join(args.output_dir, "sample.hdf5")

    with h5py.File(h5_path, "w") as f:
        f.create_dataset("video_data", data=video.transpose(3, 0, 1, 2), compression="gzip")
        f.create_dataset("ecg_data", data=wave, compression="gzip")
        if os.path.exists(hr_path):
            gt_hr = load_single_column_csv(hr_path)
            gt_hr = gt_hr[np.isfinite(gt_hr)]
            if gt_hr.size > 0:
                f.create_dataset("gt_hr", data=np.array(gt_hr.mean(), dtype=np.float32))

    print("saved:", h5_path)
    print("video_data shape:", video.transpose(3, 0, 1, 2).shape)
    print("ecg_data shape:", wave.shape)
    print("source fps:", src_fps, "target fps:", args.target_fps)


if __name__ == "__main__":
    main()