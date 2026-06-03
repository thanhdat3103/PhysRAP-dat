import argparse
import glob
import os
import subprocess
from pathlib import Path

import h5py
import numpy as np
import imageio_ffmpeg


def normalize_signal(x):
    x = x.astype(np.float32)
    x = x - np.mean(x)
    std = np.std(x)
    if std > 1e-8:
        x = x / std
    return x.astype(np.float32)


def ffmpeg_read_video(video_path, img_size):
    cmd = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-hide_banner",
        "-loglevel", "error",
        "-i", str(video_path),
        "-vf", f"scale={img_size}:{img_size}",
        "-pix_fmt", "rgb24",
        "-f", "rawvideo",
        "pipe:1",
    ]

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode("utf-8", errors="ignore"))

    frame_bytes = img_size * img_size * 3
    raw = proc.stdout
    n_frames = len(raw) // frame_bytes
    raw = raw[: n_frames * frame_bytes]

    if n_frames == 0:
        raise RuntimeError(f"No frames decoded from {video_path}")

    frames = np.frombuffer(raw, dtype=np.uint8)
    frames = frames.reshape(n_frames, img_size, img_size, 3)
    frames = frames.transpose(3, 0, 1, 2)
    return np.ascontiguousarray(frames)


def convert_subject(subject_dir, out_root, img_size):
    subject_dir = Path(subject_dir)
    subject_id = subject_dir.name

    video_path = subject_dir / "vid.avi"
    gt_path = subject_dir / "ground_truth.txt"

    if not video_path.exists():
        raise FileNotFoundError(video_path)
    if not gt_path.exists():
        raise FileNotFoundError(gt_path)

    gt = np.loadtxt(gt_path)
    ppg = gt[0].astype(np.float32)
    sensor_hr = gt[1].astype(np.float32)
    time = gt[2].astype(np.float32)

    video_data = ffmpeg_read_video(video_path, img_size)

    n = min(video_data.shape[1], len(ppg), len(sensor_hr), len(time))
    video_data = video_data[:, :n]
    ppg = ppg[:n]
    sensor_hr = sensor_hr[:n]
    time = time[:n]

    ecg_data = normalize_signal(ppg)
    gt_hr = np.float32(np.mean(sensor_hr))

    out_dir = Path(out_root) / subject_id
    out_dir.mkdir(parents=True, exist_ok=True)

    final_path = out_dir / "sample.hdf5"
    tmp_path = out_dir / "sample.hdf5.tmp"

    if tmp_path.exists():
        tmp_path.unlink()

    with h5py.File(tmp_path, "w") as f:
        f.create_dataset("video_data", data=video_data)
        f.create_dataset("ecg_data", data=ecg_data)
        f.create_dataset("gt_hr", data=gt_hr)
        f.create_dataset("sensor_hr", data=sensor_hr)
        f.create_dataset("time", data=time)
        f.attrs["dataset"] = "UBFC-rPPG DATASET_2"
        f.attrs["subject_id"] = subject_id
        f.attrs["num_frames"] = int(n)
        f.attrs["source_video"] = str(video_path)

    os.replace(tmp_path, final_path)
    return str(final_path), int(n), float(gt_hr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ubfc_root", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--subject", default="")
    parser.add_argument("--img_size", type=int, default=128)
    args = parser.parse_args()

    dataset2 = Path(args.ubfc_root) / "DATASET_2"

    if args.subject:
        subject_dirs = [dataset2 / args.subject]
    else:
        subject_dirs = sorted(glob.glob(str(dataset2 / "subject*")))

    rows = []
    for subject_dir in subject_dirs:
        out_path, n, gt_hr = convert_subject(subject_dir, args.out_root, args.img_size)
        rows.append((Path(subject_dir).name, out_path, n, gt_hr))
        print(f"converted {Path(subject_dir).name}: frames={n}, gt_hr={gt_hr:.3f}, out={out_path}", flush=True)

    manifest = Path(args.out_root) / "ubfc_dataset2_manifest.csv"
    with open(manifest, "w") as f:
        f.write("subject,hdf5_path,num_frames,gt_hr\n")
        for subject, out_path, n, gt_hr in rows:
            f.write(f"{subject},{out_path},{n},{gt_hr}\n")

    print(f"wrote manifest: {manifest}", flush=True)


if __name__ == "__main__":
    main()
