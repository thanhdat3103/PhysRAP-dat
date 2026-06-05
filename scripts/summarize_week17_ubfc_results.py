import csv
import glob
import os
import re
import shutil
from pathlib import Path

repo = Path("/u/dnguyen21/physrap_upstream")
analysis = Path("/work/hdd/bddu/dnguyen21/physrap_runs/week17_ubfc_ctta/analysis")
analysis.mkdir(parents=True, exist_ok=True)

source_csv = Path("/work/hdd/bddu/dnguyen21/physrap_runs/week17_ubfc_source_only/analysis/ubfc_source_only_epoch9_5fold_summary.csv")
evidence_dir = analysis / "evidence_logs"
evidence_dir.mkdir(parents=True, exist_ok=True)

def parse_final_metrics_from_log(path):
    text = Path(path).read_text(errors="ignore")
    matches = re.findall(
        r"MAE:\s*([-+0-9.eE]+),\s*RMSE:\s*([-+0-9.eE]+),\s*SD:\s*([-+0-9.eE]+),\s*R:\s*([-+0-9.eE]+)",
        text,
    )
    if not matches:
        return None
    mae, rmse, sd, r = matches[-1]
    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "SD": float(sd),
        "R": float(r),
    }

def find_ctta_log(fold, aug_mode):
    candidates = sorted(
        glob.glob(str(repo / f"logs/*Ssource_fold{fold}_epoch9_seed_N*.log")),
        key=os.path.getmtime,
        reverse=True,
    )
    for path in candidates:
        text = Path(path).read_text(errors="ignore")
        if f"tta_aug_mode: {aug_mode}" in text:
            return path
    return ""

def find_ablation_log(fold, mode):
    candidates = sorted(
        glob.glob(str(repo / f"logs/*Ssource_fold{fold}_epoch9_{mode}_seed_N*.log")),
        key=os.path.getmtime,
        reverse=True,
    )
    for path in candidates:
        text = Path(path).read_text(errors="ignore")
        if f"mode={mode}" in text or f"ablation_mode: {mode}" in text:
            return path
    return candidates[0] if candidates else ""

rows = []

# Source-only baseline
if source_csv.exists():
    with source_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "setting": "source_only_epoch9",
                "aug_mode": "none",
                "ablation_mode": "none",
                "fold": int(r["fold"]),
                "MAE": float(r["MAE"]),
                "RMSE": float(r["RMSE"]),
                "SD": float(r["SD"]),
                "R": float(r["R"]),
                "N": int(r["N"]),
                "log_path": "",
            })
else:
    print(f"WARNING: missing source CSV: {source_csv}")

# CTTA full identity/all
for aug_mode in ["identity", "all"]:
    for fold in [1, 2, 3, 4, 5]:
        log_path = find_ctta_log(fold, aug_mode)
        if not log_path:
            print(f"WARNING: missing CTTA log fold={fold} aug={aug_mode}")
            continue
        m = parse_final_metrics_from_log(log_path)
        if m is None:
            print(f"WARNING: could not parse metrics from {log_path}")
            continue
        rows.append({
            "setting": f"ctta_full_{aug_mode}",
            "aug_mode": aug_mode,
            "ablation_mode": "full",
            "fold": fold,
            "MAE": m["MAE"],
            "RMSE": m["RMSE"],
            "SD": m["SD"],
            "R": m["R"],
            "N": 42,
            "log_path": log_path,
        })
        shutil.copy2(log_path, evidence_dir / Path(log_path).name)

# CTTA identity ablations
for mode in ["priors_only", "priors_pa", "priors_rs"]:
    for fold in [1, 2, 3, 4, 5]:
        log_path = find_ablation_log(fold, mode)
        if not log_path:
            print(f"WARNING: missing ablation log fold={fold} mode={mode}")
            continue
        m = parse_final_metrics_from_log(log_path)
        if m is None:
            print(f"WARNING: could not parse metrics from {log_path}")
            continue
        rows.append({
            "setting": f"ctta_identity_{mode}",
            "aug_mode": "identity",
            "ablation_mode": mode,
            "fold": fold,
            "MAE": m["MAE"],
            "RMSE": m["RMSE"],
            "SD": m["SD"],
            "R": m["R"],
            "N": 42,
            "log_path": log_path,
        })
        shutil.copy2(log_path, evidence_dir / Path(log_path).name)

out_fold_csv = analysis / "week17_ubfc_master_per_fold_results.csv"
fields = ["setting", "aug_mode", "ablation_mode", "fold", "MAE", "RMSE", "SD", "R", "N", "log_path"]
with out_fold_csv.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(sorted(rows, key=lambda x: (x["setting"], x["fold"])))

summary_rows = []
settings = sorted(set(r["setting"] for r in rows))
for setting in settings:
    subset = [r for r in rows if r["setting"] == setting]
    if len(subset) == 0:
        continue
    summary_rows.append({
        "setting": setting,
        "num_folds": len(subset),
        "macro_MAE": sum(r["MAE"] for r in subset) / len(subset),
        "macro_RMSE": sum(r["RMSE"] for r in subset) / len(subset),
        "macro_SD": sum(r["SD"] for r in subset) / len(subset),
        "macro_R": sum(r["R"] for r in subset) / len(subset),
    })

out_summary_csv = analysis / "week17_ubfc_master_macro_summary.csv"
with out_summary_csv.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["setting", "num_folds", "macro_MAE", "macro_RMSE", "macro_SD", "macro_R"])
    w.writeheader()
    w.writerows(summary_rows)

out_txt = analysis / "week17_ubfc_master_summary.txt"
with out_txt.open("w") as f:
    f.write("Week 17 UBFC-rPPG DATASET_2 master result summary\n\n")
    f.write("Macro summary:\n")
    for r in summary_rows:
        f.write(
            f"{r['setting']}: folds={r['num_folds']}, "
            f"MAE={r['macro_MAE']:.4f}, RMSE={r['macro_RMSE']:.4f}, "
            f"SD={r['macro_SD']:.4f}, R={r['macro_R']:.4f}\n"
        )

    f.write("\nPer-fold results:\n")
    for r in sorted(rows, key=lambda x: (x["setting"], x["fold"])):
        f.write(
            f"{r['setting']} fold {r['fold']}: "
            f"MAE={r['MAE']:.4f}, RMSE={r['RMSE']:.4f}, "
            f"SD={r['SD']:.4f}, R={r['R']:.4f}\n"
        )

print("wrote", out_fold_csv)
print("wrote", out_summary_csv)
print("wrote", out_txt)
print("copied evidence logs to", evidence_dir)
