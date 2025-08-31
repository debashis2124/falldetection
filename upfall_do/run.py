import os, json
import numpy as np
import pandas as pd
from collections import defaultdict

from .config import (DATA_ROOT, OUT_DIR, MODALITIES, COMBINATIONS, ROUNDS)
from .data import load_file, WindowedData, window_array, majority_label
from .train import train_eval_deep_for_modality

# -----------------
# Plot imports & styling helpers
# -----------------
import matplotlib.pyplot as plt
import itertools

# Distinct color cycle (import itertools BEFORE using it)
COLOR_CYCLE = itertools.cycle([
    "tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
    "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan",
    "#e6194b","#3cb44b","#ffe119","#4363d8","#f58231",
    "#911eb4","#46f0f0","#f032e6","#bcf60c","#fabebe",
    "#008080","#e6beff","#9a6324","#fffac8","#800000"
])

# Pretty names for legends/titles
NAME_MAP = {
    "all_sensors": "sensors",
    "imu_only": "imu",
    "eeg": "eeg",
    "infrared": "infrared",
    "sensors": "sensors",
    "imu": "imu"
}

# -----------------
# Simple per-model/metric plot (used inside run_pipeline)
# -----------------
def plot_rounds(metric_series, title, out_path, rounds=ROUNDS):
    xs = list(range(1, rounds+1))
    plt.figure(figsize=(12,8))
    for label, vals in metric_series.items():
        c = next(COLOR_CYCLE)
        plt.plot(xs, vals, marker='o', linewidth=2, label=label, color=c)
    plt.xticks(xs, fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel("Round", fontsize=14, fontweight="bold")
    plt.ylabel(title.upper(), fontsize=14, fontweight="bold")
    plt.title(title, fontsize=16, fontweight="bold", pad=14)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def save_metrics_csv(metrics_dict, out_csv):
    rows = []
    for model, metr in metrics_dict.items():
        for i in range(ROUNDS):
            row = {"model": model, "round": i+1}
            for k, vals in metr.items():
                row[k] = vals[i]
            rows.append(row)
    pd.DataFrame(rows).to_csv(out_csv, index=False)

# -----------------
# Build dataset across all CSVs in each Subject/Activity folder
# -----------------
def build_dataset_by_folder(data_root, feature_cols, seq_len=16, step=8):
    triples = []
    for sdir in sorted(os.listdir(data_root)):
        subj_dir = os.path.join(data_root, sdir)
        if not os.path.isdir(subj_dir):
            continue
        for adir in sorted(os.listdir(subj_dir)):
            act_dir = os.path.join(subj_dir, adir)
            if not os.path.isdir(act_dir):
                continue
            csvs = [os.path.join(act_dir, f) for f in sorted(os.listdir(act_dir)) if f.endswith(".csv")]
            if not csvs:
                continue
            triples.append((sdir, adir, csvs))

    X_list, y_list, subs, acts, files = [], [], [], [], []
    for sdir, act, csv_list in triples:
        dfs = []
        for fpath in csv_list:
            df = load_file(fpath, act)
            if df is not None and not df.empty:
                dfs.append(df)
        if not dfs:
            continue

        df_all = pd.concat(dfs, ignore_index=True)
        feat = df_all[feature_cols].values.astype(float)
        labels = df_all["label"].values.astype(int)

        Xw = window_array(feat, seq_len, step)
        if Xw.shape[0] == 0:
            continue

        # majority label per window
        yw = []
        starts = range(0, len(labels) - seq_len + 1, step)
        for s in starts:
            yw.append(majority_label(labels[s:s+seq_len]))
        yw = np.array(yw, dtype=int)

        X_list.append(Xw)
        y_list.append(yw)
        subs += [sdir] * len(yw)
        acts += [act] * len(yw)
        files += [",".join([os.path.basename(f) for f in csv_list])] * len(yw)

    X = np.vstack(X_list) if X_list else np.empty((0, seq_len, len(feature_cols)))
    y = np.concatenate(y_list) if y_list else np.empty((0,), dtype=int)
    return WindowedData(X=X, y=y, subjects=subs, activities=acts, files=files)

# -----------------
# Run one modality
# -----------------
def run_pipeline(modality_key="all_sensors", data_root=DATA_ROOT, out_dir=OUT_DIR):
    feature_cols = COMBINATIONS.get(modality_key, MODALITIES.get(modality_key, None))
    if feature_cols is None:
        raise ValueError(f"Unknown modality_key {modality_key}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"[INFO] Building dataset for '{modality_key}' ({len(feature_cols)} features)")
    win = build_dataset_by_folder(data_root, feature_cols)
    print(f"[INFO] Windows: {win.X.shape}, positives: {int(win.y.sum())}, negatives: {int((win.y==0).sum())}")

    deep_models = ["cnn","lstm","cnn_lstm","transformer"]
    deep_results = {m: {k: [] for k in ["accuracy","precision","recall","f1","specificity","roc_auc","loss"]} for m in deep_models}

    for m in deep_models:
        print(f"[RUN] Model: {m} on {modality_key}")
        r = train_eval_deep_for_modality(win.X, win.y, m)

        model_dir = os.path.join(out_dir, f"{modality_key}_{m}")
        os.makedirs(model_dir, exist_ok=True)

        # Per-model plots
        for metric_name, series in r.items():
            plot_rounds({m: series}, f"{metric_name} ({NAME_MAP.get(modality_key, modality_key)}-{m})",
                        os.path.join(model_dir, f"{metric_name}_rounds.png"))

        for k in deep_results[m].keys():
            deep_results[m][k] = r[k]

    # Save CSV + JSON
    save_metrics_csv(deep_results, os.path.join(out_dir, f"{modality_key}_metrics.csv"))
    summary = {}
    for model, metr in deep_results.items():
        summary[model] = {k: {"mean": float(np.nanmean(v)), "std": float(np.nanstd(v))}
                          for k, v in metr.items()}
    with open(os.path.join(out_dir, f"{modality_key}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[DONE] Results saved for {modality_key} -> {out_dir}")
    return deep_results

# -----------------
# Run multiple modalities
# -----------------
def run_all_modalities(modality_keys=["all_sensors","imu_only","eeg","infrared"],
                       data_root=DATA_ROOT, out_dir=OUT_DIR):
    results = {}
    for mk in modality_keys:
        results[mk] = run_pipeline(modality_key=mk, data_root=data_root, out_dir=out_dir)
    with open(os.path.join(out_dir, "all_modalities_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    return results

# -----------------
# Combined comparison plots for all modalities + models
# -----------------
def plot_comparison(all_results, metric_name, out_path, rounds=ROUNDS):
    xs = list(range(1, rounds+1))
    plt.figure(figsize=(14,10))

    color_map = {}
    for modality, models in all_results.items():
        pretty_modality = NAME_MAP.get(modality, modality)
        for model, metrics in models.items():
            key = f"{model}-{pretty_modality}"
            if key not in color_map:
                color_map[key] = next(COLOR_CYCLE)
            vals = metrics[metric_name]
            plt.plot(xs, vals, marker='o', label=key,
                     color=color_map[key], linewidth=2)

    plt.xticks(xs, fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel("Round", fontsize=16, fontweight="bold")
    plt.ylabel(metric_name.upper(), fontsize=16, fontweight="bold")

    pretty_modalities = ", ".join([NAME_MAP.get(m, m) for m in all_results.keys()])
    plt.title(f"{metric_name.upper()} accross all models and modalities",
              fontsize=18, fontweight="bold", pad=20)

    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left',
               fontsize=12, title="Model-Modality", title_fontsize=13)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[PLOT] Saved {out_path}")

def plot_all_metrics(all_results, out_dir):
    """Generate combined plots for all metrics across all modalities and models."""
    metrics = ["accuracy","precision","recall","f1","specificity","roc_auc","loss"]
    for metric in metrics:
        plot_comparison(all_results, metric,
                        os.path.join(out_dir, f"all_{metric}_comparison.png"))