
from dataclasses import dataclass
from typing import List, Optional
import os, glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset

from .config import ALL_COLUMNS, FALL_ACTIVITIES, SEQ_LEN, STEP

def window_array(arr: np.ndarray, seq_len: int, step: int) -> np.ndarray:
    n = len(arr)
    if n < seq_len:
        return np.empty((0, seq_len, arr.shape[1]), dtype=float)
    starts = range(0, n - seq_len + 1, step)
    return np.stack([arr[s:s+seq_len] for s in starts])

def majority_label(labels: np.ndarray) -> int:
    ones = np.sum(labels); zeros = len(labels) - ones
    return 1 if ones >= zeros else 0

def load_file(csv_path: str, activity: str) -> Optional[pd.DataFrame]:
    try:
        # Read CSV with its own header
        df = pd.read_csv(csv_path)

        # Normalize column names
        df.columns = (
            df.columns.str.strip()
                      .str.lower()
                      .str.replace(" ", "_")
        )

        # Drop the non-numeric TIME column if present
        if "time" in df.columns:
            df = df.drop(columns=["time"])

        # Convert all remaining columns to numeric
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        df = df.dropna()

        # Add label (1 = fall, 0 = ADL)
        df["label"] = 1 if activity in FALL_ACTIVITIES else 0
        return df

    except Exception as e:
        print(f"[WARN] Failed to parse {csv_path}: {e}")
        return None



def collect_trials(data_root: str):
    triples = []
    for sdir in sorted(glob.glob(os.path.join(data_root, "Subject_*"))):
        for adir in sorted(glob.glob(os.path.join(sdir, "A*"))):
            act = os.path.basename(adir)
            for f in sorted(glob.glob(os.path.join(adir, "*.csv"))):
                triples.append((sdir, act, f))
    return triples

@dataclass
class WindowedData:
    X: np.ndarray  # (N, T, F)
    y: np.ndarray  # (N,)
    subjects: List[str]
    activities: List[str]
    files: List[str]

def build_windowed_dataset(data_root: str,
                           feature_cols: List[str],
                           seq_len: int = SEQ_LEN,
                           step: int = STEP) -> WindowedData:
    triples = collect_trials(data_root)
    X_list, y_list, subs, acts, files = [], [], [], [], []
    for sdir, act, fpath in triples:
        df = load_file(fpath, act)
        if df is None: continue
        feat = df[feature_cols].values.astype(float)
        labels = df['label'].values.astype(int)
        Xw = window_array(feat, seq_len, step)
        if Xw.shape[0] == 0: continue
        yw = []
        starts = range(0, len(labels) - seq_len + 1, step)
        for s in starts: yw.append(majority_label(labels[s:s+seq_len]))
        yw = np.array(yw, dtype=int)
        X_list.append(Xw); y_list.append(yw)
        subs += [os.path.basename(sdir)] * len(yw)
        acts += [act] * len(yw)
        files += [os.path.basename(fpath)] * len(yw)
    X = np.vstack(X_list) if X_list else np.empty((0, seq_len, len(feature_cols)))
    y = np.concatenate(y_list) if y_list else np.empty((0,), dtype=int)
    return WindowedData(X=X, y=y, subjects=subs, activities=acts, files=files)

class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, scaler: Optional[StandardScaler] = None):
        if scaler is None:
            self.scaler = StandardScaler().fit(X.reshape(-1, X.shape[-1]))
        else:
            self.scaler = scaler
        X_scaled = self.scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        self.X = torch.tensor(X_scaled, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]
