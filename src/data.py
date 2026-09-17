"""
UCI HAR data loading, normalisation and subject-wise splitting.

Two things differ from the original extractData.py:

1. Normalisation uses axis=(0, 1) instead of axis=0.
   axis=0     -> one mean per (timestep, channel) = 1152 statistics
   axis=(0,1) -> one mean per channel             = 9 statistics
   These are sliding windows, so "timestep 7" is not a meaningful
   position: one window's timestep 7 may be mid-stride, another's at
   heel strike. Per-channel statistics make no alignment assumption.

2. Validation is split by SUBJECT, not at random. Each person
   contributes many near-identical overlapping windows, so a random
   split lets the model recognise the person instead of the activity,
   which inflates validation scores and mis-times early stopping.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "UCI HAR Dataset"
CACHE_DIR = REPO_ROOT / "cache"

SIGNALS = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z",
]

ACTIVITY_NAMES = [
    "WALKING", "WALK_UP", "WALK_DOWN", "SITTING", "STANDING", "LAYING",
]

# Short labels for confusion-matrix axes
ACTIVITY_SHORT = ["WALK", "UP", "DOWN", "SIT", "STAND", "LAY"]


def _read_whitespace(path: Path) -> np.ndarray:
    return pd.read_csv(path, sep=r"\s+", header=None).to_numpy(dtype=np.float32)


def _load_split(split: str):
    """Returns X (n, 128, 9), y (n,) in 0..5, subjects (n,)."""
    sig_dir = DATA_ROOT / split / "Inertial Signals"
    channels = [_read_whitespace(sig_dir / f"{s}_{split}.txt") for s in SIGNALS]
    X = np.stack(channels, axis=-1)  # (samples, 128, 9)

    y = _read_whitespace(DATA_ROOT / split / f"y_{split}.txt").ravel()
    y = y.astype(np.int64) - 1  # dataset labels are 1..6, we want 0..5

    subj = _read_whitespace(DATA_ROOT / split / f"subject_{split}.txt").ravel()
    subj = subj.astype(np.int64)

    return X, y, subj


def load_har(use_cache: bool = True):
    """
    Load the official UCI HAR split, normalised with TRAIN statistics only.

    Returns a dict with X_train, y_train, subj_train, X_test, y_test,
    subj_test, plus the mean/std used (needed later for quantisation
    calibration).
    """
    CACHE_DIR.mkdir(exist_ok=True)
    cache_file = CACHE_DIR / "har_cache.npz"

    if use_cache and cache_file.exists():
        z = np.load(cache_file)
        return {k: z[k] for k in z.files}

    if not DATA_ROOT.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_ROOT}. Run scripts/download_data.sh first."
        )

    X_train, y_train, subj_train = _load_split("train")
    X_test, y_test, subj_test = _load_split("test")

    # Normalise per sensor channel, using TRAIN statistics for both splits.
    mean = X_train.mean(axis=(0, 1), keepdims=True)  # (1, 1, 9)
    std = X_train.std(axis=(0, 1), keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)  # guard against a dead channel

    X_train = ((X_train - mean) / std).astype(np.float32)
    X_test = ((X_test - mean) / std).astype(np.float32)

    out = dict(
        X_train=X_train, y_train=y_train, subj_train=subj_train,
        X_test=X_test, y_test=y_test, subj_test=subj_test,
        norm_mean=mean.astype(np.float32), norm_std=std.astype(np.float32),
    )

    np.savez_compressed(cache_file, **out)
    return out


def subject_split(X, y, subjects, val_fraction: float = 0.2, seed: int = 42):
    """
    Hold out whole subjects for validation.

    GroupShuffleSplit guarantees no subject appears in both halves, so
    validation accuracy measures generalisation to a new person - the
    same thing the official test set measures.
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    train_idx, val_idx = next(gss.split(X, y, groups=subjects))
    return (
        X[train_idx], y[train_idx],
        X[val_idx], y[val_idx],
        np.unique(subjects[train_idx]), np.unique(subjects[val_idx]),
    )


def describe(d: dict) -> None:
    """Print the dataset facts worth being able to quote in an interview."""
    tr_subj = set(d["subj_train"].tolist())
    te_subj = set(d["subj_test"].tolist())

    print("=" * 62)
    print("UCI HAR - official split")
    print("=" * 62)
    print(f"train windows : {len(d['y_train']):>6}   shape {d['X_train'].shape}")
    print(f"test  windows : {len(d['y_test']):>6}   shape {d['X_test'].shape}")
    print(f"train subjects: {len(tr_subj):>6}  {sorted(tr_subj)}")
    print(f"test  subjects: {len(te_subj):>6}  {sorted(te_subj)}")
    print(f"subject overlap: {sorted(tr_subj & te_subj) or 'NONE (subject-independent)'}")
    print()
    print("test-set class balance:")
    for i, name in enumerate(ACTIVITY_NAMES):
        n = int((d["y_test"] == i).sum())
        print(f"  {name:<12} {n:>5}  {100 * n / len(d['y_test']):5.1f}%")
    print()
    print(f"normalisation mean per channel: {d['norm_mean'].ravel().round(4)}")
    print(f"normalisation std  per channel: {d['norm_std'].ravel().round(4)}")
    print("=" * 62)


if __name__ == "__main__":
    data = load_har()
    describe(data)

    Xtr, ytr, Xva, yva, s_tr, s_va = subject_split(
        data["X_train"], data["y_train"], data["subj_train"]
    )
    print()
    print(f"grouped split -> train {Xtr.shape[0]} windows from {len(s_tr)} subjects")
    print(f"                 val   {Xva.shape[0]} windows from {len(s_va)} subjects")
    print(f"val subjects: {s_va.tolist()}")
    print(f"leakage check (should be empty): {sorted(set(s_tr) & set(s_va))}")
