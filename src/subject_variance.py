"""
How much does WHICH people you evaluate on move the score?

Stage 1 showed the largest architecture gap is 1.55 macro-F1 points.
This script asks whether that is large or small compared to the
variation between individual test subjects.

No retraining. It loads the 12 models already in models/ and scores
each one per test subject, then reports the spread.

Usage:
    python src/subject_variance.py
"""

from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score

from data import load_har

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"

DISPLAY = {"mlp": "MLP", "cnn": "CNN", "lstm": "LSTM", "cnn_lstm": "CNN-LSTM"}


def main():
    data = load_har()
    Xte, yte, subj = data["X_test"], data["y_test"], data["subj_test"]
    subjects = sorted(set(subj.tolist()))

    paths = sorted(glob.glob(str(REPO_ROOT / "models" / "*.keras")))
    if not paths:
        print("No models found. Run src/train.py first.")
        return

    rows = []
    for path in paths:
        stem = Path(path).stem                 # e.g. cnn_lstm_seed0
        arch, seed = stem.rsplit("_seed", 1)
        model = tf.keras.models.load_model(path)
        pred = model.predict(Xte, batch_size=256, verbose=0).argmax(axis=1)

        for s in subjects:
            m = subj == s
            rows.append({
                "model": DISPLAY[arch],
                "seed": int(seed),
                "subject": s,
                "n_windows": int(m.sum()),
                "accuracy": accuracy_score(yte[m], pred[m]),
                "macro_f1": f1_score(yte[m], pred[m],
                                     average="macro", zero_division=0),
            })
        print(f"  scored {stem}", flush=True)

    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(exist_ok=True)
    df.to_csv(RESULTS_DIR / "per_subject_runs.csv", index=False)

    # ---- per subject, averaged over seeds, per architecture ----
    pivot = (df.groupby(["subject", "model"])["accuracy"]
               .mean().unstack() * 100).round(2)
    order = [c for c in ["MLP", "CNN", "LSTM", "CNN-LSTM"] if c in pivot.columns]
    pivot = pivot[order]
    pivot["mean"] = pivot.mean(axis=1).round(2)
    pivot.to_csv(RESULTS_DIR / "per_subject_accuracy.csv")

    print()
    print("=" * 72)
    print("TEST ACCURACY BY SUBJECT  (%, mean over 3 seeds)")
    print("=" * 72)
    print(pivot.to_string())
    print("-" * 72)

    # ---- the comparison that matters ----
    per_subj = pivot["mean"]
    subj_spread = per_subj.max() - per_subj.min()
    subj_std = per_subj.std(ddof=1)

    arch_mean = (df.groupby("model")["accuracy"].mean() * 100)
    arch_spread = arch_mean.max() - arch_mean.min()

    print()
    print("SPREAD COMPARISON")
    print("-" * 72)
    print(f"across test SUBJECTS   : {subj_spread:6.2f} pts "
          f"(best {per_subj.idxmax()} {per_subj.max():.2f}, "
          f"worst {per_subj.idxmin()} {per_subj.min():.2f}, "
          f"std {subj_std:.2f})")
    print(f"across ARCHITECTURES   : {arch_spread:6.2f} pts "
          f"(best {arch_mean.idxmax()} {arch_mean.max():.2f}, "
          f"worst {arch_mean.idxmin()} {arch_mean.min():.2f})")
    print(f"ratio                  : {subj_spread / arch_spread:6.2f}x")
    print("-" * 72)

    # ---- is the hard/easy subject pattern consistent across models? ----
    ranks = pivot[order].rank(axis=0)
    corr = ranks.corr(method="spearman")
    print()
    print("Spearman correlation of per-subject difficulty across architectures")
    print("(high values mean all models find the same people hard)")
    print("-" * 72)
    print(corr.round(3).to_string())
    print("-" * 72)

    print()
    print(f"written: {RESULTS_DIR/'per_subject_accuracy.csv'}")
    print(f"         {RESULTS_DIR/'per_subject_runs.csv'}")


if __name__ == "__main__":
    main()