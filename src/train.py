"""
Stage 1: a benchmark you can actually defend.

What changed versus the original train.py:
  - fixed seeds, N runs per model, results reported as mean +/- std
  - subject-wise validation split (see data.py)
  - early stopping on val_ACCURACY with a floor of 10 epochs, instead of
    a flat 10 epochs or val_loss
  - macro precision / recall / F1 instead of weighted
    (weighted recall is mathematically identical to accuracy, so the
     original table had four columns carrying one column of information)
  - per-class F1 and confusion matrices
  - parameter counts, epochs trained, wall-clock time
  - trained models saved to models/ for the quantisation stage

Why val_accuracy and not val_loss:
    Validation here is 5 subjects the model has never seen. As training
    proceeds the model grows confident on the 16 training subjects, and
    that confidence transfers to the unseen ones - including where it is
    wrong. Cross-entropy punishes a confident wrong answer far harder
    than an unsure one, so val_loss starts rising while val_accuracy is
    still improving. Monitoring loss therefore stopped some runs at
    epoch 3-5, well before the model had finished learning.
    start_from_epoch=10 makes premature stopping structurally impossible.

Usage:
    python src/train.py --quick          # 2 min smoke test, 1 seed, 5 epochs
    python src/train.py                  # full run, 3 seeds per model
    python src/train.py --models cnn lstm --seeds 2
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from data import ACTIVITY_NAMES, ACTIVITY_SHORT, load_har, subject_split

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
CM_DIR = RESULTS_DIR / "confusion_matrices"
MODELS_DIR = REPO_ROOT / "models"

NUM_CLASSES = 6


# --------------------------------------------------------------------------
# Models - architectures are unchanged from the original submission so the
# new numbers stay directly comparable to last year's.
# --------------------------------------------------------------------------

def build_mlp(input_shape):
    return models.Sequential([
        layers.Input(shape=input_shape),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ], name="mlp")


def build_cnn(input_shape):
    return models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv1D(64, 3, activation="relu"),
        layers.Conv1D(64, 3, activation="relu"),
        layers.MaxPooling1D(2),
        layers.Dropout(0.5),
        layers.Conv1D(128, 3, activation="relu"),
        layers.GlobalAveragePooling1D(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ], name="cnn")


def build_lstm(input_shape):
    return models.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(64, return_sequences=True),
        layers.LSTM(64),
        layers.Dropout(0.5),
        layers.Dense(64, activation="relu"),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ], name="lstm")


def build_cnn_lstm(input_shape):
    return models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv1D(64, 3, activation="relu"),
        layers.MaxPooling1D(2),
        layers.LSTM(64),
        layers.Dropout(0.5),
        layers.Dense(64, activation="relu"),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ], name="cnn_lstm")


BUILDERS = {
    "mlp": build_mlp,
    "cnn": build_cnn,
    "lstm": build_lstm,
    "cnn_lstm": build_cnn_lstm,
}

DISPLAY_NAMES = {
    "mlp": "MLP", "cnn": "CNN", "lstm": "LSTM", "cnn_lstm": "CNN-LSTM",
}


# --------------------------------------------------------------------------
# One run
# --------------------------------------------------------------------------

def run_once(key, seed, Xtr, ytr, Xva, yva, Xte, yte,
             epochs, patience, batch_size, start_from_epoch):
    # Seeds python, numpy and tensorflow in one call.
    # Note: exact bit-for-bit reproducibility is not guaranteed for LSTM
    # layers on GPU (cuDNN kernels are non-deterministic). Running 3 seeds
    # and reporting the spread is what makes the comparison meaningful,
    # not the illusion of one perfectly repeatable number.
    tf.keras.utils.set_random_seed(seed)

    model = BUILDERS[key](Xtr.shape[1:])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    stopper = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        mode="max",
        patience=patience,
        start_from_epoch=start_from_epoch,
        restore_best_weights=True,
        verbose=0,
    )

    t0 = time.time()
    history = model.fit(
        Xtr, ytr,
        validation_data=(Xva, yva),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[stopper],
        verbose=0,
    )
    train_seconds = time.time() - t0

    y_pred = model.predict(Xte, batch_size=256, verbose=0).argmax(axis=1)

    acc = accuracy_score(yte, y_pred)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        yte, y_pred, average="macro", zero_division=0
    )
    per_class_f1 = f1_score(yte, y_pred, average=None, zero_division=0)
    cm = confusion_matrix(yte, y_pred, labels=list(range(NUM_CLASSES)))

    epochs_run = len(history.history["loss"])
    best_epoch = int(np.argmax(history.history["val_accuracy"])) + 1

    row = {
        "model": DISPLAY_NAMES[key],
        "seed": seed,
        "accuracy": acc,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "params": model.count_params(),
        "epochs_run": epochs_run,
        "best_epoch": best_epoch,
        "best_val_acc": float(np.max(history.history["val_accuracy"])),
        "train_seconds": round(train_seconds, 1),
    }
    for i, name in enumerate(ACTIVITY_NAMES):
        row[f"f1_{name}"] = per_class_f1[i]

    MODELS_DIR.mkdir(exist_ok=True)
    model.save(MODELS_DIR / f"{key}_seed{seed}.keras")

    return row, cm


# --------------------------------------------------------------------------
# Confusion matrix plot
# --------------------------------------------------------------------------

def plot_confusion(cm, display_name, out_path, n_seeds):
    pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100.0

    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)

    ax.set_xticks(range(NUM_CLASSES), ACTIVITY_SHORT, rotation=45, ha="right")
    ax.set_yticks(range(NUM_CLASSES), ACTIVITY_SHORT)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{display_name} - counts summed over {n_seeds} seed(s)")

    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            colour = "white" if pct[i, j] > 55 else "black"
            ax.text(j, i, f"{pct[i, j]:.1f}%\n{cm[i, j]}",
                    ha="center", va="center", fontsize=7.5, color=colour)

    fig.colorbar(im, ax=ax, label="% of true class")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(BUILDERS),
                    choices=list(BUILDERS))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--start-from-epoch", type=int, default=10,
                    help="earliest epoch at which early stopping may trigger")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--quick", action="store_true",
                    help="1 seed, 5 epochs - pipeline smoke test")
    args = ap.parse_args()

    if args.quick:
        args.seeds, args.epochs, args.patience = 1, 5, 5
        args.start_from_epoch = 0

    RESULTS_DIR.mkdir(exist_ok=True)
    CM_DIR.mkdir(parents=True, exist_ok=True)

    data = load_har()
    Xtr, ytr, Xva, yva, s_tr, s_va = subject_split(
        data["X_train"], data["y_train"], data["subj_train"],
        val_fraction=args.val_fraction, seed=42,
    )
    Xte, yte = data["X_test"], data["y_test"]

    print(f"train {Xtr.shape[0]} windows / {len(s_tr)} subjects")
    print(f"val   {Xva.shape[0]} windows / {len(s_va)} subjects -> {s_va.tolist()}")
    print(f"test  {Xte.shape[0]} windows / {len(set(data['subj_test'].tolist()))} subjects")
    print(f"subject leakage into val: {sorted(set(s_tr) & set(s_va)) or 'none'}")
    print(f"early stopping: monitor val_accuracy, patience {args.patience}, "
          f"no stop before epoch {args.start_from_epoch}, max {args.epochs}")
    print()

    rows, cms = [], {}

    for key in args.models:
        cm_total = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
        for seed in range(args.seeds):
            print(f"  {DISPLAY_NAMES[key]:<9} seed {seed} ...", end="", flush=True)
            row, cm = run_once(
                key, seed, Xtr, ytr, Xva, yva, Xte, yte,
                args.epochs, args.patience, args.batch_size,
                args.start_from_epoch,
            )
            cm_total += cm
            rows.append(row)
            print(f" acc {row['accuracy']:.4f}  macro-F1 {row['macro_f1']:.4f}"
                  f"  ({row['epochs_run']} ep, best {row['best_epoch']},"
                  f" {row['train_seconds']:.0f}s)")
        cms[key] = cm_total
        plot_confusion(cm_total, DISPLAY_NAMES[key],
                       CM_DIR / f"{key}.png", args.seeds)

    runs = pd.DataFrame(rows)
    runs.to_csv(RESULTS_DIR / "runs.csv", index=False)

    # ---- summary: mean +/- std across seeds ----
    metric_cols = ["accuracy", "macro_precision", "macro_recall", "macro_f1"]
    g = runs.groupby("model", sort=False)
    summary = pd.DataFrame({
        "params": g["params"].first(),
        "mean_epochs": g["epochs_run"].mean().round(1),
        "mean_best_epoch": g["best_epoch"].mean().round(1),
        "mean_train_s": g["train_seconds"].mean().round(1),
    })
    for col in metric_cols:
        summary[f"{col}_mean"] = g[col].mean()
        summary[f"{col}_std"] = g[col].std(ddof=1).fillna(0.0)
    summary = summary.reset_index()
    summary.to_csv(RESULTS_DIR / "summary.csv", index=False)

    per_class = g[[f"f1_{n}" for n in ACTIVITY_NAMES]].mean().round(4).reset_index()
    per_class.to_csv(RESULTS_DIR / "per_class_f1.csv", index=False)

    # ---- printed report ----
    print()
    print("=" * 84)
    print(f"RESULTS  ({args.seeds} seed(s) per model, mean +/- std)")
    print("=" * 84)
    print(f"{'Model':<10}{'Params':>9}  {'Accuracy':<17}{'Macro-F1':<17}"
          f"{'Epochs':>7}{'Best':>7}")
    print("-" * 84)
    for _, r in summary.iterrows():
        acc = f"{100*r['accuracy_mean']:.2f} +/- {100*r['accuracy_std']:.2f}"
        f1 = f"{100*r['macro_f1_mean']:.2f} +/- {100*r['macro_f1_std']:.2f}"
        print(f"{r['model']:<10}{int(r['params']):>9}  {acc:<17}{f1:<17}"
              f"{r['mean_epochs']:>7.1f}{r['mean_best_epoch']:>7.1f}")
    print("-" * 84)

    print()
    print("Per-class F1 (mean across seeds)")
    print("-" * 84)
    hdr = "".join(f"{n:>11}" for n in ACTIVITY_SHORT)
    print(f"{'Model':<10}{hdr}")
    for _, r in per_class.iterrows():
        vals = "".join(f"{100*r[f'f1_{n}']:>11.2f}" for n in ACTIVITY_NAMES)
        print(f"{r['model']:<10}{vals}")
    print("-" * 84)

    # ---- is the ranking real, or is it noise? ----
    best = summary.loc[summary["macro_f1_mean"].idxmax()]
    others = summary[summary["model"] != best["model"]]
    print()
    print(f"Best macro-F1: {best['model']} at {100*best['macro_f1_mean']:.2f}")
    for _, r in others.iterrows():
        gap = 100 * (best["macro_f1_mean"] - r["macro_f1_mean"])
        spread = 100 * max(best["macro_f1_std"], r["macro_f1_std"])
        verdict = "within seed noise" if gap < 2 * spread else "clear"
        print(f"  vs {r['model']:<9} gap {gap:5.2f} pts, "
              f"seed spread {spread:4.2f} -> {verdict}")

    print()
    print(f"written: {RESULTS_DIR/'runs.csv'}")
    print(f"         {RESULTS_DIR/'summary.csv'}")
    print(f"         {RESULTS_DIR/'per_class_f1.csv'}")
    print(f"         {CM_DIR}/*.png")
    print(f"         {MODELS_DIR}/*.keras  (inputs for the quantisation stage)")


if __name__ == "__main__":
    main()