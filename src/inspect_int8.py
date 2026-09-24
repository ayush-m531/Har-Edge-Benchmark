"""
Verify the INT8 TFLite models are fully integer, and measure graph size.

For every models_tflite/*_int8.tflite it records the total tensor count,
how many tensors are float32, input/output dtypes, and file size.
Tensor count is the direct measure of loop unrolling overhead.

Usage:
    python src/inspect_int8.py
"""

from __future__ import annotations

import glob
import os
import warnings
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import tensorflow as tf

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"

DISPLAY = {"mlp": "MLP", "cnn": "CNN", "lstm": "LSTM", "cnn_lstm": "CNN-LSTM"}
# 128 timesteps x 2 stacked layers = 256; CNN-LSTM pools 126 -> 63, 1 layer
UNROLLED_CELLS = {"MLP": 0, "CNN": 0, "LSTM": 256, "CNN-LSTM": 63}


def main():
    paths = sorted(glob.glob(str(REPO_ROOT / "models_tflite" / "*_int8.tflite")))
    if not paths:
        print("No INT8 models found. Run src/evaluate_tflite.py first.")
        return

    rows = []
    for p in paths:
        stem = Path(p).stem.replace("_int8", "")
        arch, seed = stem.rsplit("_seed", 1)
        interp = tf.lite.Interpreter(model_path=p)
        interp.allocate_tensors()
        tensors = interp.get_tensor_details()
        rows.append({
            "model": DISPLAY[arch],
            "seed": int(seed),
            "tensors": len(tensors),
            "float32_tensors": sum(t["dtype"] == np.float32 for t in tensors),
            "input_dtype": interp.get_input_details()[0]["dtype"].__name__,
            "output_dtype": interp.get_output_details()[0]["dtype"].__name__,
            "size_kb": round(os.path.getsize(p) / 1024, 1),
        })

    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(exist_ok=True)
    df.to_csv(RESULTS_DIR / "int8_graph_check_runs.csv", index=False)

    g = df.groupby("model", sort=False)
    summary = pd.DataFrame({
        "tensors": g["tensors"].first(),
        "float32_tensors_max": g["float32_tensors"].max(),
        "size_kb_mean": g["size_kb"].mean().round(1),
    }).reset_index()
    summary["unrolled_cells"] = summary["model"].map(UNROLLED_CELLS)
    summary["tensors_per_cell"] = [
        round(t / c, 1) if c else None
        for t, c in zip(summary["tensors"], summary["unrolled_cells"])
    ]
    summary.to_csv(RESULTS_DIR / "int8_graph_check.csv", index=False)

    print(df.to_string(index=False))
    print()
    print(summary.to_string(index=False))

    t = dict(zip(summary["model"], summary["tensors"]))
    if "LSTM" in t and "CNN-LSTM" in t:
        print()
        print(f"tensor ratio LSTM / CNN-LSTM : {t['LSTM'] / t['CNN-LSTM']:.2f}")
        print(f"cell ratio   256 / 63        : {256 / 63:.2f}")

    print()
    print(f"written: {RESULTS_DIR/'int8_graph_check.csv'}")
    print(f"         {RESULTS_DIR/'int8_graph_check_runs.csv'}")


if __name__ == "__main__":
    main()
