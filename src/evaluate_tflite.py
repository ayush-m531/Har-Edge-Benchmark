"""
Stage 2 + 3: what these models actually cost to deploy.

Stage 1 told us CNN and LSTM are statistically tied on accuracy.
When accuracy ties, cost decides. This script measures cost.

For every trained model it produces:
  - FP32 TFLite size  (the honest baseline - NOT the .keras file, which
    carries optimizer state and metadata that would inflate the ratio)
  - INT8 TFLite size, and the compression ratio between them
  - macro-F1 of both, over the full 2947-window test set
  - delta-F1: how much accuracy quantisation actually costs
  - single-window inference latency, median and p95

Latency methodology:
  - batch size 1, because a wearable classifies one window at a time
  - 20 warmup runs discarded (first calls include allocation and cache
    warming, and would otherwise dominate the mean)
  - 200 timed runs, reporting MEDIAN not mean, since a background
    process spiking once would drag a mean but not a median
  - p95 reported too: for a real-time system the tail matters more
    than the average

Each model is converted in a subprocess so one failure cannot abort
the whole sweep.

Usage:
    python src/evaluate_tflite.py
"""

from __future__ import annotations

import glob
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

sys.path.insert(0, str(Path(__file__).resolve().parent))

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
TFLITE_DIR = REPO_ROOT / "models_tflite"

N_WARMUP = 20
N_TIMED = 200
N_CALIB = 300


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def driver():
    import pandas as pd

    paths = sorted(glob.glob(str(REPO_ROOT / "models" / "*.keras")))
    if not paths:
        print("No models found. Run src/train.py first.")
        return

    rows = []
    for path in paths:
        name = Path(path).stem
        print(f"  {name:<16} ...", end="", flush=True)
        r = subprocess.run([sys.executable, __file__, path],
                           capture_output=True, text=True)
        line = [l for l in r.stdout.splitlines() if l.startswith("ROW ")]
        if not line:
            print(f" FAILED (exit {r.returncode})")
            if r.stderr:
                print("      " + r.stderr.strip().splitlines()[-1][:120])
            continue
        vals = line[-1][len("ROW "):].split("|")
        rows.append({
            "model": vals[0],
            "seed": int(vals[1]),
            "fp32_kb": float(vals[2]),
            "int8_kb": float(vals[3]),
            "fp32_macro_f1": float(vals[4]),
            "int8_macro_f1": float(vals[5]),
            "fp32_ms_median": float(vals[6]),
            "fp32_ms_p95": float(vals[7]),
            "int8_ms_median": float(vals[8]),
            "int8_ms_p95": float(vals[9]),
        })
        print(f" int8 {vals[3]}KB  F1 {float(vals[5])*100:.2f}  "
              f"{vals[8]}ms")

    if not rows:
        print("Nothing converted.")
        return

    df = pd.DataFrame(rows)
    df["size_ratio"] = (df["fp32_kb"] / df["int8_kb"]).round(2)
    df["delta_f1"] = (df["int8_macro_f1"] - df["fp32_macro_f1"]).round(5)
    df["speedup"] = (df["fp32_ms_median"] / df["int8_ms_median"]).round(2)
    RESULTS_DIR.mkdir(exist_ok=True)
    df.to_csv(RESULTS_DIR / "edge_metrics_runs.csv", index=False)

    g = df.groupby("model", sort=False)
    summary = pd.DataFrame({
        "fp32_kb": g["fp32_kb"].mean().round(1),
        "int8_kb": g["int8_kb"].mean().round(1),
        "size_ratio": g["size_ratio"].mean().round(2),
        "fp32_f1_mean": g["fp32_macro_f1"].mean(),
        "int8_f1_mean": g["int8_macro_f1"].mean(),
        "int8_f1_std": g["int8_macro_f1"].std(ddof=1).fillna(0.0),
        "delta_f1_mean": g["delta_f1"].mean(),
        "fp32_ms": g["fp32_ms_median"].mean().round(3),
        "int8_ms": g["int8_ms_median"].mean().round(3),
        "int8_ms_p95": g["int8_ms_p95"].mean().round(3),
        "speedup": g["speedup"].mean().round(2),
    }).reset_index()
    summary.to_csv(RESULTS_DIR / "edge_metrics.csv", index=False)

    print()
    print("=" * 92)
    print("EDGE METRICS  (mean over seeds)")
    print("=" * 92)
    print(f"{'Model':<10}{'FP32 KB':>9}{'INT8 KB':>9}{'ratio':>7}"
          f"{'FP32 F1':>9}{'INT8 F1':>9}{'dF1':>8}"
          f"{'INT8 ms':>9}{'p95 ms':>8}{'speedup':>9}")
    print("-" * 92)
    for _, r in summary.iterrows():
        print(f"{r['model']:<10}{r['fp32_kb']:>9.1f}{r['int8_kb']:>9.1f}"
              f"{r['size_ratio']:>6.2f}x"
              f"{100*r['fp32_f1_mean']:>9.2f}{100*r['int8_f1_mean']:>9.2f}"
              f"{100*r['delta_f1_mean']:>+8.2f}"
              f"{r['int8_ms']:>9.3f}{r['int8_ms_p95']:>8.3f}"
              f"{r['speedup']:>8.2f}x")
    print("-" * 92)
    print()
    print(f"written: {RESULTS_DIR/'edge_metrics.csv'}")
    print(f"         {RESULTS_DIR/'edge_metrics_runs.csv'}")


# ---------------------------------------------------------------------------
# Worker helpers
# ---------------------------------------------------------------------------

def set_unroll(obj):
    """Set unroll=True on every LSTM inside a Keras config. Returns count."""
    n = 0
    if isinstance(obj, dict):
        if obj.get("class_name") == "LSTM":
            obj["config"]["unroll"] = True
            n += 1
        for v in obj.values():
            n += set_unroll(v)
    elif isinstance(obj, list):
        for v in obj:
            n += set_unroll(v)
    return n


def run_tflite_dataset(tfl_bytes, X, y):
    """Full test-set pass through a TFLite model. Returns macro-F1."""
    import numpy as np
    import tensorflow as tf
    from sklearn.metrics import f1_score

    interp = tf.lite.Interpreter(model_content=tfl_bytes)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]

    quantised = inp["dtype"] == np.int8
    if quantised:
        scale, zp = inp["quantization"]

    preds = np.empty(len(X), dtype=np.int64)
    for i in range(len(X)):
        x = X[i:i + 1]
        if quantised:
            x = np.clip(np.round(x / scale + zp), -128, 127).astype(np.int8)
        interp.set_tensor(inp["index"], x)
        interp.invoke()
        preds[i] = int(np.argmax(interp.get_tensor(out["index"])))

    return float(f1_score(y, preds, average="macro", zero_division=0))


def time_tflite(tfl_bytes, x_one):
    """Median and p95 single-window latency in milliseconds."""
    import numpy as np
    import tensorflow as tf

    interp = tf.lite.Interpreter(model_content=tfl_bytes)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]

    x = x_one
    if inp["dtype"] == np.int8:
        scale, zp = inp["quantization"]
        x = np.clip(np.round(x / scale + zp), -128, 127).astype(np.int8)

    for _ in range(N_WARMUP):
        interp.set_tensor(inp["index"], x)
        interp.invoke()

    times = np.empty(N_TIMED)
    for i in range(N_TIMED):
        interp.set_tensor(inp["index"], x)
        t0 = time.perf_counter()
        interp.invoke()
        times[i] = time.perf_counter() - t0
        _ = interp.get_tensor(out["index"])

    return float(np.median(times) * 1000), float(np.percentile(times, 95) * 1000)


# ---------------------------------------------------------------------------
# Worker: one model
# ---------------------------------------------------------------------------

def worker(path):
    import numpy as np
    import tensorflow as tf
    from data import load_har, subject_split

    stem = Path(path).stem                       # e.g. cnn_lstm_seed0
    arch, seed = stem.rsplit("_seed", 1)
    display = {"mlp": "MLP", "cnn": "CNN",
               "lstm": "LSTM", "cnn_lstm": "CNN-LSTM"}[arch]

    data = load_har()
    Xtr, ytr, Xva, yva, _, _ = subject_split(
        data["X_train"], data["y_train"], data["subj_train"],
        val_fraction=0.2, seed=42,
    )
    Xte, yte = data["X_test"], data["y_test"]

    # Calibration comes from TRAIN subjects only. Calibrating on
    # validation or test would tune the quantisation ranges on data the
    # model is being judged against - a subtle but real leak.
    rng = np.random.default_rng(0)
    idx = rng.choice(len(Xtr), size=N_CALIB, replace=False)
    X_calib = Xtr[idx].astype(np.float32)

    def rep_data():
        for i in range(len(X_calib)):
            yield [X_calib[i:i + 1]]

    model = tf.keras.models.load_model(path)

    # LSTMs must be unrolled to survive TFLite conversion. Verify the
    # rewrite is lossless rather than assuming it.
    cfg = model.get_config()
    if set_unroll(cfg) > 0:
        new = type(model).from_config(cfg)
        new(X_calib[:1])
        new.set_weights(model.get_weights())
        model = new

    x_in = tf.keras.Input(shape=(128, 9), batch_size=1, dtype="float32")
    fixed = tf.keras.Model(x_in, model(x_in))

    # ---- FP32 baseline ----
    conv = tf.lite.TFLiteConverter.from_keras_model(fixed)
    tfl_fp32 = conv.convert()

    # ---- INT8 ----
    conv = tf.lite.TFLiteConverter.from_keras_model(fixed)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = rep_data
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    tfl_int8 = conv.convert()

    TFLITE_DIR.mkdir(exist_ok=True)
    (TFLITE_DIR / f"{stem}_fp32.tflite").write_bytes(tfl_fp32)
    (TFLITE_DIR / f"{stem}_int8.tflite").write_bytes(tfl_int8)

    f1_fp32 = run_tflite_dataset(tfl_fp32, Xte, yte)
    f1_int8 = run_tflite_dataset(tfl_int8, Xte, yte)

    x_one = Xte[:1].astype(np.float32)
    ms_fp32, p95_fp32 = time_tflite(tfl_fp32, x_one)
    ms_int8, p95_int8 = time_tflite(tfl_int8, x_one)

    print(f"ROW {display}|{seed}|{len(tfl_fp32)/1024:.1f}|{len(tfl_int8)/1024:.1f}|"
          f"{f1_fp32:.6f}|{f1_int8:.6f}|"
          f"{ms_fp32:.3f}|{p95_fp32:.3f}|{ms_int8:.3f}|{p95_int8:.3f}",
          flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        worker(sys.argv[1])
    else:
        driver()