import glob, os, sys, subprocess
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # hide most TF info logs


# ---------------------------------------------------------------------------
# Driver: converts each model in a separate process, so a crash in one
# model does not stop the others
# ---------------------------------------------------------------------------
def driver():
    for path in sorted(glob.glob("models/*.keras")):
        name = os.path.basename(path).replace(".keras", "")
        r = subprocess.run([sys.executable, __file__, path],
                           capture_output=True, text=True)
        results = [l for l in r.stdout.splitlines() if l.startswith("RESULT ")]
        if results:
            print(results[-1][len("RESULT "):], flush=True)
        else:
            print(f"{name:16s} CRASHED (exit code {r.returncode})", flush=True)


# ---------------------------------------------------------------------------
# Worker: converts ONE model
# ---------------------------------------------------------------------------
def set_unroll(obj):
    """Set unroll=True on every LSTM layer inside a Keras config. Returns count."""
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


def worker(path):
    import numpy as np
    import tensorflow as tf
    from data import load_har, subject_split

    name = os.path.basename(path).replace(".keras", "")

    # Same loading + split as train.py (normalisation already done inside load_har)
    data = load_har()
    Xtr, ytr, Xva, yva, s_tr, s_va = subject_split(
        data["X_train"], data["y_train"], data["subj_train"],
        val_fraction=0.2, seed=42,
    )

    # Calibration: 300 RANDOM windows from train subjects only
    rng = np.random.default_rng(0)
    idx = rng.choice(len(Xtr), size=300, replace=False)
    X_calib = Xtr[idx].astype(np.float32)

    def rep_data():
        for i in range(len(X_calib)):
            yield [X_calib[i:i+1]]      # shape (1, 128, 9)

    model = tf.keras.models.load_model(path)

    # Rebuild with unrolled LSTMs (same weights), only if the model has LSTMs
    cfg = model.get_config()
    n_lstm = set_unroll(cfg)
    note = ""
    if n_lstm > 0:
        new = type(model).from_config(cfg)
        new(X_calib[:1])                          # build the layers
        new.set_weights(model.get_weights())      # copy trained weights
        diff = float(np.max(np.abs(
            model.predict(X_calib[:5], verbose=0) - new.predict(X_calib[:5], verbose=0)
        )))
        note = f"[unrolled {n_lstm} LSTM, diff {diff:.1e}]"
        model = new

    # Wrap in a Keras model with FIXED batch size 1
    x_in = tf.keras.Input(shape=(128, 9), batch_size=1, dtype="float32")
    fixed = tf.keras.Model(x_in, model(x_in))

    conv = tf.lite.TFLiteConverter.from_keras_model(fixed)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = rep_data
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]  # strict mode
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8

    try:
        tfl = conv.convert()
    except Exception as e:
        print(f"RESULT {name:16s} FAILED -> {str(e).splitlines()[0][:100]} {note}", flush=True)
        return

    os.makedirs("models_tflite", exist_ok=True)
    with open(f"models_tflite/{name}_int8.tflite", "wb") as f:
        f.write(tfl)

    interp = tf.lite.Interpreter(model_content=tfl)
    interp.allocate_tensors()
    n_float = sum(t["dtype"] == np.float32 for t in interp.get_tensor_details())

    # Smoke test: run one real window through the int8 model
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    scale, zp = inp["quantization"]
    x_q = np.clip(np.round(X_calib[:1] / scale + zp), -128, 127).astype(np.int8)
    try:
        interp.set_tensor(inp["index"], x_q)
        interp.invoke()
        pred = int(np.argmax(interp.get_tensor(out["index"])))
        status = f"runs OK (pred {pred})"
    except Exception as e:
        status = f"INVOKE FAILED: {str(e)[:50]}"

    print(f"RESULT {name:16s} OK  {len(tfl)/1024:6.1f} KB  float left: {n_float}  "
          f"{status} {note}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        worker(sys.argv[1])
    else:
        driver()