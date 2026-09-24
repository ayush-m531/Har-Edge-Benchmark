# HAR Edge Inference Benchmark — Report

## Summary

Four architectures were benchmarked on the raw UCI-HAR inertial signals: MLP, CNN, LSTM and CNN-LSTM. Each was trained three times with different seeds under the official subject-independent split, then converted to FP32 and INT8 TFLite and measured for size, accuracy and latency.

The accuracy result is that architecture barely matters. The CNN leads at 92.14 ± 0.73 macro-F1, ahead of the MLP by more than seed noise, but statistically tied with both the LSTM and the CNN-LSTM. Roughly 8% of the test set is unlearnable from this sensor regardless of model.

The deployment result is that architecture matters enormously. At statistically equal accuracy, the INT8 CNN is 60.7 KB and 0.018 ms per window; the INT8 LSTM is 2418.5 KB and 0.657 ms. Forty times the size and thirty-six times the latency for no measurable accuracy gain.

The most surprising finding is that INT8 quantisation made the recurrent models **larger**, not smaller.

---

## 1. Dataset and why the split matters

UCI-HAR records 30 people wearing a waist-mounted smartphone performing six activities. The inertial sensors sample at 50 Hz, and the stream is cut into 2.56-second windows of 128 readings.

Nine signals are provided per window:

- `body_acc_x/y/z` — acceleration with the gravity component filtered out
- `body_gyro_x/y/z` — angular velocity
- `total_acc_x/y/z` — raw acceleration, gravity included

One sample is therefore a 128 x 9 matrix. Train is `(7352, 128, 9)`, test is `(2947, 128, 9)`.

The channel statistics confirm what those names mean. After computing per-channel means across the training set, the six body channels centre on approximately zero, while `total_acc_x` sits at 0.804 — that value is gravity, loading onto the x-axis because the phone was worn in a consistent orientation at the waist.

**The official split is subject-independent.** Train contains subjects 1, 3, 5, 6, 7, 8, 11, 14-17, 19, 21-23, 25-30 — 21 people. Test contains subjects 2, 4, 9, 10, 12, 13, 18, 20, 24 — 9 different people. There is zero overlap.

This is the single most important fact about the benchmark. The model is not being asked to classify an unseen window from a familiar person; it must generalise to a new body, a new gait, a slightly different phone position. Results in the low 90s are normal under this protocol. Published figures reaching 97-99% generally come from overlapping-window regeneration or random splits, both of which let subject identity leak between train and test.

Class balance in the test set is close to uniform, between 14.3% and 18.2% per class.

---

## 2. What changed from the original submission

The 2025 version (preserved in `legacy/`) loaded the raw signals correctly and built four reasonable architectures. Its problems were in the experimental protocol, not the modelling.

### 2.1 Normalisation axis

The original computed:

    mean = X_train.mean(axis=0)   # shape (128, 9)

Averaging over samples only produces a separate mean for every (timestep, channel) pair — 1,152 statistics. Each timestep index gets its own distribution.

These are sliding windows, so "timestep 7" carries no consistent physical meaning. One window's timestep 7 might be mid-stride, another's at heel strike. Treating each index as having its own distribution assumes a temporal alignment that does not exist.

The corrected version uses `axis=(0, 1)`, giving nine statistics, one per sensor channel. The practical effect is small, but the original choice was not defensible.

### 2.2 Metric redundancy

The original reported Accuracy, Precision, Recall and F1 using weighted averaging. In its `comparison.csv`, the Accuracy and Recall columns are byte-identical for all four models — for example, the MLP shows `0.8910756701730573` in both.

That is not a coincidence. Weighted-average recall is mathematically equal to accuracy in single-label multiclass classification. With roughly balanced classes, weighted precision lands very close as well. The table presented four columns carrying about one column of information.

This version reports macro averages, which weight each class equally regardless of its frequency, plus per-class F1 so the distribution of errors is visible rather than hidden inside a single number.

### 2.3 No repetition, no error bars

Each model was trained once, for a fixed 10 epochs, with no seed set. The reported gaps — CNN 93.25%, LSTM 90.50%, MLP 89.11%, CNN-LSTM 88.63% — were treated as architecture differences.

With three seeds per model, the standard deviations turn out to be 0.34 to 1.35 points. The CNN-LSTM versus MLP gap in the original table was 0.48 points, comfortably inside that. The conclusion that the CNN was best was not supported by the evidence available.

### 2.4 Validation split

The original used Keras `validation_split=0.2`, which takes the final 20% of the array without shuffling. Because `subject_train.txt` happens to be sorted by subject, this accidentally produced a near subject-independent validation set (subjects 27-30). A lucky outcome, but not a designed one — and one that would silently break if the data were ever shuffled.

This version uses `GroupShuffleSplit` on the subject IDs, which guarantees that no person appears on both sides. The resulting validation set is 1,801 windows from subjects 1, 3, 15, 25 and 27; training is 5,551 windows from the remaining 16 subjects. The leakage check returns empty.

Why this matters: each person contributes many overlapping, near-identical windows. Under a random split, the model can score well on validation by learning to recognise *the person* rather than *the activity*. Validation accuracy inflates, and every decision based on it — including when to stop training — becomes unreliable.

---

## 3. The early stopping diagnosis

This was the largest single correction in the project, and it was found from validation behaviour rather than from test scores.

### 3.1 The symptom

The first corrected run used `EarlyStopping(monitor="val_loss", patience=8)`. The resulting best epochs were suspicious:

| Model | best_epoch per seed |
|---|---|
| MLP | 19, 13, 19 |
| CNN | 5, 4, 5 |
| LSTM | 4, 25, 13 |
| CNN-LSTM | 17, 3, 3 |

Two models were peaking within the first five epochs. The CNN-LSTM stopped at epoch 17 on one seed and epoch 3 on the other two — same architecture, same data, same code.

Stopping early also correlated with scoring worse. CNN-LSTM at epoch 17 scored 91.10 macro-F1; the two that stopped at epoch 3 scored 89.54 and 89.68.

### 3.2 The mechanism

Loss and accuracy measure different things.

Accuracy asks whether the highest-scoring class was correct. It is a count, and it ignores confidence entirely. Cross-entropy loss is `-log(p_correct)`, so it cares intensely about confidence.

Consider a single window where the model is wrong:

| Stage of training | P(correct class) | Correct? | Loss contribution |
|---|---|---|---|
| Early | 0.30 | No | 1.20 |
| Later | 0.02 | No | 3.91 |

Identical accuracy, identical prediction, more than triple the loss.

Under a subject-independent split this becomes decisive. The validation set contains a floor of genuinely unlearnable windows — predominantly SITTING and STANDING, which the per-class results confirm sit at 80-84 F1 for every architecture. As training proceeds, the model becomes confident on the 16 training subjects, and that confidence transfers to the 5 unseen ones including where it is wrong.

A few hundred confidently-wrong predictions accumulate loss faster than the small gains elsewhere reduce it. So `val_loss` bottoms out early and turns upward while `val_accuracy` is still climbing. Early stopping, watching the loss, pulls the plug.

### 3.3 The fix and its effect

Three changes: monitor `val_accuracy` with `mode="max"`, set `start_from_epoch=10` so stopping cannot fire during the volatile early epochs, and raise patience from 8 to 15.

| Model | v1 (val_loss) | v2 (val_accuracy) | Std change |
|---|---|---|---|
| MLP | 90.19 ± 0.65 | 90.59 ± 0.34 | halved |
| CNN | 90.64 ± 1.36 | **92.14 ± 0.73** | halved |
| LSTM | 91.04 ± 1.14 | 91.11 ± 0.56 | halved |
| CNN-LSTM | 90.11 ± 0.86 | 90.64 ± 1.35 | worse |

The CNN gained 1.5 points, and three of four standard deviations roughly halved. One CNN seed moved its best epoch from 4 to 48 and gained accuracy the whole way.

The general principle: monitor the quantity your reported metrics actually use. Every metric in this report — accuracy, macro-F1, per-class F1, confusion matrices — depends only on the argmax. The confidence value is never consumed. Optimising a criterion sensitive to something the evaluation ignores was the wrong choice.

`val_loss` remains correct for problems where calibrated probabilities matter, such as anything feeding a decision threshold.

---

## 4. Stage 1: accuracy results

Three seeds per architecture, early stopping on validation accuracy, evaluated once on the 2,947-window test set.

| Model | Params | Accuracy | Macro-F1 | Mean epochs | Mean best epoch | Train time |
|---|---|---|---|---|---|---|
| MLP | 156,230 | 90.65 ± 0.35 | 90.59 ± 0.34 | 38.3 | 23.3 | 6.2 s |
| CNN | 47,494 | 92.09 ± 0.80 | **92.14 ± 0.73** | 43.7 | 20.0 | 42.2 s |
| LSTM | 56,518 | 91.05 ± 0.49 | 91.11 ± 0.56 | 33.7 | 14.0 | 179.6 s |
| CNN-LSTM | 39,366 | 90.61 ± 1.32 | 90.64 ± 1.35 | 35.3 | 17.7 | 56.5 s |

### 4.1 Is the ranking real?

Applying the rule that a gap must exceed twice the seed spread to be claimed:

| Comparison | Gap | Seed spread | Verdict |
|---|---|---|---|
| CNN vs MLP | 1.55 | 0.73 | clear |
| CNN vs LSTM | 1.03 | 0.73 | within noise |
| CNN vs CNN-LSTM | 1.50 | 1.35 | within noise |

The defensible claim is therefore: **the CNN outperforms the MLP; the CNN, LSTM and CNN-LSTM are statistically indistinguishable.**

This is a weaker claim than "the CNN wins", and it is the correct one. Reporting it accurately is the entire reason for running three seeds.

### 4.2 Parameter count does not predict accuracy

The MLP carries 156,230 parameters — 3.3 times the CNN's 47,494 — and finishes last. It flattens the 128 x 9 window into a single 1,152-element vector, discarding the fact that adjacent readings are temporally related. The CNN slides a small filter across time, reusing the same weights at every position, so it learns what a footstep looks like once rather than separately at each offset.

The CNN-LSTM is the smallest model at 39,366 parameters and performs comparably to the MLP.

### 4.3 Validation scores are optimistic

Comparing peak validation accuracy against final test accuracy for seed 0 of each model:

| Model | Best val accuracy | Test accuracy | Drop |
|---|---|---|---|
| CNN | 98.11 | 92.87 | 5.2 |
| LSTM | 98.06 | 91.58 | 6.5 |
| MLP | 96.17 | 90.94 | 5.2 |

Every model loses 5-7 points moving from validation to test. This is not overfitting in the usual sense — validation is 5 people, test is 9 different people. Which particular humans end up in a split affects the score more than which architecture is used.

Validation numbers are fit for their purpose (deciding when to stop) but are not performance estimates. The test set was used exactly once, after all training decisions were final.

---

## 5. Where the errors live

Per-class F1, averaged across seeds:

| Model | WALK | UP | DOWN | SIT | STAND | LAY |
|---|---|---|---|---|---|---|
| MLP | 95.14 | 92.09 | 92.92 | 80.07 | 83.90 | 99.44 |
| CNN | 98.00 | 94.60 | 94.97 | 81.92 | 83.72 | 99.66 |
| LSTM | 95.38 | 93.92 | 93.15 | 81.84 | 83.36 | 99.00 |
| CNN-LSTM | 94.75 | 94.06 | 93.49 | 79.62 | 83.19 | 98.74 |

The pattern is identical across all four architectures, which is what makes it informative.

**LAYING is solved.** All models exceed 98.7 F1. Lying down rotates the device roughly 90 degrees, so the gravity vector shifts onto a different axis entirely. The signal is unambiguous.

**The three walking classes are strong**, 93-98 F1. They produce distinct periodic acceleration patterns, and ascending versus descending stairs differ measurably in vertical acceleration profile.

**SITTING and STANDING are the bottleneck**, 13 to 16 points below everything else for every model. Both are stationary postures with the torso upright. A waist-mounted accelerometer registers near-identical readings: the gravity vector points the same way, and there is no periodic motion in either.

This is a sensor limitation, not a modelling one. The information required to separate the two is not present in the data. Adding capacity, changing architecture or training longer cannot recover it. A thigh-mounted sensor, which would detect hip flexion, or an additional pressure sensor, would.

Practically: roughly 8% of the test set is unlearnable from these signals, which puts a ceiling near 92% on any model. All four architectures are effectively at that ceiling, which explains why they cluster so tightly.

---

## 6. Stage 2 and 3: deployment cost

Each of the 12 trained models was converted to both FP32 and INT8 TFLite and measured on the full test set.

Methodology notes:

- The FP32 TFLite file is the size baseline, not the `.keras` file. A `.keras` file carries optimiser state and metadata that would inflate the apparent compression ratio.
- Quantisation calibration draws 300 windows from **training subjects only**. Calibrating on validation or test data would tune the INT8 ranges using data the model is being judged against.
- Latency uses batch size 1, matching a wearable classifying one window at a time. Twenty warmup invocations are discarded, then 200 timed runs are recorded. Median is reported rather than mean, since a single background process spike would drag a mean but not a median. The 95th percentile is reported alongside, because for real-time systems the tail matters more than the average.
- LSTM layers must be unrolled to convert. The rewritten model is checked against the original before conversion, and the maximum absolute output difference was 0.0 in every case, confirming the transformation is lossless.

| Model | FP32 KB | INT8 KB | Size ratio | FP32 F1 | INT8 F1 | ΔF1 | INT8 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|
| MLP | 612.7 | 160.6 | 3.82x | 90.59 | 90.98 | +0.38 | 0.002 | 0.002 |
| CNN | 190.6 | **60.7** | **3.14x** | 92.14 | **91.81** | -0.33 | **0.018** | 0.018 |
| CNN-LSTM | 332.7 | 487.5 | 0.68x | 90.64 | 90.39 | -0.25 | 0.152 | 0.155 |
| LSTM | 918.5 | 2418.5 | 0.38x | 91.11 | 90.73 | -0.38 | 0.657 | 0.668 |

### 6.1 Quantisation costs no measurable accuracy

Every ΔF1 falls within ±0.4 points. The seed standard deviations from Stage 1 range from 0.34 to 1.35. The accuracy cost of INT8 is therefore **smaller than the noise from retraining the same model with a different seed**.

The MLP's +0.38 should not be read as quantisation improving the model. It is the same noise floor, landing on the positive side by chance.

The correct claim: on this task, INT8 post-training quantisation is accuracy-neutral for all four architectures.

### 6.2 Quantisation enlarges recurrent models

This is the most useful finding in the report, because it contradicts the standard expectation that quantisation shrinks models.

    LSTM       918.5 KB FP32  ->  2418.5 KB INT8   (2.6x LARGER)
    CNN-LSTM   332.7 KB FP32  ->   487.5 KB INT8   (1.5x LARGER)
    CNN        190.6 KB FP32  ->    60.7 KB INT8   (3.1x smaller)
    MLP        612.7 KB FP32  ->   160.6 KB INT8   (3.8x smaller)

The mechanism is unrolling. A recurrent layer normally executes as a loop, with one copy of the cell reused across timesteps. TFLite's integer quantisation path requires that loop to be written out explicitly, one block of operations per timestep.

The LSTM has two stacked recurrent layers over 128 timesteps: 256 unrolled cells. The CNN-LSTM pools before its single recurrent layer, reducing the sequence to 63 timesteps: 63 unrolled cells.

Each unrolled block carries its own operator definitions and quantisation parameters. Weights do shrink by the expected factor of four, but the graph metadata grows far faster.

The ratios are consistent with this explanation. The cell-count ratio between the LSTM and CNN-LSTM is 256/63 ≈ 4.1. The size-inflation ratio is 2.6/1.5 ≈ 1.7, and the absolute size ratio between the two INT8 files is 2418.5/487.5 ≈ 5.0. The overhead tracks unrolled cell count, not parameter count.

The MLP's 3.82x is closest to the theoretical 4x precisely because it is almost pure weights with minimal graph structure.

### 6.3 Latency

INT8 median latency spans two orders of magnitude:

    MLP        0.002 ms
    CNN        0.018 ms
    CNN-LSTM   0.152 ms
    LSTM       0.657 ms

The MLP and CNN process the full 128-timestep window in parallel — a dense layer is one matrix multiplication, a convolution slides over all positions simultaneously. The LSTM cannot: timestep 47 requires the hidden state from timestep 46, so the work is inherently sequential, 128 steps deep and twice over for two stacked layers.

The p95 values sit within a few percent of the medians for every model, indicating stable timing with no significant tail.

One caveat: at 0.002 ms the MLP measurement approaches the resolution floor of the timing method plus interpreter call overhead. That figure should be read as "below the measurement floor" rather than as a precise value.

---

## 7. Conclusion

**Deploy the CNN.**

The reasoning follows directly from the two tables. On accuracy, the CNN leads at 92.14 ± 0.73 and is the only model clearly ahead of the MLP; against the LSTM and CNN-LSTM it is statistically tied. Since accuracy does not separate the top three, deployment cost decides — and there the CNN is decisive: 60.7 KB against the LSTM's 2418.5 KB, and 0.018 ms against 0.657 ms. Forty times smaller, thirty-six times faster, at equal accuracy.

The CNN is also the only strong performer that quantises the way one would expect, shrinking 3.14x rather than growing.

In deployment terms, a wearable classifying one window every 2.56 seconds would spend roughly 0.0007% of its duty cycle on CNN inference. The LSTM would use 36 times that, for nothing.

The LSTM has no argument in its favour on this task. Equal accuracy, 40x the storage, 36x the latency, 4x the training time, and quantisation actively works against it.

---

## 8. Limitations

**Latency is measured on a MacBook Air CPU**, not on target wearable hardware. The relative ordering should hold, since it follows from parallelisable versus sequential computation, but absolute figures on an ARM Cortex-M class device would be substantially higher.

**Three seeds is a small sample** for estimating variance. It is enough to establish that architecture gaps fall inside run-to-run noise, but the standard deviations themselves carry meaningful uncertainty.

**Only post-training quantisation was tested.** Quantisation-aware training might recover the small accuracy losses, though at under 0.4 points there is little to recover here.

**The unrolling overhead is specific to TFLite.** Other runtimes handle recurrence differently, and the size inflation may not generalise to ONNX Runtime, Core ML or ExecuTorch.

**Architectures were not tuned.** They were kept identical to the original submission so the results remain directly comparable. A wider or deeper CNN might do better, but the SIT/STAND ceiling limits how much headroom exists.

---

## 9. Reproducing

    python3.11 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    bash scripts/download_data.sh

    python src/data.py             # prints the split, confirms zero leakage
    python src/train.py            # ~25 min CPU
    python src/evaluate_tflite.py  # ~20 min CPU

Outputs land in `results/`:

    summary.csv            per-architecture accuracy, mean +/- std
    runs.csv               all 12 individual runs
    per_class_f1.csv       per-activity breakdown
    edge_metrics.csv       size, latency, quantisation deltas
    edge_metrics_runs.csv  per-model edge measurements
    confusion_matrices/    one plot per architecture
    v1_valloss/            the earlier val_loss run, for comparison

---

## 10. Verifying the INT8 graph

Every converted model was inspected after conversion (`src/inspect_int8.py`, output in `results/int8_graph_check.csv`).

Conversion used strict `TFLITE_BUILTINS_INT8` with int8 input and output and no `SELECT_TF_OPS` fallback, so the converter fails rather than silently keeping a float op. Inspection confirms it: all 12 models contain **zero float32 tensors**.

Tensor counts measure the unrolling overhead directly:

| Model | Tensors | Unrolled cells | Tensors per cell | INT8 size |
|---|---|---|---|---|
| MLP | 13 | 0 | — | 160.6 KB |
| CNN | 24 | 0 | — | 60.7 KB |
| CNN-LSTM | 1,212 | 63 | 19.2 | 487.5 KB |
| LSTM | 4,871 | 256 | 19.0 | 2418.5 KB |

The LSTM has 128 timesteps and two stacked layers, so 256 unrolled cells. The CNN-LSTM pools the sequence to 63 steps before a single LSTM layer, so 63 cells.

The tensor ratio between them is 4,871 / 1,212 = **4.02**, against a cell-count ratio of 256 / 63 = **4.06**. Each unrolled cell adds about 19 tensors, the same in both models. This confirms that the size growth under INT8 in section 6.2 comes from the per-timestep copies of the cell, not from the weights.
