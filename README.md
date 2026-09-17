# HAR Edge Inference Benchmark

Which neural architecture should run human activity recognition on a wearable device?

Accuracy alone does not answer that. This repository benchmarks four architectures on the raw UCI-HAR inertial signals across three seeds each, then measures what each one actually costs to deploy after INT8 quantisation.

## Headline result

| Model | Macro-F1 | INT8 size | Latency / window | INT8 vs FP32 size |
|---|---|---|---|---|
| **CNN** | **92.14 ± 0.73** | **60.7 KB** | **0.018 ms** | 3.14x smaller |
| LSTM | 91.11 ± 0.56 | 2418.5 KB | 0.657 ms | 2.6x **larger** |
| CNN-LSTM | 90.64 ± 1.35 | 487.5 KB | 0.152 ms | 1.5x **larger** |
| MLP | 90.59 ± 0.34 | 160.6 KB | 0.002 ms | 3.82x smaller |

Three findings:

**1. Accuracy barely separates the architectures.** The CNN beats the MLP by more than seed noise, but ties the LSTM and CNN-LSTM. Single-run comparisons on this dataset are not meaningful — the spread from simply retraining with a different seed is as large as the gap between architectures.

**2. INT8 quantisation is effectively free.** Every architecture loses under 0.4 macro-F1 points, which is smaller than the seed-to-seed variation. The MLP even gains 0.38, which is the same noise floor rather than quantisation improving anything.

**3. Quantisation makes recurrent models bigger, not smaller.** LSTMs must be unrolled to convert to TFLite, writing the recurrent cell out once per timestep. The resulting graph overhead outweighs the 4x weight saving: the LSTM grows from 918 KB to 2419 KB. The CNN, with no recurrence to unroll, shrinks as expected.

**Conclusion:** the CNN is the deployment choice. Statistically equal accuracy to the LSTM, 40x smaller on device, 36x faster per window.

## Setup

    python3.11 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    bash scripts/download_data.sh

## Running

    python src/data.py             # verify the split, build the cache
    python src/train.py            # 4 architectures x 3 seeds, ~25 min CPU
    python src/evaluate_tflite.py  # FP32/INT8 size, accuracy, latency, ~20 min

`python src/train.py --quick` runs a two-minute smoke test of the whole pipeline.

## Experimental design

**Subject-independent evaluation.** The official UCI-HAR split puts 21 people in train and 9 different people in test, with no overlap. The model must generalise to a new body, not to a new window from a familiar one. This is why results here sit near 92% while some published figures reach 99% — those typically use overlapping-window generation or random splits that leak subject identity.

**Subject-wise validation.** Validation holds out 5 whole people using `GroupShuffleSplit`. A random split would let the model recognise the person rather than the activity, since each person contributes many near-identical overlapping windows.

**Early stopping on accuracy, not loss.** Validation loss rises here while validation accuracy is still improving. See `report.md` for the full diagnosis. The earlier `val_loss` run is kept in `results/v1_valloss/` for comparison.

**Three seeds, reported as mean ± std.** Differences smaller than twice the seed spread are not claimed as real.

**Honest quantisation baseline.** INT8 TFLite is compared against FP32 TFLite, not against the `.keras` file, which carries optimiser state and would inflate the ratio. Calibration samples training subjects only, so validation and test data never influence the quantisation ranges.

## Where the errors are

Per-class F1 is nearly identical across all four architectures:

| Model | WALK | UP | DOWN | SIT | STAND | LAY |
|---|---|---|---|---|---|---|
| MLP | 95.14 | 92.09 | 92.92 | 80.07 | 83.90 | 99.44 |
| CNN | 98.00 | 94.60 | 94.97 | 81.92 | 83.72 | 99.66 |
| LSTM | 95.38 | 93.92 | 93.15 | 81.84 | 83.36 | 99.00 |
| CNN-LSTM | 94.75 | 94.06 | 93.49 | 79.62 | 83.19 | 98.74 |

SITTING and STANDING sit 13-16 points below everything else for every model. A waist-mounted accelerometer sees a stationary torso in both cases, so the signal does not contain the information needed to separate them. No architecture change closes this gap; a different sensor placement would.

## Layout

    src/data.py              loading, normalisation, subject-wise splitting
    src/train.py             multi-seed benchmark
    src/evaluate_tflite.py   FP32/INT8 conversion, accuracy, size, latency
    src/check_int8.py        quick conversion sanity check
    results/                 metrics, confusion matrices, run logs
    legacy/                  original 2025 submission, kept for comparison
    report.md                full analysis and methodology notes

## Relationship to the original submission

`legacy/` holds the 2025 version, which reported CNN 93.25%, LSTM 90.50%, MLP 89.11% and CNN-LSTM 88.63% from one run each, and concluded the CNN was clearly best.

Re-running with three seeds and a proper validation protocol shows those gaps sit inside run-to-run variance. The original conclusion did not survive.
