# How to Run (Tail Prediction)

This guide is Windows-friendly and works on other OSes with equivalent shell syntax.

## 1) Environment setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 2) Data layout

Place your data in:

```
data/
  train.txt
  valid.txt
  test.txt
```

Or set an alternate location:

```powershell
$env:MUCOS_DATA_DIR = "path\to\data"
```

## 3) One-time full run (preprocess + train + test)

```powershell
$env:MUCOS_FORCE_PREPROCESS = "1"
$env:MUCOS_CONTEXT_ORDER = "auto"
$env:MUCOS_TYPE_CONSTRAINT_ENABLED = "1"
$env:MUCOS_REQUIRE_TOKENIZED_CACHE = "1"
python Tail_Prediction\run_one_shot.py --data-dir data --processed-dir processed --output-dir outputs --epochs 30 --batch-size 8 --use-amp
```

## 4) Pause and resume

### Pause (safe)
- Press `Ctrl+C` once. A resume checkpoint is saved to `outputs\checkpoints\latest_checkpoint.pth`.

### Resume
```powershell
$env:MUCOS_RESUME_TRAINING = "1"
python Tail_Prediction\run_one_shot.py --data-dir data --processed-dir processed --output-dir outputs
```

### Start fresh (ignore checkpoints)
```powershell
$env:MUCOS_RESUME_TRAINING = "0"
python Tail_Prediction\run_one_shot.py --data-dir data --processed-dir processed --output-dir outputs
```

## 5) Useful flags and what they do

### Core speed/accuracy flags
- `MUCOS_CONTEXT_ORDER=auto`  : reorders contexts when truncation is likely (accuracy-friendly).
- `MUCOS_TYPE_CONSTRAINT_ENABLED=1` : relation-based candidate sets for faster evaluation.
- `MUCOS_REQUIRE_TOKENIZED_CACHE=1` : fail fast if tokenized cache is missing (prevents slow fallback).
- `MUCOS_SAVE_TOKENIZED_CACHE=1` : save tokenized caches (on by default).

### Runtime controls
- `--use-amp` / `--no-use-amp` : enable/disable AMP on CUDA.
- `MUCOS_PAD_TO_MULTIPLE_OF=8` : padding alignment for faster GPU kernels.
- `MUCOS_CHECKPOINT_EVERY_STEPS=100` : checkpoint frequency in optimizer steps.

### Data and output overrides
- `--data-dir`, `--processed-dir`, `--output-dir` : command-line overrides.
- `MUCOS_DATA_DIR`, `MUCOS_PROCESSED_DIR`, `MUCOS_OUTPUT_DIR` : environment overrides.
- `MUCOS_FORCE_PREPROCESS=1` : rebuild processed data even if present.

## 6) Where outputs go

- Processed data: `processed/`
- Checkpoints: `outputs/checkpoints/`
- Best model: `outputs/best_model/`
- Metrics: `outputs/test_metrics.json`, `outputs/speed_metrics.json`
