# How to Run

This guide covers the reproducible tail-prediction pipeline.

## 1) Create and activate a virtual environment (optional)

Windows PowerShell:

```bash
python -m venv venv
venv\Scripts\activate
```

## 2) Install dependencies

```bash
pip install -r requirements.txt.txt
```

## 3) Prepare data

Place your files in a data directory with this structure:

```bash
data/
  train.txt
  valid.txt
  test.txt
```

Each line must be a tab-separated triple:

```bash
head_entity\trelation\ttail_entity
```

## 4) (Optional) Set paths

If your data is not in data/, set the environment variable:

```bash
set MUCOS_DATA_DIR=path\to\data
```

You can also override output locations:

```bash
set MUCOS_PROCESSED_DIR=path\to\processed
set MUCOS_OUTPUT_DIR=path\to\outputs
```

## 5) Run the reproducible tail-prediction pipeline

```bash
python Tail_Prediction/main_tail.py
```

For a single command with explicit overrides, use the wrapper:

```bash
python Tail_Prediction/run_one_shot.py --data-dir data --processed-dir processed --output-dir outputs --epochs 3
```

## 6) Stop and resume when you want

You can stop training safely by letting the current run finish its step, or by setting a time budget so it pauses automatically and writes `outputs/checkpoints/latest_checkpoint.pth`.

To pause after about an hour:

```bash
set MUCOS_MAX_TRAIN_SECONDS=3600
python Tail_Prediction/run_one_shot.py --data-dir data --processed-dir processed --output-dir outputs
```

To resume later, run the same command again. If `MUCOS_RESUME_TRAINING` is left at its default value of `1`, the trainer will reload the latest checkpoint and continue from there.

If you want to force a fresh run, set:

```bash
set MUCOS_RESUME_TRAINING=0
```

Notes:
- If you have a CUDA-capable GPU and want faster training, ensure a compatible `torch` build is installed (see `requirements.txt.txt`).
- You can override data and output locations with environment variables as described above.

## 6) Outputs

After running, you should see:

- processed/ with precomputed contexts and vocabularies
- outputs/ with best_model, checkpoints, metrics, and speed stats
