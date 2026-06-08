How to Run (Tail Prediction)

This guide focuses on running the Tail Prediction pipeline on Windows (PowerShell) and is usable on other OSes with equivalent shell commands.

1) Create environment and install dependencies

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2) Data layout

Place your data files under a folder (example: `data/`) with the following names:

```
data\train.txt
data\valid.txt
data\test.txt
```

Or set an alternate location via environment variable:

```powershell
$env:MUCOS_DATA_DIR = "path\to\data"
```

3) One-time full run (preprocess + train + test)

Set recommended runtime flags (these control preprocessing, cache behavior and context ordering), then run the single-command pipeline which executes preprocessing, training and testing.

```powershell
$env:MUCOS_FORCE_PREPROCESS = "1"
$env:MUCOS_CONTEXT_ORDER = "auto"
$env:MUCOS_TYPE_CONSTRAINT_ENABLED = "1"
$env:MUCOS_REQUIRE_TOKENIZED_CACHE = "1"
python Tail_Prediction\run_one_shot.py --data-dir data --processed-dir processed --output-dir outputs --epochs 30 --batch-size 8 --use-amp
```

Notes:
- If `MUCOS_REQUIRE_TOKENIZED_CACHE=1` the run will fail unless preprocessing writes tokenized caches to `processed/` (set `MUCOS_FORCE_PREPROCESS=1` to force preprocessing).
- `--use-amp` enables automatic mixed precision if a CUDA device is available.

4) Pause and resume

Pause safely by pressing `Ctrl+C` once; the pipeline will save a resume checkpoint in `outputs\checkpoints\latest_checkpoint.pth`.

To resume a run:

```powershell
$env:MUCOS_RESUME_TRAINING = "1"
python Tail_Prediction\run_one_shot.py --data-dir data --processed-dir processed --output-dir outputs
```

Start fresh (ignore checkpoints):

```powershell
$env:MUCOS_RESUME_TRAINING = "0"
python Tail_Prediction\run_one_shot.py --data-dir data --processed-dir processed --output-dir outputs
```

5) Useful flags (tail-focused)

- `MUCOS_CONTEXT_ORDER=auto`: reorder contexts to preserve relation tokens when truncation is likely.
- `MUCOS_TYPE_CONSTRAINT_ENABLED=1`: enable precomputed relation→tail candidate sets for faster constrained evaluation.
- `MUCOS_REQUIRE_TOKENIZED_CACHE=1`: require pre-built tokenized caches; fail fast if missing.
- `MUCOS_SAVE_TOKENIZED_CACHE=1`: save tokenized caches during preprocessing (on by default).
- `--use-amp` / `--no-use-amp`: enable/disable AMP.
- `MUCOS_PAD_TO_MULTIPLE_OF=8`: pad to multiples for faster GPU kernels.
- `MUCOS_CHECKPOINT_EVERY_STEPS=100`: checkpoint frequency (optimizer steps).

6) Outputs

- Processed data: `processed/`
- Checkpoints: `outputs\checkpoints/`
- Best model: `outputs\best_model/`
- Metrics: `outputs\test_metrics.json`, `outputs\speed_metrics.json`

7) Quick local test (small dataset)

For a rapid smoke test using a tiny dataset, use:

```powershell
python Tail_Prediction\run_one_shot.py --data-dir data_small --processed-dir processed_small --output-dir outputs_small --epochs 2 --batch-size 4
```

This run helps verify end-to-end preprocessing, caching, training and evaluation without full-scale resources.

