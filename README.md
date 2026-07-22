# Bunny Classifier

Image classifier for bunny poses/activities (7 classes), built end-to-end: PyTorch fine-tuning,
MLflow tracking, ONNX export, FastAPI inference API with Prometheus/Grafana monitoring,
deployed via Docker on GCP.

> Work in progress — full README lands with the portfolio-polish phase.

## Development

```bash
uv sync                 # base + dev tooling
uv sync --extra train   # + PyTorch/MLflow/TensorBoard (training)
uv sync --extra serve   # + FastAPI/onnxruntime (serving)

uv run pytest           # tests
uv run ruff check .     # lint
uv run mypy             # type check
uv run pre-commit install  # git hooks (once)
```

## Data

Images live in `data/bunnies_batch_<YYMMDD>/` (gitignored) named `<label><number>.png`.
The committed `data/manifest.csv` maps every image to a train/val/test split — append-only,
so the test set stays stable as new batches arrive.

```bash
uv run bunny-data import-batch data/bunnies_batch_<YYMMDD>  # label subfolders -> <label>NN.png
uv run bunny-data build-manifest  # scan, dedupe, assign splits; re-run is a no-op
uv run bunny-data eda             # class/size distributions + sample grids -> reports/eda/
```

Cross-label near-duplicates are never auto-resolved; they land in
`reports/near_dup_review.csv` (+ contact sheet PNG) for human review.

## Training

```bash
uv sync --extra train
uv run bunny-train                       # ResNet18, frozen backbone, weighted loss
uv run bunny-train --config configs/baseline.yaml

mlflow ui --backend-store-uri sqlite:///mlflow.db   # runs, params, artifacts
tensorboard --logdir runs                          # curves
```

Headline metric is macro-F1 on the validation split, logged alongside the
majority-class baseline. The test split is only scored with an explicit
`--eval-test`.

## Docs

Detailed documentation lives in [`docs/`](docs/README.md) —
[the data pipeline](docs/data.md) for manifest, deduplication, and split policy;
[training](docs/training.md) for transfer learning, metrics, and tracking.
