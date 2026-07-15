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
uv run bunny-data build-manifest  # scan, dedupe, assign splits; re-run is a no-op
uv run bunny-data eda             # class/size distributions + sample grids -> reports/eda/
```

Cross-label near-duplicates are never auto-resolved; they land in
`reports/near_dup_review.csv` (+ contact sheet PNG) for human review.

## Docs

Detailed documentation lives in [`docs/`](docs/README.md) — see
[the data pipeline](docs/data.md) for manifest, deduplication, and split policy.
