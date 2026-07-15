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
