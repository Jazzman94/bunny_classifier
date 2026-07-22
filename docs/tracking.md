# Experiment Tracking — MLflow & TensorBoard

A practical guide to the two tracking tools: what each concept means, exactly
where the wiring lives in this repo, and how to add something new without
breaking a run.

For *why* metrics are defined the way they are, see
[training.md](training.md). This document is about mechanics.

## The two tools, and why both

| | MLflow | TensorBoard |
|---|---|---|
| Unit of interest | one **run** (a whole experiment) | one **series over time** (steps within a run) |
| Answers | *Which configuration won?* | *What happened inside this run?* |
| Stores | params, metrics, artifacts, tags | scalars, images, histograms, graphs |
| Backend here | SQLite `mlflow.db` + `mlartifacts/` | event files in `runs/<run-name>/` |
| Comparison | table + parallel-coordinates plot across runs | overlaid curves |

Both are written from the same loop, so every run is recorded twice on purpose.
The redundancy is cheap and each tool is bad at the other's job: MLflow's metric
plots are clumsy, TensorBoard has no notion of "the config that produced this".

## MLflow

### Concepts

| Concept | Meaning | Mutable? |
|---|---|---|
| **Tracking URI** | where run metadata is stored — here `sqlite:///mlflow.db` | — |
| **Experiment** | a named group of runs (`bunny-classifier`) | — |
| **Run** | one execution: params + metrics + artifacts + tags | — |
| **Param** | an input you chose (learning rate, backbone). Always a string. | **No — write once** |
| **Metric** | a measured number; may carry a `step` to form a time series | Yes, append |
| **Artifact** | any file (image, CSV, checkpoint, JSON) | append |
| **Tag** | free-form key/value metadata (git SHA, note) | Yes |

The param/metric distinction matters: **a param is what you set, a metric is
what you got.** MLflow enforces it — params are immutable within a run.

### Where it lives in the code

All MLflow wiring is in `src/bunny_classifier/training/train.py`:

| What | Function / location |
|---|---|
| Resolve the backend | `tracking_uri()` — env `MLFLOW_TRACKING_URI` wins, else `sqlite:///<repo>/mlflow.db` |
| Create/find the experiment | `_experiment_id()` — pins `artifact_location` to `<repo>/mlartifacts` |
| Open the run | `with mlflow.start_run(experiment_id=..., run_name=...) as run:` inside `train()` |
| Log the config | `mlflow.log_params(config.as_params())` |
| Log run facts | second `log_params({...})` — resolved device, image counts, parameter counts |
| Log per-epoch metrics | `mlflow.log_metrics({...}, step=epoch)` in the epoch loop |
| Log final metrics | `mlflow.log_metrics({...})` after the loop |
| Log artifacts | `mlflow.log_artifacts(str(out_dir))` — uploads a whole temp directory at once |

`_experiment_id()` exists instead of the simpler `mlflow.set_experiment()`
because that helper resolves the artifact location relative to the current
working directory, which scatters artifacts when a run is launched from
elsewhere. Pinning it at experiment-creation time keeps everything under the
repo.

### Recipe: add a new parameter

Parameters describe the run's inputs, so the right place is almost always
`TrainConfig` — then it is logged, saved to `config.json`, and settable from
YAML and the CLI in one go.

1. Add the field to `TrainConfig` (`training/config.py`) with a default:

   ```python
   label_smoothing: float = 0.0
   ```

2. Validate it in `__post_init__` if a bad value would be silently wrong:

   ```python
   if not 0.0 <= self.label_smoothing < 1.0:
       raise ValueError("label_smoothing must be in [0, 1)")
   ```

3. Add a CLI flag in `training/cli.py` — **`default=None`**, and pass it through
   in `build_config()`:

   ```python
   parser.add_argument("--label-smoothing", type=float, default=None)
   # ...
   return base.replace(..., label_smoothing=args.label_smoothing)
   ```

4. Use it in `train()`:

   ```python
   criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=config.label_smoothing)
   ```

Nothing else is needed: `as_params()` serializes every dataclass field, so the
new knob appears in MLflow, in `config.json`, and in the YAML schema
automatically.

For facts that are *not* config — something resolved at runtime, like the git
SHA or the actual device — add them to the second `log_params()` call instead.

> **Pitfall — params are write-once.** Logging the same key twice with different
> values raises `MlflowException: Changing param values is not allowed`. This
> repo hit exactly that: `TrainConfig.device` is `"auto"`, and the resolved value
> is `"cuda"`, so the runtime fact is logged under the distinct key
> `resolved_device`. When adding a runtime param, make sure its key does not
> collide with a `TrainConfig` field.

### Recipe: add a new metric

```python
mlflow.log_metric("val_top2_accuracy", value, step=epoch)   # a series
mlflow.log_metric("best_val_macro_f1", value)               # a single summary number
```

Rules that keep the UI usable:

- **Include `step=`** for anything measured per epoch; without it the run shows
  one point and you lose the curve.
- **Omit `step=`** for end-of-run summaries (`best_*`, `test_*`), which is how
  the sortable column in the run table behaves sensibly.
- **Keep names stable** across runs — a renamed metric cannot be compared with
  history, and the comparison view keys on the name.
- Values must be numeric. Log a `bool` as `int`, a category as a **tag**.

Prefer computing the number in `training/metrics.py` (pure Python, unit-tested
in CI) and only logging it here. Metric bugs are hard to notice and easy to
believe.

### Recipe: add a new artifact

Artifacts are written into the run's temp directory in `_write_artifacts()` and
uploaded in one `mlflow.log_artifacts()` call at the end:

```python
def _write_artifacts(out_dir: Path, ...) -> None:
    ...
    render_my_new_plot(data, out_dir / f"{prefix}_my_plot.png")
```

Follow the existing conventions: render with PIL (`training/artifacts.py`), keep
the rendering function torch-free so it is testable in CI, and prefix
val/test-specific files with `prefix` so both splits can coexist.

Writing to a `TemporaryDirectory` and uploading once is deliberate — it keeps
the run's artifact set atomic and leaves no debris in the working tree if the
run crashes.

### Reading the results

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db     # http://localhost:5000
```

- The **run table** lists params and metrics as sortable columns; the column
  picker controls which are shown.
- Tick two or more runs → **Compare** for a side-by-side table and a
  parallel-coordinates plot (each param becomes an axis, lines are runs) —
  the fastest way to see which knob actually moved the metric.
- A run's own page holds its **Artifacts** tab: confusion matrix, prediction
  grid, `config.json`, `model.pt`.
- The search box takes filter expressions:

  ```
  params.backbone = 'resnet18' and metrics.best_val_macro_f1 > 0.75
  ```

> **Pitfall — a metric filter uses the LAST logged value, not the best one.**
> `metrics.val_macro_f1` is a per-epoch series, so filtering on it compares
> against whatever the final epoch happened to score. Measured on this repo's
> two runs:
>
> | filter | matches |
> |---|---|
> | `metrics.val_macro_f1 > 0.75` | **0 runs** |
> | `metrics.best_val_macro_f1 > 0.75` | 2 runs |
>
> Both runs peaked above 0.75 (0.7547 and 0.7755) but ended below it (0.7132 and
> 0.7372) — validation metrics wobble from epoch to epoch, and the last epoch is
> not the best one. This is exactly why the loop logs step-less summary metrics
> (`best_val_macro_f1`, `best_epoch`, `best_val_loss`) in addition to the series:
> **filter and sort on the `best_*` metrics, plot the series.**

### Programmatic access

Useful when a sweep produces more runs than the UI is pleasant for:

```python
import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")
df = mlflow.search_runs(
    experiment_names=["bunny-classifier"],
    order_by=["metrics.best_val_macro_f1 DESC"],
)
print(df[["run_id", "params.backbone", "params.epochs", "metrics.best_val_macro_f1"]])
```

This becomes the basis of the Phase 3 sweep comparison and of choosing which run
to promote in the registry.

## TensorBoard

### Concepts

TensorBoard reads append-only **event files**. One `SummaryWriter` writes one
directory; `--logdir` points at the *parent*, and each subdirectory becomes a
separately toggleable series in the UI. That is why runs are written to
`runs/<run-name>/` — the run name is what you will see in the legend, so
`--run-name` is worth setting on every experiment.

### Where it lives in the code

```python
writer = SummaryWriter(log_dir=str(repo_root / "runs" / run_name))   # train()
...
writer.add_scalars("loss", {"train": train_loss, "val": val_result.loss}, epoch)
writer.add_scalar("val/macro_f1", val_f1, epoch)
writer.add_scalar("val/accuracy", val_accuracy, epoch)
...
writer.close()
```

Two things to notice:

- **`add_scalars` (plural)** puts several series on *one* chart — used for
  train-vs-val loss, because their divergence is the whole point.
- **The `/` in a tag** creates a group: `val/macro_f1` and `val/accuracy` are
  filed together under a `val` heading. Use it to keep the sidebar navigable.

`writer.close()` flushes the buffer. Without it the last epochs can be missing
from the UI if the process exits promptly.

### Recipe: add a scalar

```python
writer.add_scalar("train/learning_rate", scheduler.get_last_lr()[0], epoch)
```

The third argument is `global_step` — the x-axis. Use the epoch number
consistently; mixing epochs and batch indices in one run produces a chart nobody
can read.

### Recipe: add an image

Useful in Phase 3 to watch the confusion matrix evolve rather than only seeing
its final state:

```python
import numpy as np
from PIL import Image

array = np.array(Image.open(path))              # HWC, uint8
writer.add_image("val/confusion_matrix", array, epoch, dataformats="HWC")
```

`add_image` defaults to CHW tensors; passing `dataformats="HWC"` is what lets a
PIL/numpy image go straight in.

### Other writers worth knowing

| Call | Shows |
|---|---|
| `add_histogram("head/weight", param, epoch)` | distribution of a tensor over time — spot dead or exploding weights |
| `add_graph(model, example_input)` | the network structure |
| `add_hparams({...}, {...})` | a hyperparameter table; largely redundant here since MLflow owns params |
| `add_pr_curve(...)` | precision-recall curve per class |

### Reading the results

```bash
tensorboard --logdir runs                             # http://localhost:6006
```

- Each `runs/<name>/` is a toggleable series — untick to declutter.
- **Smoothing** (top left) is a display filter only; with 15 noisy points turn it
  down or you will read the smoother, not the data.
- Charts are live: refresh picks up an in-progress run.
- Delete a directory under `runs/` to drop a series from the UI. Note this does
  **not** delete the MLflow run — the two stores are independent.

## Operational notes

**Run them from separate terminals.** Both are ordinary web servers reading files
from disk. Start them before, during or after training; they need no restart
between runs and pick up new data on refresh. `--port N` if a default port is
taken.

**Nothing is committed.** `mlflow.db`, `mlartifacts/` and `runs/` are all
gitignored, so run history is machine-local and a fresh clone starts empty. What
*is* portable is the config: a run is reproducible from `config.json` + the
manifest + the seed.

**Point at a different store** with `MLFLOW_TRACKING_URI` — e.g. a shared MLflow
server. The code reads it in `tracking_uri()` and everything else follows.

**Keeping the two stores aligned.** MLflow and TensorBoard are written
independently; if a metric is added to only one, the two disagree and it is
never obvious which is stale. When adding a per-epoch metric, add it to both in
the same edit.

**Housekeeping.** To start clean:

```bash
rm -rf mlflow.db mlartifacts runs
```

Runs can also be deleted from the MLflow UI, but that soft-deletes the metadata
and leaves artifact files behind; for a local store the `rm` above is simpler
and honest.
