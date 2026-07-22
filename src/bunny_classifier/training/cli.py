"""`bunny-train` CLI: run a tracked training job.

Flags default to None so that an unset flag falls through to the YAML config
(if any) and then to the TrainConfig defaults — see TrainConfig.replace().
"""

from __future__ import annotations

import argparse
from pathlib import Path

from bunny_classifier.models.backbone import SUPPORTED_BACKBONES
from bunny_classifier.training.config import TrainConfig


def build_config(args: argparse.Namespace) -> TrainConfig:
    base = TrainConfig.from_yaml(args.config) if args.config else TrainConfig()
    return base.replace(
        backbone=args.backbone,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        num_workers=args.num_workers,
        device=args.device,
        experiment=args.experiment,
        run_name=args.run_name,
        class_weighting=False if args.no_class_weighting else None,
        eval_test=True if args.eval_test else None,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bunny-train", description="Train a bunny classifier.")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--manifest", type=Path, default=None, help="default: <repo-root>/data/manifest.csv"
    )
    parser.add_argument("--config", type=Path, default=None, help="YAML config file")
    parser.add_argument("--backbone", choices=SUPPORTED_BACKBONES, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None)
    parser.add_argument("--experiment", default=None, help="MLflow experiment name")
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--no-class-weighting", action="store_true", help="disable inverse-frequency loss weights"
    )
    parser.add_argument(
        "--eval-test",
        action="store_true",
        help="also score the held-out test split — only for a model you intend to ship",
    )
    args = parser.parse_args(argv)

    from bunny_classifier.training.train import track_uri_hint, train  # heavy: after arg parsing

    config = build_config(args)
    repo_root: Path = args.repo_root.resolve()
    summary = train(config, repo_root, args.manifest)

    print(f"\nrun {summary.run_id}")
    print(f"best epoch: {summary.best_epoch}")
    print(
        f"val macro-F1: {summary.val_macro_f1:.4f} "
        f"(majority-class baseline {summary.majority_baseline_macro_f1:.4f}), "
        f"val accuracy: {summary.val_accuracy:.4f}"
    )
    if summary.test_macro_f1 is not None:
        print(f"test macro-F1: {summary.test_macro_f1:.4f}")
    print(f"\n{track_uri_hint(repo_root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
