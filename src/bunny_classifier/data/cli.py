"""`bunny-data` CLI: build the dataset manifest and generate EDA artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bunny_classifier.data.eda import run_eda
from bunny_classifier.data.manifest import build_manifest
from bunny_classifier.data.review import write_review_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bunny-data")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build-manifest", help="scan data/, dedupe, assign splits (append-only)")
    build.add_argument("--data-dir", type=Path, default=None, help="default: <repo-root>/data")
    build.add_argument(
        "--manifest", type=Path, default=None, help="default: <data-dir>/manifest.csv"
    )
    build.add_argument(
        "--reports-dir", type=Path, default=None, help="default: <repo-root>/reports"
    )
    build.add_argument("--seed", type=int, default=42)

    eda = sub.add_parser("eda", help="write class/size distributions and sample grids")
    eda.add_argument("--manifest", type=Path, default=None)
    eda.add_argument("--out-dir", type=Path, default=None, help="default: <repo-root>/reports/eda")

    args = parser.parse_args(argv)
    repo_root: Path = args.repo_root.resolve()
    data_dir = getattr(args, "data_dir", None) or repo_root / "data"
    manifest = args.manifest or data_dir / "manifest.csv"

    if args.command == "build-manifest":
        result = build_manifest(data_dir, manifest, repo_root, seed=args.seed)
        included = sum(1 for row in result.rows if row.split != "excluded")
        print(f"manifest: {manifest}")
        print(f"rows: {len(result.rows)} total, {included} included")
        print(f"this run: {result.new_count} new, {result.excluded_count} excluded as duplicates")
        for path in result.missing_paths:
            print(f"WARNING: in manifest but missing on disk: {path}", file=sys.stderr)
        reports_dir = args.reports_dir or repo_root / "reports"
        write_review_report(result.review_pairs, repo_root, reports_dir)
        if result.review_pairs:
            print(
                f"REVIEW NEEDED: {len(result.review_pairs)} cross-label near-duplicate pairs -> "
                f"{reports_dir / 'near_dup_review.csv'} (+ .png contact sheet)"
            )
    elif args.command == "eda":
        out_dir = args.out_dir or repo_root / "reports" / "eda"
        run_eda(manifest, repo_root, out_dir)
        print(f"EDA artifacts written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
