from __future__ import annotations

import argparse
import csv
from datetime import datetime
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "train_critical_points_3sensor.py"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an automatic hyperparameter grid search for 3-sensor training."
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[8, 16, 24],
        help="Batch sizes to try.",
    )
    parser.add_argument(
        "--learning-rates",
        type=float,
        nargs="+",
        default=[1e-3, 5e-4, 2e-4],
        help="Learning rates to try.",
    )
    parser.add_argument(
        "--dropouts",
        type=float,
        nargs="+",
        default=[0.1, 0.2],
        help="Dropout values to try.",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Epochs per run.")
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=0,
        help="Training sample cap per run. Use 0 for full split.",
    )
    parser.add_argument(
        "--max-val-samples",
        type=int,
        default=0,
        help="Validation sample cap per run. Use 0 for full split.",
    )
    parser.add_argument(
        "--max-test-samples",
        type=int,
        default=0,
        help="Test sample cap per run. Use 0 for full split.",
    )
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=2,
        help="DataLoader prefetch factor.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed forwarded to the training script.",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        default=True,
        help="Use cache during training.",
    )
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        default=False,
        help="Rebuild cache before each run.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional explicit cache directory.",
    )
    parser.add_argument(
        "--target-strategy",
        type=str,
        default=None,
        help="Optional TARGET_STRATEGY forwarded to the training script.",
    )
    parser.add_argument(
        "--dataset-mode",
        type=str,
        default=None,
        help="Optional DATASET_MODE forwarded to the training script.",
    )
    parser.add_argument(
        "--multisensor-dataset-path",
        type=Path,
        default=None,
        help="Optional MULTISENSOR_DATASET_PATH forwarded to the training script.",
    )
    parser.add_argument(
        "--pressure-dataset-path",
        type=Path,
        default=None,
        help="Optional PRESSURE_DATASET_PATH forwarded to the training script.",
    )
    parser.add_argument(
        "--two-sensor-dataset-path",
        type=Path,
        default=None,
        help="Optional TWO_SENSOR_DATASET_PATH forwarded to the training script.",
    )
    parser.add_argument(
        "--manual-labels-csv",
        type=Path,
        default=None,
        help="Optional MANUAL_LABELS_CSV forwarded to the training script.",
    )
    parser.add_argument(
        "--manual-split-json",
        type=Path,
        default=None,
        help="Optional MANUAL_SPLIT_JSON forwarded to the training script.",
    )
    parser.add_argument(
        "--stop-on-fail",
        action="store_true",
        help="Stop immediately if one run fails.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned runs without executing them.",
    )
    return parser.parse_args()


def _artifact_dirs() -> set[Path]:
    return {
        path.resolve()
        for path in ARTIFACTS_DIR.glob("valve-cnn-3sensor-*")
        if path.is_dir()
    }


def _find_new_run_dir(before: set[Path], after: set[Path]) -> Path | None:
    new_dirs = sorted(after - before, key=lambda path: path.stat().st_mtime)
    if new_dirs:
        return new_dirs[-1]
    return None


def _load_summary(run_dir: Path | None) -> dict | None:
    if run_dir is None:
        return None
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return None
    with summary_path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def _scan_output_paths(scan_name: str) -> tuple[Path, Path]:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    return (
        ARTIFACTS_DIR / f"{scan_name}.json",
        ARTIFACTS_DIR / f"{scan_name}.csv",
    )


def main() -> None:
    args = parse_args()

    combinations = list(
        itertools.product(args.batch_sizes, args.learning_rates, args.dropouts)
    )
    scan_name = datetime.now().strftime("grid-search-3sensor-%Y%m%d-%H%M%S")
    json_path, csv_path = _scan_output_paths(scan_name)

    print("3-sensor grid search")
    print("Project root:", PROJECT_ROOT)
    print("Training script:", TRAIN_SCRIPT)
    print("Planned runs:", len(combinations))
    print("JSON summary:", json_path)
    print("CSV summary:", csv_path)
    print()

    results: list[dict[str, object]] = []

    for run_index, (batch_size, learning_rate, dropout) in enumerate(combinations, start=1):
        run_label = (
            f"[{run_index}/{len(combinations)}] "
            f"BATCH_SIZE={batch_size} LEARNING_RATE={learning_rate} DROPOUT={dropout}"
        )
        print(run_label)

        env = os.environ.copy()
        env.update(
            {
                "USE_CACHE": "1" if args.use_cache else "0",
                "REBUILD_CACHE": "1" if args.rebuild_cache else "0",
                "SEED": str(args.seed),
                "WORKERS": str(args.workers),
                "PREFETCH_FACTOR": str(args.prefetch_factor),
                "EPOCHS": str(args.epochs),
                "BATCH_SIZE": str(batch_size),
                "LEARNING_RATE": str(learning_rate),
                "DROPOUT": str(dropout),
                "MAX_TRAIN_SAMPLES": str(args.max_train_samples),
                "MAX_VAL_SAMPLES": str(args.max_val_samples),
                "MAX_TEST_SAMPLES": str(args.max_test_samples),
            }
        )
        if args.cache_dir is not None:
            env["CACHE_DIR"] = str(args.cache_dir)
        if args.target_strategy is not None:
            env["TARGET_STRATEGY"] = args.target_strategy
        if args.dataset_mode is not None:
            env["DATASET_MODE"] = args.dataset_mode
        if args.multisensor_dataset_path is not None:
            env["MULTISENSOR_DATASET_PATH"] = str(args.multisensor_dataset_path)
        if args.pressure_dataset_path is not None:
            env["PRESSURE_DATASET_PATH"] = str(args.pressure_dataset_path)
        if args.two_sensor_dataset_path is not None:
            env["TWO_SENSOR_DATASET_PATH"] = str(args.two_sensor_dataset_path)
        if args.manual_labels_csv is not None:
            env["MANUAL_LABELS_CSV"] = str(args.manual_labels_csv)
        if args.manual_split_json is not None:
            env["MANUAL_SPLIT_JSON"] = str(args.manual_split_json)

        command = [sys.executable, str(TRAIN_SCRIPT)]
        if args.dry_run:
            print("  DRY RUN:", " ".join(command))
            continue

        before_dirs = _artifact_dirs()
        started = time.perf_counter()
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=env,
            check=False,
        )
        duration_seconds = time.perf_counter() - started
        after_dirs = _artifact_dirs()

        run_dir = _find_new_run_dir(before_dirs, after_dirs)
        summary = _load_summary(run_dir)

        result = {
            "status": "ok" if completed.returncode == 0 else "failed",
            "returncode": int(completed.returncode),
            "batch_size": int(batch_size),
            "learning_rate": float(learning_rate),
            "dropout": float(dropout),
            "duration_seconds": float(duration_seconds),
            "run_dir": str(run_dir) if run_dir is not None else None,
            "best_val_loss": summary.get("best_val_loss") if summary else None,
            "final_train_loss": summary.get("final_train_loss") if summary else None,
            "final_val_loss": summary.get("final_val_loss") if summary else None,
            "train_samples": summary.get("train_samples") if summary else None,
            "val_samples": summary.get("val_samples") if summary else None,
            "test_samples": summary.get("test_samples") if summary else None,
        }
        results.append(result)

        print(
            "  status:",
            result["status"],
            "| returncode:",
            result["returncode"],
            "| run_dir:",
            result["run_dir"],
            "| best_val_loss:",
            result["best_val_loss"],
        )
        print()

        if completed.returncode != 0 and args.stop_on_fail:
            break

    with json_path.open("w", encoding="utf-8") as fp:
        json.dump(
            {
                "scan_name": scan_name,
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "project_root": str(PROJECT_ROOT),
                "train_script": str(TRAIN_SCRIPT),
                "batch_sizes": args.batch_sizes,
                "learning_rates": args.learning_rates,
                "dropouts": args.dropouts,
                "epochs": args.epochs,
                "max_train_samples": args.max_train_samples,
                "max_val_samples": args.max_val_samples,
                "max_test_samples": args.max_test_samples,
                "workers": args.workers,
                "prefetch_factor": args.prefetch_factor,
                "use_cache": args.use_cache,
                "rebuild_cache": args.rebuild_cache,
                "cache_dir": str(args.cache_dir) if args.cache_dir is not None else None,
                "target_strategy": args.target_strategy,
                "dataset_mode": args.dataset_mode,
                "multisensor_dataset_path": (
                    str(args.multisensor_dataset_path)
                    if args.multisensor_dataset_path is not None
                    else None
                ),
                "pressure_dataset_path": (
                    str(args.pressure_dataset_path)
                    if args.pressure_dataset_path is not None
                    else None
                ),
                "two_sensor_dataset_path": (
                    str(args.two_sensor_dataset_path)
                    if args.two_sensor_dataset_path is not None
                    else None
                ),
                "manual_labels_csv": (
                    str(args.manual_labels_csv)
                    if args.manual_labels_csv is not None
                    else None
                ),
                "manual_split_json": (
                    str(args.manual_split_json)
                    if args.manual_split_json is not None
                    else None
                ),
                "results": results,
            },
            fp,
            indent=2,
        )

    fieldnames = [
        "status",
        "returncode",
        "batch_size",
        "learning_rate",
        "dropout",
        "duration_seconds",
        "run_dir",
        "best_val_loss",
        "final_train_loss",
        "final_val_loss",
        "train_samples",
        "val_samples",
        "test_samples",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    if not args.dry_run and results:
        successful = [row for row in results if row["status"] == "ok" and row["best_val_loss"] is not None]
        if successful:
            best = min(successful, key=lambda row: float(row["best_val_loss"]))
            print("Best run:")
            print(json.dumps(best, indent=2))
        else:
            print("No successful runs with summary metrics were recorded.")


if __name__ == "__main__":
    main()
