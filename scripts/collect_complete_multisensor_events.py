from __future__ import annotations

import argparse
import shutil
from pathlib import Path


DEFAULT_PRESSURE_ROOT = Path("/mnt/d/Bachelor_data/data/raw_1sensor")
DEFAULT_TWO_SENSOR_ROOT = Path("/mnt/d/Bachelor_data/data/raw_2SENSOR")
DEFAULT_OUTPUT_ROOT = Path("data/raw_complete_3sensor")
SENSOR_NAMES = ("pressure", "strain", "travel")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect events that have pressure, strain, and travel into one dataset root."
        )
    )
    parser.add_argument(
        "--pressure-root",
        default=str(DEFAULT_PRESSURE_ROOT),
        help="Root containing one-sensor pressure data per site.",
    )
    parser.add_argument(
        "--two-sensor-root",
        default=str(DEFAULT_TWO_SENSOR_ROOT),
        help="Root containing strain and travel data per site.",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Destination root for the collected complete 3-sensor dataset.",
    )
    parser.add_argument(
        "--site",
        action="append",
        default=[],
        help="Optional site to collect. Can be passed multiple times.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of events per site to copy.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be copied without writing files.",
    )
    return parser.parse_args()


def _existing_site_names(pressure_root: Path, two_sensor_root: Path) -> list[str]:
    pressure_sites = {
        path.name
        for path in pressure_root.iterdir()
        if path.is_dir() and (path / "events").exists()
    }
    two_sensor_sites = {
        path.name
        for path in two_sensor_root.iterdir()
        if path.is_dir() and (path / "events").exists()
    }
    return sorted(pressure_sites & two_sensor_sites)


def _resolve_sites(args: argparse.Namespace, pressure_root: Path, two_sensor_root: Path) -> list[str]:
    if args.site:
        return sorted(set(args.site))
    return _existing_site_names(pressure_root, two_sensor_root)


def _sensor_dir(root: Path, site: str, sensor_name: str) -> Path:
    if sensor_name == "pressure":
        return root / site / "events"
    return root / site / "events" / sensor_name


def _sensor_files_by_name(directory: Path) -> dict[str, Path]:
    if not directory.exists():
        return {}
    return {path.name: path for path in directory.glob("*.json")}


def _copy_event_group(
    *,
    site: str,
    filename: str,
    source_paths: dict[str, Path],
    output_root: Path,
    dry_run: bool,
) -> None:
    for sensor_name, source_path in source_paths.items():
        destination = output_root / site / "events" / sensor_name / filename
        if dry_run:
            print(f"would copy {source_path} -> {destination}")
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination)


def collect_site(
    *,
    site: str,
    pressure_root: Path,
    two_sensor_root: Path,
    output_root: Path,
    limit: int | None,
    dry_run: bool,
) -> tuple[int, int, int, int]:
    pressure_files = _sensor_files_by_name(_sensor_dir(pressure_root, site, "pressure"))
    strain_files = _sensor_files_by_name(_sensor_dir(two_sensor_root, site, "strain"))
    travel_files = _sensor_files_by_name(_sensor_dir(two_sensor_root, site, "travel"))

    complete_filenames = sorted(set(pressure_files) & set(strain_files) & set(travel_files))
    if limit is not None:
        complete_filenames = complete_filenames[:limit]

    for filename in complete_filenames:
        _copy_event_group(
            site=site,
            filename=filename,
            source_paths={
                "pressure": pressure_files[filename],
                "strain": strain_files[filename],
                "travel": travel_files[filename],
            },
            output_root=output_root,
            dry_run=dry_run,
        )

    return (
        len(pressure_files),
        len(strain_files),
        len(travel_files),
        len(complete_filenames),
    )


def main() -> None:
    args = parse_args()
    pressure_root = Path(args.pressure_root)
    two_sensor_root = Path(args.two_sensor_root)
    output_root = Path(args.output_root)

    if not pressure_root.exists():
        raise FileNotFoundError(f"Pressure root does not exist: {pressure_root}")
    if not two_sensor_root.exists():
        raise FileNotFoundError(f"Two-sensor root does not exist: {two_sensor_root}")

    sites = _resolve_sites(args, pressure_root, two_sensor_root)
    if not sites:
        raise FileNotFoundError("No sites found to collect.")

    total_complete = 0

    print("pressure root:", pressure_root)
    print("two-sensor root:", two_sensor_root)
    print("output root:", output_root)
    print("sites:", ", ".join(sites))
    print()

    for site in sites:
        pressure_count, strain_count, travel_count, complete_count = collect_site(
            site=site,
            pressure_root=pressure_root,
            two_sensor_root=two_sensor_root,
            output_root=output_root,
            limit=args.limit,
            dry_run=args.dry_run,
        )
        total_complete += complete_count
        print(
            f"[{site}] pressure={pressure_count} strain={strain_count} "
            f"travel={travel_count} complete={complete_count}"
        )

    print()
    if args.dry_run:
        print(f"Dry run complete. Would collect {total_complete} complete events.")
    else:
        print(f"Collection complete. Copied {total_complete} complete events.")


if __name__ == "__main__":
    main()
