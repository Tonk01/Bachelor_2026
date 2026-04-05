from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.build_cache import build_cache_roots
from src.data.data_loader import find_event_files


def main() -> None:
    raw_root = PROJECT_ROOT / "src" / "data" / "raw"
    cache_dir = PROJECT_ROOT / "src" / "data" / "cache"

    print("Building cache...")
    print("Raw:", raw_root)
    print("Cache:", cache_dir)

    raw_paths = find_event_files(raw_root)
    print(f"Event files found: {len(raw_paths)}")

    saved, errors = build_cache_roots(
        raw_root,
        cache_dir=cache_dir,
    )

    print(f"Saved: {len(saved)}")
    print(f"Errors: {len(errors)}")

    if errors:
        print("Some errors occurred:")
        for path, err in errors[:5]:
            print(path, "->", err)


if __name__ == "__main__":
    main()