from collections import Counter
from pathlib import Path

from .data_loader import load_all_events


DEFAULT_ROOTS = [
    Path("src/data/raw/raw1"),
    Path("src/data/raw/raw2"),
    Path("/mnt/d/Bachelor_data/data/raw_1sensor"),
]

DEFAULT_LIMIT = 5
DEFAULT_SHOW_EVENTS = 5


def _existing_roots() -> list[Path]:
    return [root for root in DEFAULT_ROOTS if root.exists()]


def _sanity_check() -> None:
    roots = _existing_roots()
    if not roots:
        print("No data roots found.")
        return

    events, errors = load_all_events(*roots, limit=DEFAULT_LIMIT)

    print("roots")
    for root in roots:
        print(root)
    print()

    print("events loaded", len(events))
    print("errors", len(errors))
    print()

    if not events:
        return

    samplerates = Counter(e.samplerate for e in events)

    print("sample rate distribution")
    for rate, count in samplerates.most_common():
        print(f"{rate} Hz: {count}")
    print()

    duration = [e.duration_sec for e in events]
    print("Durations:")
    print("min", min(duration))
    print("max", max(duration))
    print("avg", sum(duration) / len(duration))
    print()

    length = [e.n_samples for e in events]
    print("Sample length")
    print("min", min(length))
    print("max", max(length))
    print("avg", sum(length) / len(length))
    print()

    missing_site = sum(1 for e in events if not e.sitename)
    missing_valve = sum(1 for e in events if not e.valvetag)

    print("missing sitename", missing_site)
    print("missing valvetag", missing_valve)
    print()

    for e in events[:DEFAULT_SHOW_EVENTS]:
        print("eventid", e.eventid)
        print("samplerate", e.samplerate)
        print("samples", e.n_samples)
        print("duration", e.duration_sec)
        print("sitename", e.sitename)
        print("valvetag", e.valvetag)
        print("path", e.path)
        print()

    if errors:
        print("example errors")
        for path, message in errors[:DEFAULT_SHOW_EVENTS]:
            print(path)
            print(message)
            print()


if __name__ == "__main__":
    _sanity_check()
