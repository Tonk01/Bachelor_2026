from collections import Counter
from pathlib import Path

from .data_loader import load_all_events
from .preprocess import EventProcessor


DEFAULT_ROOTS = [
    Path("src/data/raw/raw1"),
    Path("src/data/raw/raw2"),
    Path("/mnt/d/Bachelor_data/data/raw_1sensor"),
]

DEFAULT_LIMIT = 100
DEFAULT_SHOW_EVENTS = 5


def _existing_roots() -> list[Path]:
    return [root for root in DEFAULT_ROOTS if root.exists()]


def _run_preprocess() -> None:
    roots = _existing_roots()
    if not roots:
        print("No data roots found.")
        return

    events, errors = load_all_events(*roots, limit=DEFAULT_LIMIT)

    print("roots")
    for root in roots:
        print(root)
    print()

    print("loaded events", len(events))
    print("load errors", len(errors))
    print("original samplerates")
    print(Counter(e.samplerate for e in events))
    print()

    processor = EventProcessor()
    processed, preprocess_errors = processor.preprocess_all(events)

    print("processed events", len(processed))
    print("preprocess errors", len(preprocess_errors))
    print("processed samplerates")
    print(Counter(e.samplerate for e in processed))
    print()

    if processed:
        print("processed lengths")
        print(Counter(e.n_samples for e in processed).most_common(10))
        print()

        for e in processed[:DEFAULT_SHOW_EVENTS]:
            print("eventid", e.eventid)
            print("samplerate", e.samplerate)
            print("samples", e.n_samples)
            print("duration", e.duration_sec)
            print("sitename", e.sitename)
            print("valvetag", e.valvetag)
            print("path", e.path)
            print()

    if preprocess_errors:
        print("example preprocess errors")
        for path, message in preprocess_errors[:DEFAULT_SHOW_EVENTS]:
            print(path)
            print(message)
            print()


if __name__ == "__main__":
    _run_preprocess()
