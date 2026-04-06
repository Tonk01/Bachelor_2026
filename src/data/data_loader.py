from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Iterator

@dataclass
class Event:
    eventid: int
    valvetag: str
    sitename: str
    samplerate: int
    signal: list[float]
    n_samples: int
    duration_sec: float
    path: str

def load_event(path: Path) -> Event:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

        eventid = int(data["eventid"])
        valvetag = str(data["valvetag"])
        sitename = str(data["sitename"])
        samplerate = int(data["samplerate"])
        signal = [float(x) for x in data["eventdata"]]

        if samplerate <= 0:
            raise ValueError(f"Invalid samplerate in {path}")
        
        n_samples = len(signal)

        if n_samples == 0:
            raise ValueError(f"Empty eventdata in {path}")

        duration_sec = n_samples / samplerate

        return Event(
            eventid = eventid,
            valvetag = valvetag,
            sitename = sitename,
            samplerate = samplerate, 
            signal = signal,
            n_samples = n_samples,
            duration_sec = duration_sec,
            path = str(path),
        )
    
def iter_event_files(*roots: str | Path) -> Iterator[Path]:
    seen: set[Path] = set()

    for root in roots:
        root_path = Path(root)

        if not root_path.exists():
            continue

        for path in root_path.rglob("*.json"):
            resolved_path = path.resolve()

            if resolved_path in seen:
                continue

            seen.add(resolved_path)
            yield path


def find_event_files(*roots: str | Path) -> list[Path]:
    return list(iter_event_files(*roots))


def load_all_events(*roots: str | Path, limit: int | None = None
) -> tuple[list[Event], list[tuple[str, str]]]:
    
    events: list[Event] = []
    errors: list[tuple[str, str]] = []

    for path in iter_event_files(*roots):
        if limit is not None and len(events) >= limit:
            break

        try:
            events.append(load_event(path))
        except Exception as e:
            errors.append((str(path), str(e)))

    return events, errors
