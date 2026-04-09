from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import re
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

@dataclass
class EventMetaData:
    path: str
    samplerate: int
    n_samples: int
    duration_sec: int


SAMPLERATE_PATTERN = re.compile(r'"samplerate"\s*:\s*(\d+)')


def _is_event_file(path: Path) -> bool:
    # Support both direct ".../events" roots and higher-level dataset roots
    # that contain site folders with nested "events" directories.
    return path.suffix.lower() == ".json" and "events" in path.parts


def peek_event_samplerate(path: Path, max_lines: int = 64) -> int | None:
    try:
        with path.open("r", encoding="utf-8") as file:
            for index, line in enumerate(file):
                match = SAMPLERATE_PATTERN.search(line)
                if match is not None:
                    return int(match.group(1))

                if index + 1 >= max_lines:
                    break
    except OSError:
        return None

    return None

def peak_event_metadata(path: Path) -> EventMetaData | None:
    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)

            samplerate = int(data["samplerate"])
            eventdata = data["eventdata"]

            if samplerate <= 0:
                return None
            
            if not isinstance(eventdata, list):
                return None
            
            n_samples = len(eventdata)

            if n_samples == 0:
                return None
            
            duration_sec = n_samples / samplerate

            return EventMetaData(
                path=str(path),
                samplerate=samplerate,
                n_samples=n_samples,
                duration_sec=duration_sec,
            )
        
    except Exception:
        return None

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

        if root_path.is_file():
            if _is_event_file(root_path):
                normalized_path = root_path.absolute()
                if normalized_path not in seen:
                    seen.add(normalized_path)
                    yield root_path
            continue

        for path in root_path.rglob("*.json"):
            if not _is_event_file(path):
                continue

            normalized_path = path.absolute()

            if normalized_path in seen:
                continue

            seen.add(normalized_path)
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
