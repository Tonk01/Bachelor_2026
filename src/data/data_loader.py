from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional
import json

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
    
def find_event_files(*roots: str | Path) -> list[Path]:
    event_files = []

    for root in roots:
        root = Path(root)
        
        for path in root.rglob("*.json"):
            if "events" in path.parts and "eventids" not in path.parts:
                event_files.append(path)
            
    return event_files

def load_all_events(*roots, limit = None):
    events = []
    errors = []

    for path in find_event_files(*roots):
        if limit is not None and len(events) >= limit:
            return events, errors
            
        try:     
            events.append(load_event(path))
        except Exception as e:
            errors.append((str(path), str(e)))

    return events, errors
