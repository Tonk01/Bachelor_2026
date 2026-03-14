from collections import Counter
from .data_loader import load_all_events

def _sanity_check():
    events, errors = load_all_events(
        "src/data/raw/raw1",
        "src/data/raw/raw2",
        limit = 5
    )

    print("events loaded", len(events))
    
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

        e = events[0]

    for e in events:
        print("eventid", e.eventid)
        print("samplerate", e.samplerate)
        print("samples", e.n_samples)
        print("duration", e.duration_sec)
        print("sitename", e.sitename)
        print("valvetag", e.valvetag)
        print()

if __name__ == "__main__":
    _sanity_check()


