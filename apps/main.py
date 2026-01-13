"""apps/main.py

Operational entrypoint used by scheduled jobs / monitoring.

Historically this file executed work at import time. It now exposes a `main()`
function so it can be called safely from wrappers and schedulers.
"""

from __future__ import annotations

from train.monitor import monitor_and_update_model


def main() -> None:
    """Run one monitoring iteration (customize to your needs)."""
    # Default example: update a model once. Replace '^GSPC' and iterations as needed.
    monitor_and_update_model("^GSPC", iterations=1)


if __name__ == "__main__":
    main()
