"""Compatibility wrapper.

The runtime entrypoint was moved to apps/main.py.
"""

from apps.main import main


if __name__ == "__main__":
    main()
