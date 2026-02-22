"""
conftest.py – mock heavyweight/side-effectful modules before any test file
imports signals.engine.

train.pipeline reads config.toml and creates a TwelveData client at import
time, which would break tests that have no API key.  We inject MagicMock
stubs into sys.modules so Python returns the stubs instead of actually
executing that module.
"""

import sys
from unittest.mock import MagicMock

_HEAVY_MODULES = [
    "train",
    "train.pipeline",
    "train.core",
    "train.evaluate",
    "train.monitor",
]

for _m in _HEAVY_MODULES:
    sys.modules.setdefault(_m, MagicMock())
