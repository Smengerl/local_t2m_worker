"""Shared test fixtures.

Adds the project root to ``sys.path`` so ``import batch`` / ``import
pipeline_config`` work when pytest is run from anywhere.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
