#!/usr/bin/env python3
"""Convenience entry point: ``python scripts/run_pipeline.py run [options]``.

Equivalent to the ``dnlssm-run`` console script installed by ``pip install
-e .``; provided for users who have installed the dependencies but not the
package itself. See ``dnlssm.cli`` for the full pipeline implementation.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dnlssm.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
