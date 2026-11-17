"""Automated test runner for the Recommendation Engine Prototype.

Ensures environment flags (such as disabling autoload of OS-restricted C-extension plugins)
are properly set prior to invoking the pytest test suite.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Enforce disabling plugin autoload to bypass host WDAC DLL restrictions
os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"

if __name__ == "__main__":
    import pytest
    args = sys.argv[1:] if len(sys.argv) > 1 else [str(PROJECT_ROOT / "tests"), "-v"]
    exit_code = pytest.main(args)
    sys.exit(exit_code)
