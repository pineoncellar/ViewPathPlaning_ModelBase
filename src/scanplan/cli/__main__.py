"""`python -m scanplan.cli` is the same entry point as the console script."""

from __future__ import annotations

import sys

from . import main

sys.exit(main())
