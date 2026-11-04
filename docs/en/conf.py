"""Sphinx configuration for the English edition."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_shared"))

from conf_common import *  # noqa: E402,F401,F403
from conf_common import apply_language  # noqa: E402

project = "Regression Analysis: Lecture Notes"
html_title = "Regression Analysis"

apply_language("en", globals())
