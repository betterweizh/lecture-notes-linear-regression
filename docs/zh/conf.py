"""Sphinx configuration for the Chinese (Simplified) edition / 中文版配置。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_shared"))

from conf_common import *  # noqa: E402,F401,F403
from conf_common import apply_language  # noqa: E402

project = "回归分析讲义"
html_title = "回归分析讲义"

# Chinese search: Sphinx splits words with jieba when it is installed;
# otherwise it falls back to single-character search, which still works.

apply_language("zh", globals())
