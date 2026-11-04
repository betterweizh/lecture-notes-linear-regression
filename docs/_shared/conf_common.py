"""Sphinx configuration shared by the English and Chinese editions.

Each language has its own ``conf.py`` (``docs/en/conf.py`` and
``docs/zh/conf.py``). Both import everything from this module and then set
only the language-specific values (language code, titles, labels).
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import date
from pathlib import Path

SHARED_DIR = Path(__file__).resolve().parent
DOCS_DIR = SHARED_DIR.parent
REPO_ROOT = DOCS_DIR.parent

sys.path.insert(0, str(SHARED_DIR / "_ext"))

# -- Project information -----------------------------------------------------

author = "Better W.Z."
copyright = f"{date.today().year}, {author.strip(".")}"

# The two editions. The key is the output folder name under _build/html/.
LANGUAGES = {
    "en": {"sphinx": "en", "label": "English", "short": "EN", "toc_title": "Contents"},
    "zh": {"sphinx": "zh_CN", "label": "中文", "short": "中文", "toc_title": "本页目录"},
}


def _detect_repository_url() -> str | None:
    """Return the GitHub URL of this repository, if it can be found.

    Order of precedence: the ``DOCS_REPO_URL`` environment variable, the
    ``GITHUB_REPOSITORY`` variable set by GitHub Actions, and finally the
    ``origin`` remote of the local git checkout.
    """
    if os.environ.get("DOCS_REPO_URL"):
        return os.environ["DOCS_REPO_URL"].rstrip("/")
    if os.environ.get("GITHUB_REPOSITORY"):
        return f"https://github.com/{os.environ['GITHUB_REPOSITORY']}"
    try:
        url = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    if url.startswith("git@github.com:"):
        url = "https://github.com/" + url.split(":", 1)[1]
    url = url.removesuffix(".git")
    return url if url.startswith("https://github.com/") else None


REPOSITORY_URL = _detect_repository_url()

# -- General configuration ---------------------------------------------------

extensions = [
    "myst_parser",
    "sphinx.ext.mathjax",
    "proof_zh",
    "sphinx_proof",
    "sphinx_copybutton",
    "sphinx_design",
    "continuous_numbering",
    "lang_switch",
]

source_suffix = {".md": "markdown"}
root_doc = "index"
exclude_patterns = [
    "_build", "Thumbs.db", ".DS_Store", "**/README.md",
    # Safety net: the original notes (file names with spaces) live in sources/;
    # never publish copies left in docs/.
    "* *.md",
]
templates_path = [str(SHARED_DIR / "_templates")]

# MyST Markdown: $...$ / $$...$$ math, ::: fences for directives, etc.
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "smartquotes",
]
myst_heading_anchors = 3
myst_dmath_double_inline = True

# Equations, figures and tables are numbered per chapter (e.g. "(2.3)").
math_numfig = True
numfig = True
numfig_secnum_depth = 1
math_eqref_format = "({number})"

# MathJax 3: shared macros keep the notation identical in both editions.
mathjax3_config = {
    "tex": {
        "macros": {
            "R": r"\mathbb{R}",
            "E": r"\mathbb{E}",
            "Prob": r"\mathbb{P}",
            "Var": r"\operatorname{Var}",
            "Cov": r"\operatorname{Cov}",
            "tr": r"\operatorname{tr}",
            "rank": r"\operatorname{rank}",
            "diag": r"\operatorname{diag}",
            "sign": r"\operatorname{sign}",
            "logit": r"\operatorname{logit}",
            "bX": r"\mathbf{X}",
            "bH": r"\mathbf{H}",
            "bI": r"\mathbf{I}",
            "bW": r"\mathbf{W}",
            "bx": r"\mathbf{x}",
            "by": r"\mathbf{y}",
            "bY": r"\mathbf{Y}",
            "be": r"\mathbf{e}",
            "bv": r"\mathbf{v}",
            "bz": r"\mathbf{z}",
            "bbeta": r"\boldsymbol{\beta}",
            "bvarepsilon": r"\boldsymbol{\varepsilon}",
            "bmu": r"\boldsymbol{\mu}",
            "bSigma": r"\boldsymbol{\Sigma}",
            "bzero": r"\mathbf{0}",
            "T": r"^{\top}",
        }
    }
}

copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

# -- HTML output -------------------------------------------------------------

html_theme = "sphinx_book_theme"
html_static_path = [str(SHARED_DIR / "_static")]
html_css_files = ["custom.css"]
html_js_files = ["lang-switch.js", "open-dropdown.js"]
html_last_updated_fmt = "%Y-%m-%d"
html_copy_source = True
html_show_sourcelink = False

html_theme_options = {
    "home_page_in_toc": True,
    "show_toc_level": 2,
    "show_navbar_depth": 1,
    "navigation_with_keys": False,
    "use_download_button": True,
    "use_fullscreen_button": True,
    # The language switch sits in the header next to the standard buttons.
    "article_header_end": ["lang-switch.html", "article-header-buttons.html"],
}

if REPOSITORY_URL:
    html_theme_options.update(
        {
            "repository_url": REPOSITORY_URL,
            "repository_branch": "main",
            "use_repository_button": True,
            "use_issues_button": True,
            "use_edit_page_button": True,
        }
    )


def apply_language(lang: str, conf_globals: dict) -> None:
    """Set the values that differ between the editions.

    Call this at the end of each language's ``conf.py``::

        apply_language("en", globals())
    """
    info = LANGUAGES[lang]
    conf_globals["language"] = info["sphinx"]
    conf_globals["lang_switch_current"] = lang
    conf_globals["lang_switch_languages"] = LANGUAGES
    options = dict(conf_globals["html_theme_options"])
    options["toc_title"] = info["toc_title"]  # heading of the right-hand page TOC
    if REPOSITORY_URL:
        options["path_to_docs"] = f"docs/{lang}"
    conf_globals["html_theme_options"] = options
