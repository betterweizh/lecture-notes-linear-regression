"""Chinese names for sphinx-proof blocks in the Chinese edition.

sphinx-proof's own zh_CN catalog writes "例子 1.2"; Chinese textbooks write
"例 1.2". This extension registers a catalog in docs/_shared/locales that
overrides only those names. It must be listed before "sphinx_proof" in the
extensions, because Sphinx keeps the first catalog registered for a domain.
"""

from __future__ import annotations

from pathlib import Path


def setup(app):
    locales = Path(__file__).resolve().parents[1] / "locales"
    app.add_message_catalog("proof", str(locales))
    return {"parallel_read_safe": True, "parallel_write_safe": True}
