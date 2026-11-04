#!/usr/bin/env python3
"""Build the bilingual Sphinx site.

Usage (from the repository root)::

    python scripts/build_docs.py            # build English and Chinese
    python scripts/build_docs.py --strict   # treat warnings as errors (CI)
    python scripts/build_docs.py --lang zh  # build one edition only
    python scripts/build_docs.py --serve    # build, then serve on :8000
    python scripts/build_docs.py --clean    # remove _build first

Output::

    _build/html/index.html   redirects to /en/ or /zh/ (reader's preference)
    _build/html/en/          English edition
    _build/html/zh/          Chinese edition
"""

from __future__ import annotations

import argparse
import functools
import http.server
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS = REPO_ROOT / "docs"
OUT = REPO_ROOT / "_build" / "html"
LANGS = ("en", "zh")

ROOT_INDEX = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Study Notes on Linear Regression · 回归分析学习笔记</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link rel="alternate" hreflang="en" href="en/index.html">
  <link rel="alternate" hreflang="zh-CN" href="zh/index.html">
  <noscript><meta http-equiv="refresh" content="0; url=en/index.html"></noscript>
  <script>
    (function () {
      var lang = null;
      try { lang = localStorage.getItem("lnlr-preferred-language"); } catch (e) {}
      if (lang !== "en" && lang !== "zh") {
        var nav = (navigator.languages && navigator.languages[0]) || navigator.language || "";
        lang = nav.toLowerCase().indexOf("zh") === 0 ? "zh" : "en";
      }
      window.location.replace(lang + "/index.html");
    })();
  </script>
  <style>
    body { font-family: system-ui, sans-serif; display: grid; place-items: center;
           min-height: 90vh; margin: 0; background: #fff; color: #222; }
    a { margin: 0 .75rem; font-size: 1.15rem; }
    @media (prefers-color-scheme: dark) { body { background: #14181e; color: #ddd; } a { color: #79b0ef; } }
  </style>
</head>
<body>
  <p><a href="en/index.html">English</a> | <a href="zh/index.html">中文</a></p>
</body>
</html>
"""


def build(lang: str, strict: bool) -> None:
    cmd = [
        sys.executable, "-m", "sphinx",
        "-b", "html",
        str(DOCS / lang),
        str(OUT / lang),
    ]
    if strict:
        cmd[3:3] = ["-W", "--keep-going"]
    print(f"\n=== Building '{lang}' edition ===\n$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def write_root_files() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "index.html").write_text(ROOT_INDEX, encoding="utf-8")
    (OUT / ".nojekyll").write_text("", encoding="utf-8")  # GitHub Pages: keep _static/


def serve(port: int) -> None:
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(OUT))
    with http.server.ThreadingHTTPServer(("127.0.0.1", port), handler) as httpd:
        print(f"\nServing {OUT} at http://127.0.0.1:{port}/  (Ctrl+C to stop)")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lang", choices=LANGS, action="append", help="edition(s) to build (default: all)")
    parser.add_argument("--strict", action="store_true", help="fail on warnings")
    parser.add_argument("--clean", action="store_true", help="delete _build before building")
    parser.add_argument("--serve", action="store_true", help="serve the site after building")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if args.clean and OUT.parent.exists():
        shutil.rmtree(OUT.parent)
    for lang in args.lang or LANGS:
        build(lang, args.strict)
    write_root_files()
    print(f"\nDone. Open {OUT / 'index.html'}")
    if args.serve:
        serve(args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
