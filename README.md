# Study Notes on Linear Regression · 回归分析学习笔记

[中文说明](README.zh-CN.md)

Bilingual (English / Simplified Chinese) study notes on regression
analysis, from the classical linear model to generalized linear models and
quantile regression. The notes are written in Markdown (MyST) and published
as a website with [Sphinx](https://www.sphinx-doc.org/) and
[sphinx-book-theme](https://sphinx-book-theme.readthedocs.io/), the theme
used by the [Shapely documentation](https://shapely.readthedocs.io/).
Every page has a translate icon at the top right, next to
the other header buttons, that opens the same chapter in the other
language.

## Contents

| Part | Chapters |
|---|---|
| I · Preliminaries | 1 Probability and random vectors · 2 Sampling distributions and limit theorems · 3 Correlation analysis · 4 Goodness-of-fit tests and Q–Q plots · 5 Linear algebra · 6 Matrix calculus |
| II · Linear regression | 7 Model and assumptions · 8 Least squares and maximum likelihood · 9 Geometry and properties of OLS · 10 ANOVA and inference · 11 One-way ANOVA · 12 Random effects and blocking · 13 Prediction and model assessment |
| III · Model building | 14 Standardization and multicollinearity · 15 Variable and subset selection · 16 Ridge and lasso regression |
| IV · Generalized linear models | 17 GLMs · 18 Logistic regression · 19 Poisson and count regression · 20 Ordinal regression · 21 Multinomial choice models |
| V · Beyond the conditional mean | 22 Quantile regression |
| VI · Python and R | 23 Correlation and goodness-of-fit tests in R and Python · 24 Linear regression in Python · 25 Linear regression in R · 26 Ridge and lasso in Python · 27 GLMs in Python · 28 Quantile regression in Python |
| Appendix | Bilingual glossary · References (all works cited, by type) |

## Repository layout

```text
docs/
  _shared/            configuration, extensions, templates and CSS shared by both editions
    conf_common.py    Sphinx settings (theme, MyST, MathJax macros, language switch)
    _ext/             continuous_numbering.py, lang_switch.py, proof_zh.py
    _templates/       lang-switch.html (the translate icon in the page header)
    _static/          custom.css, lang-switch.js, open-dropdown.js
    locales/          Chinese names of the proof blocks (proof.po, compiled to proof.mo)
  en/                 English edition: conf.py, index.md, one NN-<topic>.md per chapter (NN = chapter number), glossary.md
  zh/                 Chinese edition: same file names as en/
scripts/build_docs.py builds both editions into _build/html/{en,zh}/
scripts/run_code_cells.py runs the Python code blocks of the chapters
scripts/references.py  APA reference lists of every chapter and the references page
sources/             original notes and slides (git-ignored; kept locally only)
.github/workflows/    builds and deploys the site to GitHub Pages
```

## Build locally

Requires Python 3.10 or later.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

python scripts/build_docs.py --serve    # build both editions and open http://127.0.0.1:8000/
python scripts/build_docs.py --strict   # fail on any warning (what CI runs)
python scripts/build_docs.py --lang zh  # rebuild one edition only
```

## Editing workflow

1. Edit a chapter in `docs/en/` **and** the same file in `docs/zh/`. Keep
   equation labels (`(eq-…)`), theorem labels (`:label: thm-…`) and code
   identical in both files; the switch button relies on matching file names,
   and equation links keep their position when switching.
2. To add or change a reference, edit `REFS` (the APA entry) and
   `CHAPTERS` (which chapters cite it) in `scripts/references.py`, then run
   `python scripts/references.py`; it rewrites the References section of
   every chapter and the references page in both languages.
3. Run `python scripts/build_docs.py --strict --serve` and check the pages.
4. Commit and push; GitHub Actions publishes the site.

Each chapter has an Overview, learning objectives, its body, a closing
"Summary and Common Mistakes" section, and References. Proofs and the
solutions of examples are folded. Shared LaTeX macros
such as `\bX`, `\bbeta`, `\E`, `\Var` and `\T` (transpose) are defined in
`docs/_shared/conf_common.py`.

## License

The text and code of these notes are released under the
[MIT License](./LICENSE).
You may use, copy, modify, and distribute them, provided that the copyright
notice "Copyright (c) 2026 Better W.Z." and the license text are included.

## How to cite

If you use these notes, please cite them as follows (APA Style):

> Better W.Z. (2026). *Study notes on linear regression* [Study notes].
> <https://github.com/betterweizh/lecture-notes-linear-regression/>

```bibtex
@misc{betterwz2026regression,
  author       = {Better W.Z.},
  title        = {Study Notes on Linear Regression},
  year         = {2026},
  howpublished = {\url{https://github.com/betterweizh/lecture-notes-linear-regression/}},
  note         = {Source: \url{https://github.com/betterweizh/lecture-notes-linear-regression}}
}
```
