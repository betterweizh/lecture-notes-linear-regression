# Study Notes on Linear Regression

These are personal study notes on regression analysis. They develop the
subject from the classical linear model to penalized regression,
generalized linear models, and quantile regression. Each chapter states its
assumptions, derives the main results, and explains how to interpret and
check a fitted model; the final part collects the Python and R code.

The notes were first written between 2022 and 2023 and published as a blog
series on [cnblogs](https://www.cnblogs.com/veager/articles/15901917.html),
covering theory, proofs, and implementations in Python and R. In the
revision that turned them into this book, the notation was standardized, proofs were
completed and placed next to the results they support, errors were
corrected, the code was tested, new chapters were added, and the text was
translated into Chinese.

In November 2026, the content of these notes was reviewed, revised, and
organized, translated into Simplified Chinese, and the website was built,
with the assistance of Claude Opus 5.5 (Anthropic).

The author's knowledge is limited, and errors and omissions are inevitable;
corrections from readers are sincerely welcome. If you find any problem,
please report it through a [GitHub Issue](https://github.com/betterweizh/lecture-notes-linear-regression/issues) to help improve these
notes.

:::{admonition} 中文版
:class: tip
A complete Simplified Chinese edition is available. Click the translate
icon {fas}`language` at the top right of any page to open the same chapter
in the other language.
:::

## How the book is organized

- **Part I** reviews the probability, statistics, linear algebra, and
  matrix calculus used throughout the book, together with correlation
  analysis and goodness-of-fit tests.
- **Part II** develops the linear regression model: assumptions,
  estimation, the geometry and sampling properties of least squares,
  analysis of variance, hypothesis tests, single-factor experiments
  (fixed effects, random effects, and blocking), prediction, and
  diagnostics.
- **Part III** covers practical model building: standardized coefficients,
  multicollinearity, the selection of predictors, and ridge and lasso
  regression.
- **Part IV** introduces generalized linear models (GLMs) and applies them
  to binary, count, ordered, and multinomial responses.
- **Part V** moves beyond the conditional mean with quantile regression.
- **Part VI** implements the methods in Python and R.

The chapters are written to be read in order; the overview of each chapter
names the earlier chapters it builds on. Chapters are numbered 1 to 28 across
the book, and equations, figures, tables, examples, and results are numbered within
each chapter (equations as (8.1), (8.2), and so on). Proofs are folded: click
a **Proof** bar to open one, and try to prove each result before reading the
proof. The {doc}`glossary` collects the English and Chinese names of the main
terms, and the {doc}`references` lists all works cited.

## Notation

The symbols used in several chapters are listed in {doc}`notation`.

## License

The text and code of these notes are released under the
[MIT License](https://github.com/betterweizh/lecture-notes-linear-regression/blob/main/LICENSE).
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

## References

The works cited in each chapter are listed at the end of that chapter and
collected, grouped into books, research articles, and online documents, in
the {doc}`references`.

## Table of contents

```{toctree}
:caption: Part I — Preliminaries
:numbered:
:maxdepth: 1

01-probability-and-random-vectors
02-sampling-distributions-and-limit-theorems
03-correlation-analysis
04-goodness-of-fit-tests
05-linear-algebra
06-matrix-calculus
```

```{toctree}
:caption: Part II — Linear Regression
:numbered:
:maxdepth: 1

07-linear-regression-model
08-ols-and-ml-estimation
09-ols-properties-and-geometry
10-anova-and-inference
11-one-way-anova
12-random-effects-and-blocking
13-prediction-and-model-assessment
```

```{toctree}
:caption: Part III — Model Building
:numbered:
:maxdepth: 1

14-standardization-and-multicollinearity
15-variable-selection
16-ridge-and-lasso
```

```{toctree}
:caption: Part IV — Generalized Linear Models
:numbered:
:maxdepth: 1

17-generalized-linear-models
18-logistic-regression
19-poisson-and-count-regression
20-ordinal-regression
21-multinomial-choice-models
```

```{toctree}
:caption: Part V — Beyond the Conditional Mean
:numbered:
:maxdepth: 1

22-quantile-regression
```

```{toctree}
:caption: Part VI — Implementation in Python and R
:numbered:
:maxdepth: 1

23-correlation-and-goodness-of-fit-code
24-linear-regression-python
25-linear-regression-r
26-penalized-regression-python
27-glm-python
28-quantile-regression-python
```

```{toctree}
:caption: Appendix
:maxdepth: 1

notation
glossary
references
```
