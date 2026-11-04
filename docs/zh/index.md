# 回归分析学习笔记

这是一份关于回归分析的个人学习笔记。内容从经典线性模型出发，逐步推广到惩罚回归、广义线性模型和分位数回归。每一章都会先给出模型假设，再推导主要结论，并说明如何解释和检验拟合得到的模型；最后一部分汇集了 Python 和 R 代码。

这些笔记最初写于 2022 年至 2023 年，以系列博客的形式发布在[博客园（cnblogs）](https://www.cnblogs.com/veager/articles/15901917.html)上，内容包括理论、证明以及 Python 和 R 实现。在整理成书的修订中，统一了符号，补全了证明并将其放在相应结论之后，更正了错误，测试了代码，新增了若干章节，并翻译为中文。

本笔记于 2026 年 11 月在 Claude Opus 5.5（Anthropic）的协助下完成内容审阅、修订与整理，译为简体中文，并构建了网站。

作者水平有限，书中难免存在疏漏与错误，恳请读者不吝指正。如发现任何问题，欢迎通过 [GitHub Issue](https://github.com/betterweizh/lecture-notes-linear-regression/issues) 提交反馈，共同完善这份笔记。

:::{admonition} English edition
:class: tip
本书提供完整的英文版。点击任意页面右上角的翻译图标 {fas}`language`，即可打开另一种语言中的同一章节。
:::

## 本书结构

- **第一部分** 回顾全书用到的概率论、统计学、线性代数和矩阵微积分知识，以及相关分析和拟合优度检验。
- **第二部分** 讨论线性回归模型：模型假设、参数估计、最小二乘的几何意义与抽样性质、方差分析、假设检验、单因素试验（固定效应、随机效应和区组设计）、预测与模型诊断。
- **第三部分** 讨论实际建模中的问题：标准化回归系数、多重共线性、预测变量的选择，以及岭回归和 lasso 回归。
- **第四部分** 介绍广义线性模型（GLM），并将其应用于二元、计数、有序类别和多项选择数据。
- **第五部分** 介绍分位数回归，将研究对象从条件均值扩展到条件分位数。
- **第六部分** 用 Python 和 R 实现上述方法。

建议按顺序阅读；每章概述中注明了本章所依据的前面章节。第 1 章至第 28 章连续编号，公式、图、表、例子和结论在各章内编号，公式如 (8.1)、(8.2)。证明默认折叠，点击 **证明** 即可展开，建议先自己尝试证明，再阅读证明。{doc}`glossary` 汇总了主要术语的中英文对照，{doc}`references` 列出了全部引用文献。

## 主要符号

多章通用的符号见 {doc}`notation`。

## 许可证

本笔记的文本和代码以 [MIT 许可证](https://github.com/betterweizh/lecture-notes-linear-regression/blob/main/LICENSE)发布。在保留版权声明"Copyright (c) 2026 Better W.Z."和许可证全文的前提下，可以使用、复制、修改和分发。

## 引用方式

如使用本笔记，请按如下方式引用（APA 格式）：

> Better W.Z. (2026). *Study notes on linear regression* [Study notes].
> <https://github.com/betterweizh/lecture-notes-linear-regression/>

```bibtex
@misc{betterwz2026regression,
  author       = {Better Z.W.},
  title        = {Study Notes on Linear Regression},
  year         = {2026},
  howpublished = {\url{https://github.com/betterweizh/lecture-notes-linear-regression/}},
  note         = {Source: \url{https://github.com/betterweizh/lecture-notes-linear-regression}}
}
```

## 参考文献

各章引用的文献列在该章末尾，并按书籍、研究论文和在线文档分类汇总于 {doc}`references`。

## 目录

```{toctree}
:caption: 第一部分 — 预备知识
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
:caption: 第二部分 — 线性回归
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
:caption: 第三部分 — 模型构建
:numbered:
:maxdepth: 1

14-standardization-and-multicollinearity
15-variable-selection
16-ridge-and-lasso
```

```{toctree}
:caption: 第四部分 — 广义线性模型
:numbered:
:maxdepth: 1

17-generalized-linear-models
18-logistic-regression
19-poisson-and-count-regression
20-ordinal-regression
21-multinomial-choice-models
```

```{toctree}
:caption: 第五部分 — 超越条件均值
:numbered:
:maxdepth: 1

22-quantile-regression
```

```{toctree}
:caption: 第六部分 — Python 与 R 实现
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
:caption: 附录
:maxdepth: 1

notation
glossary
references
```
