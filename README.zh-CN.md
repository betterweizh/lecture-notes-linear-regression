# 回归分析学习笔记 · Study Notes on Linear Regression

[English](README.md) | **中文** | [Read online](https://betterweizh.github.io/lecture-notes-linear-regression/en/) | [在线阅读](https://betterweizh.github.io/lecture-notes-linear-regression/zh)

中英双语的回归分析学习笔记，内容从经典线性模型延伸到广义线性模型和分位数回归。
笔记用 Markdown（MyST）编写，通过 [Sphinx](https://www.sphinx-doc.org/) 和 [sphinx-book-theme](https://sphinx-book-theme.readthedocs.io/)
（与 [Shapely 文档](https://shapely.readthedocs.io/) 相同的主题）发布为网站。每个页面右上角都有一个翻译图标，
与其他页眉按钮并列，点击即可打开另一种语言中的同一章节。

## 目录

| 部分 | 章节 |
|---|---|
| 一 · 预备知识 | 1 概率与随机向量 · 2 抽样分布与极限定理 · 3 相关分析 · 4 拟合优度检验与 Q–Q 图 · 5 线性代数回顾 · 6 向量与矩阵求导 |
| 二 · 线性回归 | 7 模型与假设 · 8 最小二乘估计与极大似然估计 · 9 最小二乘的几何意义与统计性质 · 10 方差分析与统计推断 · 11 单因素方差分析 · 12 随机效应与区组设计 · 13 预测与模型评价 |
| 三 · 模型构建 | 14 标准化与多重共线性 · 15 变量选择与子集选择 · 16 岭回归与 lasso 回归 |
| 四 · 广义线性模型 | 17 广义线性模型 · 18 logistic 回归 · 19 泊松回归与计数回归 · 20 有序回归 · 21 多项选择模型 |
| 五 · 超越条件均值 | 22 分位数回归 |
| 六 · Python 与 R 实现 | 23 用 R 与 Python 实现相关分析和拟合优度检验 · 24 用 Python 实现线性回归 · 25 用 R 实现线性回归 · 26 用 Python 实现岭回归与 lasso 回归 · 27 用 Python 实现广义线性模型 · 28 用 Python 实现分位数回归 |
| 附录 | 中英术语表 · 参考文献（全部引用文献，按类型分类） |

## 仓库结构

```text
docs/
  _shared/            两个语言版本共用的配置、扩展、模板和样式
    conf_common.py    Sphinx 设置（主题、MyST、MathJax 宏、语言切换）
    _ext/             continuous_numbering.py（跨部分连续编号）、lang_switch.py（语言切换）、proof_zh.py（证明块的中文名称）
    _templates/       lang-switch.html（页眉中的翻译图标）
    _static/          custom.css、lang-switch.js、open-dropdown.js
    locales/          证明块的中文名称（proof.po，编译为 proof.mo）
  en/                 英文版：conf.py、index.md、每章一个 NN-<主题>.md（NN 为章节号）、glossary.md
  zh/                 中文版：文件名与 en/ 完全相同
scripts/build_docs.py 将两个版本构建到 _build/html/{en,zh}/
scripts/references.py  生成各章 APA 格式参考文献及参考文献页
sources/             原始笔记与讲义（已加入 .gitignore，仅保存在本地）
.github/workflows/    自动构建并部署到 GitHub Pages
```

## 本地构建

需要 Python 3.10 或更高版本。

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

python scripts/build_docs.py --serve    # 构建两个版本并在 http://127.0.0.1:8000/ 预览
python scripts/build_docs.py --strict   # 出现任何警告即失败（CI 使用此模式）
python scripts/build_docs.py --lang zh  # 只重新构建一个版本
```

## 发布到 GitHub Pages

1. 在 GitHub 上新建一个空仓库（不要勾选 README 和许可证）。
2. 推送本仓库：

   ```bash
   git remote add origin https://github.com/<你的账号>/lecture-notes-linear-regression.git
   git push -u origin main
   ```

3. 在 GitHub 仓库中打开 **Settings → Pages**，将 **Source** 设为 **GitHub Actions**。

此后每次推送到 `main` 都会自动重新构建网站，地址为 `https://<你的账号>.github.io/lecture-notes-linear-regression/`；根页面会根据浏览器语言和读者上次的选择跳转到 `/en/` 或 `/zh/`。

## 编辑流程

1. 修改 `docs/en/` 中的某一章，**同时**修改 `docs/zh/` 中的同名文件。两个文件中的公式标签（`(eq-…)`）、定理标签（`:label: thm-…`）和代码应保持一致；语言切换依赖相同的文件名，公式链接在切换后仍能定位到同一公式。
2. 新增或修改参考文献时，编辑 `scripts/references.py` 中的 `REFS`（APA 条目）和 `CHAPTERS`（引用该文献的章节），然后运行 `python scripts/references.py`；它会重写两种语言中每章的参考文献部分和参考文献页。
3. 运行 `python scripts/build_docs.py --strict --serve` 检查页面。
4. 提交并推送，GitHub Actions 会自动发布。

每章包括概述、学习目标、正文、“小结与常见错误”和参考文献；证明与例题的解答默认折叠。`\bX`、`\bbeta`、`\E`、`\Var`、`\T`（转置）等共用 LaTeX 宏定义在 `docs/_shared/conf_common.py` 中。

## 许可证

本笔记的文本和代码以 [MIT 许可证](./LICENSE)发布。在保留版权声明"Copyright (c) 2026 Better W.Z."和许可证全文的前提下，可以使用、复制、修改和分发。

## 引用方式

如使用本笔记，请按如下方式引用（APA 格式）：

> Better W.Z. (2026). *Study notes on linear regression* [Study notes].
> <https://github.com/betterweizh/lecture-notes-linear-regression/>
