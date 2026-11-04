"""Single source of truth for the bibliography.

Running this script rewrites the "References" section of every chapter in
docs/en and docs/zh from the keys listed in CHAPTERS, and regenerates
docs/{en,zh}/references.md, grouped by category. Entries follow APA Style
(7th edition) and cite whole works: a book rather than one of its chapters,
the home page of a software project rather than one page of its manual.

    python scripts/references.py          # both editions
    python scripts/references.py en       # one edition
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

BOOK, ARTICLE, ONLINE = "book", "article", "online"

REFS = {
    # ---- Books ---------------------------------------------------------
    "agresti": (BOOK, "Agresti, A. (2013). *Categorical data analysis* (3rd ed.). Wiley."),
    "ben_akiva": (BOOK, "Ben-Akiva, M., & Lerman, S. R. (1985). *Discrete choice analysis: Theory and application to travel demand*. MIT Press."),
    "boyd": (BOOK, "Boyd, S., & Vandenberghe, L. (2004). *Convex optimization*. Cambridge University Press. <https://doi.org/10.1017/CBO9780511804441>"),
    "casella": (BOOK, "Casella, G., & Berger, R. L. (2002). *Statistical inference* (2nd ed.). Duxbury."),
    "fox_weisberg": (BOOK, "Fox, J., & Weisberg, S. (2019). *An R companion to applied regression* (3rd ed.). Sage."),
    "harville": (BOOK, "Harville, D. A. (1997). *Matrix algebra from a statistician's perspective*. Springer. <https://doi.org/10.1007/b98818>"),
    "hastie_esl": (BOOK, "Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The elements of statistical learning: Data mining, inference, and prediction* (2nd ed.). Springer. <https://doi.org/10.1007/978-0-387-84858-7>"),
    "heumann": (BOOK, "Heumann, C., Schomaker, M., & Shalabh. (2016). *Introduction to statistics and data analysis: With exercises, solutions and applications in R*. Springer. <https://doi.org/10.1007/978-3-319-46162-5>"),
    "hilbe": (BOOK, "Hilbe, J. M. (2011). *Negative binomial regression* (2nd ed.). Cambridge University Press. <https://doi.org/10.1017/CBO9780511973420>"),
    "hollander": (BOOK, "Hollander, M., Wolfe, D. A., & Chicken, E. (2014). *Nonparametric statistical methods* (3rd ed.). Wiley."),
    "hosmer": (BOOK, "Hosmer, D. W., Lemeshow, S., & Sturdivant, R. X. (2013). *Applied logistic regression* (3rd ed.). Wiley. <https://doi.org/10.1002/9781118548387>"),
    "james": (BOOK, "James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). *An introduction to statistical learning: With applications in R*. Springer. <https://doi.org/10.1007/978-1-4614-7138-7>"),
    "kanji": (BOOK, "Kanji, G. K. (2006). *100 statistical tests* (3rd ed.). Sage."),
    "mccullagh": (BOOK, "McCullagh, P., & Nelder, J. A. (1989). *Generalized linear models* (2nd ed.). Chapman & Hall."),
    "montgomery_doe": (BOOK, "Montgomery, D. C. (2017). *Design and analysis of experiments* (9th ed.). Wiley."),
    "montgomery_runger": (BOOK, "Montgomery, D. C., & Runger, G. C. (2018). *Applied statistics and probability for engineers* (7th ed.). Wiley."),
    "roback": (BOOK, "Roback, P., & Legler, J. (2021). *Beyond multiple linear regression: Applied generalized linear models and multilevel models in R*. CRC Press. <https://bookdown.org/roback/bookdown-BeyondMLR/>"),
    "seber_lee": (BOOK, "Seber, G. A. F., & Lee, A. J. (2003). *Linear regression analysis* (2nd ed.). Wiley. <https://doi.org/10.1002/9780471722199>"),
    "train": (BOOK, "Train, K. E. (2009). *Discrete choice methods with simulation* (2nd ed.). Cambridge University Press. <https://doi.org/10.1017/CBO9780511805271>"),
    # ---- Research articles ---------------------------------------------
    "cook": (ARTICLE, "Cook, R. D. (1977). Detection of influential observation in linear regression. *Technometrics, 19*(1), 15–18. <https://doi.org/10.1080/00401706.1977.10489493>"),
    "frank_hall": (ARTICLE, "Frank, E., & Hall, M. (2001). A simple approach to ordinal classification. In L. De Raedt & P. Flach (Eds.), *Machine learning: ECML 2001* (pp. 145–156). Springer. <https://doi.org/10.1007/3-540-44795-4_13>"),
    "friedman2010": (ARTICLE, "Friedman, J., Hastie, T., & Tibshirani, R. (2010). Regularization paths for generalized linear models via coordinate descent. *Journal of Statistical Software, 33*(1), 1–22. <https://doi.org/10.18637/jss.v033.i01>"),
    "hoerl": (ARTICLE, "Hoerl, A. E., & Kennard, R. W. (1970). Ridge regression: Biased estimation for nonorthogonal problems. *Technometrics, 12*(1), 55–67. <https://doi.org/10.1080/00401706.1970.10488634>"),
    "hurvich": (ARTICLE, "Hurvich, C. M., & Tsai, C.-L. (1989). Regression and time series model selection in small samples. *Biometrika, 76*(2), 297–307. <https://doi.org/10.1093/biomet/76.2.297>"),
    "scipy": (ARTICLE, "Virtanen, P., Gommers, R., Oliphant, T. E., Haberland, M., Reddy, T., Cournapeau, D., Burovski, E., Peterson, P., Weckesser, W., Bright, J., van der Walt, S. J., Brett, M., Wilson, J., Millman, K. J., Mayorov, N., Nelson, A. R. J., Jones, E., Kern, R., Larson, E., … SciPy 1.0 Contributors. (2020). SciPy 1.0: Fundamental algorithms for scientific computing in Python. *Nature Methods, 17*(3), 261–272. <https://doi.org/10.1038/s41592-019-0686-2>"),
    "tibshirani": (ARTICLE, "Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. *Journal of the Royal Statistical Society: Series B (Methodological), 58*(1), 267–288. <https://doi.org/10.1111/j.2517-6161.1996.tb02080.x>"),
    "zou_hastie": (ARTICLE, "Zou, H., & Hastie, T. (2005). Regularization and variable selection via the elastic net. *Journal of the Royal Statistical Society: Series B (Statistical Methodology), 67*(2), 301–320. <https://doi.org/10.1111/j.1467-9868.2005.00503.x>"),
    "zou_df": (ARTICLE, "Zou, H., Hastie, T., & Tibshirani, R. (2007). On the \"degrees of freedom\" of the lasso. *The Annals of Statistics, 35*(5), 2173–2192. <https://doi.org/10.1214/009053607000000127>"),
    # ---- Online documents (lecture notes, web pages, software) ---------
    "apollo": (ONLINE, "Hess, S., & Palma, D. (n.d.). *Apollo: Choice modelling in R* [Computer software]. <http://www.apollochoicemodelling.com/>"),
    "biogeme": (ONLINE, "Bierlaire, M. (n.d.). *Biogeme* [Computer software]. EPFL. <https://biogeme.epfl.ch/>"),
    "he": (ONLINE, "He, Z. [何志坚]. (n.d.). *数理统计讲义* [Lecture notes on mathematical statistics]. <https://bookdown.org/hezhijian/book/>"),
    "islr_pkg": (ONLINE, "James, G., Witten, D., Hastie, T., & Tibshirani, R. (2021). *ISLR: Data for an introduction to statistical learning with applications in R* (R package version 1.4) [Computer software]. <https://CRAN.R-project.org/package=ISLR>"),
    "leaps": (ONLINE, "Lumley, T. (2024). *leaps: Regression subset selection* (R package version 3.2) [Computer software]. <https://CRAN.R-project.org/package=leaps>"),
    "lianxh": (ONLINE, "Lianxh [连享会]. (n.d.). *Stata+R：分位数回归一文读懂* [Quantile regression with Stata and R]. <https://www.lianxh.cn/news/f29d83cd303dc.html>"),
    "mlogit": (ONLINE, "Croissant, Y. (n.d.). *mlogit: Multinomial logit models* (R package) [Computer software]. <https://CRAN.R-project.org/package=mlogit>"),
    "petersen": (ONLINE, "Petersen, K. B., & Pedersen, M. S. (2012). *The matrix cookbook* (Version November 15, 2012). Technical University of Denmark. <https://www2.imm.dtu.dk/pubdb/pubs/3274-full.html>"),
    "r_core": (ONLINE, "R Core Team. (2025). *R: A language and environment for statistical computing* [Computer software]. R Foundation for Statistical Computing. <https://www.R-project.org/>"),
    "rigollet": (ONLINE, "Rigollet, P. (2016). *18.650 Statistics for applications* [Lecture notes]. MIT OpenCourseWare. <https://ocw.mit.edu/courses/18-650-statistics-for-applications-fall-2016/>"),
    "shedden": (ONLINE, "Shedden, K. (2021). *Statistics 600: Regression analysis* [Lecture notes]. University of Michigan. <https://dept.stat.lsa.umich.edu/~kshedden/stats600/>"),
    "sklearn": (ONLINE, "scikit-learn developers. (n.d.). *scikit-learn: Machine learning in Python* [Software documentation]. <https://scikit-learn.org/>"),
    "soch": (ONLINE, "Soch, J., & The Book of Statistical Proofs contributors. (n.d.). *The book of statistical proofs*. <https://statproofbook.github.io/>"),
    "stata": (ONLINE, "StataCorp. (n.d.). *Stata documentation* [Software documentation]. <https://www.stata.com/features/documentation/>"),
    "statology": (ONLINE, "Statology. (n.d.). *A complete guide to stepwise regression in R*. <https://www.statology.org/stepwise-regression-r/>"),
    "statsmodels": (ONLINE, "statsmodels developers. (n.d.). *statsmodels: Statistical models, hypothesis tests, and data exploration* [Software documentation]. <https://www.statsmodels.org/>"),
    "ucla": (ONLINE, "UCLA Statistical Methods and Data Analytics. (n.d.). *Ordinal logistic regression: R data analysis examples*. <https://stats.oarc.ucla.edu/r/dae/ordinal-logistic-regression/>"),
    "wiki_aic": (ONLINE, "Wikipedia contributors. (n.d.). Akaike information criterion. In *Wikipedia, the free encyclopedia*. <https://en.wikipedia.org/wiki/Akaike_information_criterion>"),
    "wiki_bic": (ONLINE, "Wikipedia contributors. (n.d.). Bayesian information criterion. In *Wikipedia, the free encyclopedia*. <https://en.wikipedia.org/wiki/Bayesian_information_criterion>"),
    "wiki_cp": (ONLINE, "Wikipedia contributors. (n.d.). Mallows's Cp. In *Wikipedia, the free encyclopedia*. <https://en.wikipedia.org/wiki/Mallows%27s_Cp>"),
    "wiki_glm": (ONLINE, "Wikipedia contributors. (n.d.). Generalized linear model. In *Wikipedia, the free encyclopedia*. <https://en.wikipedia.org/wiki/Generalized_linear_model>"),
    "wiki_kendall": (ONLINE, "Wikipedia contributors. (n.d.). Kendall rank correlation coefficient. In *Wikipedia, the free encyclopedia*. <https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient>"),
    "wiki_ncchi2": (ONLINE, "Wikipedia contributors. (n.d.). Noncentral chi-squared distribution. In *Wikipedia, the free encyclopedia*. <https://en.wikipedia.org/wiki/Noncentral_chi-squared_distribution>"),
    "wiki_ncf": (ONLINE, "Wikipedia contributors. (n.d.). Noncentral F-distribution. In *Wikipedia, the free encyclopedia*. <https://en.wikipedia.org/wiki/Noncentral_F-distribution>"),
    "wiki_ordinal": (ONLINE, "Wikipedia contributors. (n.d.). Ordinal regression. In *Wikipedia, the free encyclopedia*. <https://en.wikipedia.org/wiki/Ordinal_regression>"),
    "zaiontz": (ONLINE, "Zaiontz, C. (n.d.). *Multiple regression without intercept*. Real Statistics Using Excel. <https://real-statistics.com/multiple-regression/multiple-regression-without-intercept>"),
}

CHAPTERS = {
    "01-probability-and-random-vectors": ["casella", "montgomery_runger", "seber_lee"],
    "02-sampling-distributions-and-limit-theorems": ["casella", "seber_lee", "wiki_ncchi2", "wiki_ncf"],
    "03-correlation-analysis": ["heumann", "hollander", "kanji", "wiki_kendall"],
    "04-goodness-of-fit-tests": ["heumann", "hollander", "kanji"],
    "05-linear-algebra": ["harville", "seber_lee"],
    "06-matrix-calculus": ["boyd", "petersen"],
    "07-linear-regression-model": ["heumann", "montgomery_runger", "he"],
    "08-ols-and-ml-estimation": ["montgomery_runger", "he"],
    "09-ols-properties-and-geometry": ["seber_lee", "he", "shedden"],
    "10-anova-and-inference": ["james", "montgomery_runger", "he", "zaiontz"],
    "11-one-way-anova": ["montgomery_doe", "montgomery_runger", "wiki_ncchi2", "wiki_ncf"],
    "12-random-effects-and-blocking": ["montgomery_doe", "montgomery_runger"],
    "13-prediction-and-model-assessment": ["montgomery_runger", "cook", "he"],
    "14-standardization-and-multicollinearity": ["james", "montgomery_runger"],
    "15-variable-selection": ["hastie_esl", "james", "montgomery_runger", "hurvich", "soch", "wiki_aic", "wiki_bic", "wiki_cp"],
    "16-ridge-and-lasso": ["hastie_esl", "james", "friedman2010", "hoerl", "tibshirani", "zou_hastie", "zou_df"],
    "17-generalized-linear-models": ["mccullagh", "rigollet", "wiki_glm"],
    "18-logistic-regression": ["agresti", "hosmer", "mccullagh", "train"],
    "19-poisson-and-count-regression": ["hilbe", "roback", "stata"],
    "20-ordinal-regression": ["agresti", "frank_hall", "ucla", "wiki_ordinal"],
    "21-multinomial-choice-models": ["ben_akiva", "train", "apollo", "biogeme", "mlogit"],
    "22-quantile-regression": ["lianxh"],
    "23-correlation-and-goodness-of-fit-code": ["heumann", "hollander", "scipy", "r_core"],
    "24-linear-regression-python": ["statsmodels", "sklearn"],
    "25-linear-regression-r": ["fox_weisberg", "james", "islr_pkg", "leaps", "r_core", "statology"],
    "26-penalized-regression-python": ["friedman2010", "sklearn"],
    "27-glm-python": ["statsmodels", "sklearn"],
    "28-quantile-regression-python": ["sklearn", "statsmodels"],
}

LANG = {
    "en": {
        "heading": "## References",
        "title": "# References",
        "intro": (
            "All works cited in the book, in APA Style (7th edition), grouped by\n"
            "type. Each chapter lists the works it uses at its end. Books are cited\n"
            "as whole works and software by its home page."
        ),
        "cats": {BOOK: "## Books", ARTICLE: "## Research articles", ONLINE: "## Online documents and software"},
    },
    "zh": {
        "heading": "## 参考文献",
        "title": "# 参考文献",
        "intro": (
            "本书引用的全部文献，按 APA 格式（第 7 版）分类列出。各章末尾列出该章"
            "所用的文献。书籍按整本引用，软件按其主页引用。"
        ),
        "cats": {BOOK: "## 书籍", ARTICLE: "## 研究论文", ONLINE: "## 在线文档与软件"},
    },
}


def sort_key(key):
    return REFS[key][1].lower()


def chapter_list(keys):
    by_cat = {BOOK: [], ARTICLE: [], ONLINE: []}
    for k in keys:
        by_cat[REFS[k][0]].append(k)
    ordered = [k for cat in (BOOK, ARTICLE, ONLINE) for k in sorted(by_cat[cat], key=sort_key)]
    return "\n".join(f"- {REFS[k][1]}" for k in ordered) + "\n"


def rewrite_chapter(path, heading, keys):
    text = path.read_text(encoding="utf-8")
    start = text.index("\n" + heading + "\n")
    rest = text[start + 1 + len(heading) + 1:]
    nxt = rest.find("\n## ")
    tail = rest[nxt:] if nxt >= 0 else ""
    new = text[: start + 1] + heading + "\n\n" + chapter_list(keys) + tail.rstrip("\n")
    path.write_text(new.rstrip("\n") + "\n", encoding="utf-8")


def bibliography(lang):
    cfg = LANG[lang]
    used = sorted({k for keys in CHAPTERS.values() for k in keys}, key=sort_key)
    lines = [cfg["title"], "", cfg["intro"], ""]
    for cat in (BOOK, ARTICLE, ONLINE):
        lines += [cfg["cats"][cat], ""]
        lines += [f"- {REFS[k][1]}" for k in used if REFS[k][0] == cat]
        lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n"


def main(langs=("en", "zh")):
    for lang in langs:
        cfg = LANG[lang]
        for chapter, keys in CHAPTERS.items():
            rewrite_chapter(ROOT / "docs" / lang / f"{chapter}.md", cfg["heading"], keys)
        (ROOT / "docs" / lang / "references.md").write_text(bibliography(lang), encoding="utf-8")
    unused = set(REFS) - {k for keys in CHAPTERS.values() for k in keys}
    if unused:
        print("unused references:", sorted(unused))


if __name__ == "__main__":
    import sys

    main(sys.argv[1:] or ("en", "zh"))
