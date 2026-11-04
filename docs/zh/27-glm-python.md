# 用 Python 实现广义线性模型

## 概述

本章用 Python 拟合第四部分的广义线性模型：二元响应的 logistic 回归（{doc}`18-logistic-regression`）和计数响应的泊松回归（{doc}`19-poisson-and-count-regression`）。所用的库包括 `statsmodels` 和 `scikit-learn`：前者以统计软件的风格报告标准误、检验和残差，后者侧重于预测。本章还说明两个库在默认设置上的差异。 本章应用 {doc}`17-generalized-linear-models`、{doc}`18-logistic-regression` 和 {doc}`19-poisson-and-count-regression` 中的模型。代码已在 `statsmodels` 0.14 和 `scikit-learn` 1.9 下测试。

:::{admonition} 学习目标
:class: objectives
学完本章后，读者应能够：

- 用 `statsmodels`（`Logit`、`GLM`）和 `scikit-learn`（`LogisticRegression`）拟合 logistic 回归，并报告优势比；
- 实现 logistic 回归的 IRLS 算法；
- 拟合含偏移项和暴露量的泊松回归，并把拟合模型的属性与理论章节中的公式对应起来；
- 计算 Pearson 离散统计量以检查过度离散。
:::

## logistic 回归

### 示例数据

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm

rng = np.random.default_rng(7)
n = 500
data_X = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.binomial(1, 0.4, size=n)})
eta = -0.5 + 1.2 * data_X["x1"] - 0.8 * data_X["x2"]
data_y = rng.binomial(1, 1 / (1 + np.exp(-eta)))
```

### 用 `statsmodels` 拟合 logistic 回归

```python
from statsmodels.discrete.discrete_model import Logit

exog = sm.add_constant(data_X)
logit_res = Logit(data_y, exog).fit()
print(logit_res.summary())

odds_ratios = pd.DataFrame({
    "odds_ratio": np.exp(logit_res.params),
    "ci_lower": np.exp(logit_res.conf_int()[0]),
    "ci_upper": np.exp(logit_res.conf_int()[1]),
})
print(odds_ratios)
```

| 属性 | 对应的量 |
|---|---|
| `params`, `bse` | $\widehat{\bbeta}$ 以及由 $(\bX\T\widehat{\mathbf{W}}\bX)^{-1}$ 得到的 Wald 标准误 |
| `tvalues`, `pvalues` | Wald 统计量 $z = \widehat{\beta}_j/\mathrm{SE}(\widehat{\beta}_j)$ 及其 $p$ 值（以正态分布为参照） |
| `llf`, `llnull` | $\ell(\widehat{\bbeta})$ 和只含截距模型的对数似然 |
| `llr`, `llr_pvalue` | 相对于只含截距模型的似然比统计量及其 $\chi^2$ $p$ 值 |
| `prsquared` | McFadden 伪 $R^2 = 1 - \ell(\widehat{\bbeta})/\ell(\widehat{\beta}_0)$ |
| `predict()` | 拟合概率 $\widehat{\pi}_i$ |
| `get_margeff()` | 平均边际效应 $\beta_j\pi(1-\pi)$ |

同一模型也可以作为 GLM 拟合，此时还会报告偏差；对未分组的二元数据，它等于 $-2\ell(\widehat{\bbeta})$：

```python
glm_logit = sm.GLM(data_y, exog, family=sm.families.Binomial()).fit()
print(glm_logit.deviance, -2 * logit_res.llf)        # equal
```

### 用 `scikit-learn` 拟合 logistic 回归

```python
from sklearn.linear_model import LogisticRegression

sk_logit = LogisticRegression(C=np.inf, max_iter=1000, tol=1e-10)   # no penalty
sk_logit.fit(data_X, data_y)
print(sk_logit.intercept_, sk_logit.coef_)
print(sk_logit.predict_proba(data_X)[:5, 1])          # fitted probabilities
```

`LogisticRegression` 默认施加 $L_2$ 惩罚，由惩罚强度的倒数 `C`（默认 `1.0`）控制。设置 `C=np.inf` 即拟合无惩罚的极大似然模型。在 `scikit-learn` 1.8 之前的版本中，这通过 `penalty=None` 实现；从 1.8 版起该参数已弃用，原来的 `multi_class` 参数也已删除：当响应多于两类时会自动拟合多项模型。

### 从零实现 IRLS

{doc}`18-logistic-regression` 中的 Newton–Raphson（IRLS）迭代只需几行代码，就能在机器精度内复现 `statsmodels` 的结果：

```python
X = exog.to_numpy()
y = data_y
beta = np.zeros(X.shape[1])
for _ in range(25):
    pi = 1 / (1 + np.exp(-X @ beta))
    w = pi * (1 - pi)                                 # diagonal of W
    z = X @ beta + (y - pi) / w                       # working response
    beta = np.linalg.solve(X.T @ (w[:, None] * X), X.T @ (w * z))

se = np.sqrt(np.diag(np.linalg.inv(X.T @ (w[:, None] * X))))
print(np.max(np.abs(beta - logit_res.params.to_numpy())))   # about 1e-16
print(np.max(np.abs(se - logit_res.bse.to_numpy())))
```

## 泊松回归

### 示例数据

一个含两个预测变量的模拟计数数据集：

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 500
data_X = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.uniform(0, 2, size=n)})
true_beta = np.array([0.5, 0.3, -0.4])           # intercept, x1, x2
lam = np.exp(true_beta[0] + data_X.to_numpy() @ true_beta[1:])
data_Y = rng.poisson(lam)
```

### 用 `statsmodels` 拟合泊松回归

#### 拟合模型

```python
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Poisson

exog = sm.add_constant(data_X)        # add the intercept column
model = Poisson(
    endog=data_Y, exog=exog,
    offset=None, exposure=None,
    missing="none", check_rank=True,
)
res = model.fit()
print(res.summary())

# the same model as a GLM with the Poisson family and log link
glm_pois = sm.GLM(data_Y, exog, family=sm.families.Poisson()).fit()
print(glm_pois.deviance, glm_pois.pearson_chi2)
```

主要参数如下：

`endog`, `exog`
: 响应向量 $\by$（因变量，或称内生变量）与预测变量矩阵 $\bX$（自变量，或称外生变量）。默认**不**包含截距，需要用 `statsmodels.tools.add_constant`（即 `sm.add_constant`）添加。

`offset`
: 偏移项 $o_i$，加入线性预测子，其系数固定为 1。

`exposure`
: 暴露量 $E_i$；$\ln(E_i)$ 加入线性预测子，其系数固定为 1。因此传入 `exposure=E` 等价于传入 `offset=np.log(E)`。

#### 读取结果

对拟合结果 `res`，下列属性与 {doc}`19-poisson-and-count-regression` 中的量相对应：

| 属性 | 对应的量 | 等价表达式 |
|---|---|---|
| `res.params` | $\widehat{\bbeta}$ | |
| `res.predict()` | 拟合均值 $\widehat{\lambda}_i$ | `res.predict(model.exog)`（无偏移项或暴露量时） |
| `res.fittedvalues` | 线性预测子 $\bx_i\T\widehat{\bbeta}$，不含偏移项 | `np.log(res.predict())`（无偏移项或暴露量时） |
| `res.resid` | 原始残差 $y_i - \widehat{\lambda}_i$ | `model.endog - res.predict()` |
| `res.resid_pearson` | Pearson 残差 $(y_i - \widehat{\lambda}_i)/\sqrt{\widehat{\lambda}_i}$ | |
| `res.df_resid` | 残差自由度 $n-p$ | |

注意，对 `Poisson` 这类离散模型，`fittedvalues` 是**线性预测子**而不是拟合均值；拟合计数应使用 `predict()`。

#### Pearson 离散度

{doc}`19-poisson-and-count-regression` 中拟泊松一节的离散参数估计 $\widehat{\phi} = \sum_i r_{P,i}^2/(n-p)$ 可按如下方式计算：

```python
def compute_pearson_dispersion(res):
    """Pearson dispersion: sum of squared Pearson residuals / residual df."""
    return np.sum(np.square(res.resid_pearson)) / res.df_resid

print(compute_pearson_dispersion(res))   # close to 1 if there is no overdispersion
```

### 用 `scikit-learn` 拟合泊松回归

```python
from sklearn.linear_model import PoissonRegressor

sk_pois = PoissonRegressor(alpha=0, fit_intercept=True, max_iter=1000)
sk_pois.fit(data_X, data_Y)

coef = pd.DataFrame(
    np.hstack([sk_pois.intercept_, sk_pois.coef_]),
    index=["const"] + sk_pois.feature_names_in_.tolist(),
    columns=["coef"],
)
print(coef)
```

`alpha=0` 关闭默认的 $L_2$ 惩罚，因此所得估计是普通的极大似然估计，可与 `statsmodels` 的结果比较。只有当 `data_X` 是带列名的 `pandas` DataFrame 时，才存在属性 `feature_names_in_`。`scikit-learn` 不报告标准误或检验统计量。

### 两个库的结果为何可能不同

`statsmodels` 中的 `Poisson` 和 `scikit-learn` 中的 `PoissonRegressor` 默认使用不同的优化算法和收敛准则，因此估计值可能在最后几位有效数字上不同。增加 `PoissonRegressor` 的迭代次数（`max_iter`）或收紧其容差（`tol`），通常能使两组估计一致。

## 本章回顾
### 小结

- `statsmodels` 中的 `Logit` 或 `GLM(..., family=Binomial())` 给出估计值、Wald 检验和似然比检验以及伪 $R^2$；对系数及其置信限取指数即得优势比。
- 在 `scikit-learn` 中，`LogisticRegression(C=np.inf)` 和 `PoissonRegressor(alpha=0)` 给出无惩罚的极大似然估计。
- `statsmodels.discrete.discrete_model.Poisson` 支持偏移项和暴露量；`predict()` 返回拟合均值，`fittedvalues` 返回线性预测子，`resid_pearson` 返回 Pearson 残差。
- 简短的 IRLS 循环即可复现库的估计值。

### 常见错误

- 忘记 `sm.add_constant`，结果拟合了不含截距的模型。
- 保留 `LogisticRegression`（`C=1.0`）或 `PoissonRegressor`（`alpha=1.0`）的默认惩罚，然后把系数与无惩罚估计比较。
- 把 logistic 回归中的 $e^{\beta_j}$ 报告为概率之比；它是优势比。
- 把 `res.fittedvalues` 当作拟合计数；对离散模型，它是线性预测子。
- 把对数暴露量作为 `exposure` 传入（会被取两次对数）；应传入 `exposure=E` 或 `offset=np.log(E)` 之一。

## 参考文献

- scikit-learn developers. (n.d.). *scikit-learn: Machine learning in Python* [Software documentation]. <https://scikit-learn.org/>
- statsmodels developers. (n.d.). *statsmodels: Statistical models, hypothesis tests, and data exploration* [Software documentation]. <https://www.statsmodels.org/>
