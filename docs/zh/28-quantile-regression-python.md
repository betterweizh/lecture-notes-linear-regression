# 用 Python 实现分位数回归

## 概述

本章用 `scikit-learn` 和 `statsmodels` 拟合 {doc}`22-quantile-regression` 中的线性分位数回归模型，比较二者的估计值，并同时拟合多个分位数，以描述整个条件分布如何依赖于预测变量。 本章应用 {doc}`22-quantile-regression` 中的方法。代码已在 `statsmodels` 0.14 和 `scikit-learn` 1.9 下测试。

:::{admonition} 学习目标
:class: objectives
学完本章后，读者应能够：

- 用 `QuantileRegressor` 和 `QuantReg` 拟合线性分位数回归；
- 关闭 `QuantileRegressor` 默认的惩罚项；
- 拟合并比较多个分位数。
:::

## 拟合单个分位数

有两个库提供线性分位数回归：

- `scikit-learn`：
  [`sklearn.linear_model.QuantileRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.QuantileRegressor.html)；
- `statsmodels`：
  [`statsmodels.regression.quantile_regression.QuantReg`](https://www.statsmodels.org/stable/generated/statsmodels.regression.quantile_regression.QuantReg.html)。

下面的例子在同一组模拟数据上用两个库拟合 $0.1$ 分位数，以便比较估计值。

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import QuantileRegressor

# Simulated data
rng = np.random.default_rng(0)
x = np.linspace(0, 10, 100) + (1 - rng.random(100)) * 5
X = x.reshape(-1, 1)                      # predictors as a 2-D array
y = 2 * x + (1 - rng.random(100)) * 5     # response as a 1-D array

# scikit-learn: alpha=0 switches off the default L1 penalty
model = QuantileRegressor(quantile=0.1, alpha=0.0, fit_intercept=True)
model.fit(X, y)
print(model.intercept_, model.coef_)

# statsmodels: add the intercept column explicitly
res = sm.QuantReg(endog=y, exog=sm.add_constant(X)).fit(q=0.1)
print(res.params)
```

`QuantileRegressor` 会加入权重为 `alpha`（默认 $1.0$）的 $L_1$ 惩罚；设置 `alpha=0.0` 即得到无惩罚估计 {eq}`eq-quantile-regression`。`QuantReg` 在 `res.summary()` 中报告标准误和置信区间。

两个库给出的估计值在求解器容差范围内相同（对这一模拟样本，截距约为 $0.3205$，斜率约为 $2.0118$）。`QuantileRegressor` 默认使用 HiGHS 线性规划求解器（`solver="highs"`）。

## 拟合多个分位数

```python
taus = [0.1, 0.25, 0.5, 0.75, 0.9]
fits = {tau: sm.QuantReg(y, sm.add_constant(X)).fit(q=tau).params for tau in taus}
print(pd.DataFrame(fits, index=["intercept", "slope"]).T)
```

由于本例中的噪声不依赖于 $x$，各分位数的拟合斜率相近，拟合直线大致平行；如果斜率随分位数而变化，则说明预测变量不仅影响条件分布的位置，还影响其形状。

## 本章回顾
### 小结

- `QuantileRegressor(quantile=tau, alpha=0.0)` 与 `sm.QuantReg(y, sm.add_constant(X)).fit(q=tau)` 拟合的是同一个模型。
- `QuantReg` 报告标准误和置信区间。
- 拟合多个 $\tau$ 值可以显示条件分布如何随预测变量变化。

### 常见错误

- 本想使用无惩罚估计，却保留了 `QuantileRegressor` 的默认惩罚 `alpha=1.0`。
- 向 `scikit-learn` 传入形状为 `(n, 1)` 的列向量 `y`；应使用一维数组。
- 使用 `QuantReg` 时忘记添加截距列。

## 参考文献

- scikit-learn developers. (n.d.). *scikit-learn: Machine learning in Python* [Software documentation]. <https://scikit-learn.org/>
- statsmodels developers. (n.d.). *statsmodels: Statistical models, hypothesis tests, and data exploration* [Software documentation]. <https://www.statsmodels.org/>
