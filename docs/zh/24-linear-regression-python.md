# 用 Python 实现线性回归

## 概述

本章用 Python 实现第二、三部分的线性回归方法。内容包括 `statsmodels` 的使用流程、拟合模型的每个属性所对应的公式、一个能复现 `statsmodels` 结果的普通最小二乘从零实现，以及影响诊断和方差膨胀因子的代码。 本章应用 {doc}`08-ols-and-ml-estimation`、{doc}`10-anova-and-inference`、{doc}`13-prediction-and-model-assessment` 和 {doc}`14-standardization-and-multicollinearity` 中的方法。代码已在 Python 3.13、`numpy` 2.5、`pandas` 3.0、`scipy` 1.18、`statsmodels` 0.14 和 `scikit-learn` 1.9 下测试。

:::{admonition} 学习目标
:class: objectives
学完本章后，读者应能够：

- 用 `statsmodels` 拟合线性回归，并读取估计值、检验、方差分析量和信息准则；
- 计算新观测的置信区间和预测区间；
- 计算杠杆值、学生化残差和 Cook 距离；
- 正确计算方差膨胀因子（包括常数列）；
- 直接根据矩阵公式实现 OLS 及其推断。
:::

## 示例数据

示例使用 `scikit-learn` 自带的糖尿病数据集：$n = 442$ 名患者的十项基线指标（年龄、性别、体重指数、血压和六项血清指标），响应为一年后疾病进展的定量指标。

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.datasets import load_diabetes

data_X, data_y = load_diabetes(return_X_y=True, as_frame=True, scaled=False)
X_train, y_train = data_X.iloc[:300], data_y.iloc[:300]
X_new = data_X.iloc[300:310]          # observations to predict
```

## `statsmodels` 使用流程

涉及三个对象：

| 对象 | 作用 |
|---|---|
| [`statsmodels.regression.linear_model.OLS`](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLS.html) | 模型：响应与设计矩阵 |
| [`RegressionResults`](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.RegressionResults.html) | 拟合结果，由 `model.fit()` 返回 |
| [`PredictionResults`](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.PredictionResults.html) | 预测结果，由 `res.get_prediction()` 返回 |

```python
X = sm.add_constant(X_train)          # add the column of ones (intercept)
model = sm.OLS(y_train, X)
res = model.fit()
print(res.summary())

pred = res.get_prediction(sm.add_constant(X_new, has_constant="add"))
print(pred.summary_frame(alpha=0.05))
```

`OLS` **不会**自动添加截距；`sm.add_constant` 会添加一列名为 `const` 的全 1 列。对单个新观测要传入 `has_constant="add"`，否则只有一行的矩阵会被误认为已经包含常数列。

### 模型信息与拟合结果

| 属性 | 含义 |
|---|---|
| `model.endog`, `model.exog` | 响应 $\by$（一维数组）与设计矩阵 $\bX$（二维数组） |
| `model.endog_names`, `model.exog_names` | 变量名 |
| `res.nobs` | 观测数 $n$ |
| `res.summary(alpha=0.05)`, `res.summary2()` | 汇总表 |
| `res.fittedvalues` | 拟合值 $\widehat{\by}$ |
| `res.resid` | 残差 $\be = \by - \widehat{\by}$ |
| `res.resid_pearson` | 除以 $\widehat{\sigma}$ 的残差，即标准化残差 $d_i$ |

### 系数与 $t$ 检验

| 属性 | 公式 |
|---|---|
| `res.params` | $\widehat{\bbeta} = (\bX\T\bX)^{-1}\bX\T\by$ |
| `res.scale` | $\widehat{\sigma}^2 = \mathit{SS}_E/(n-p)$ |
| `res.bse` | $\mathrm{SE}(\widehat{\beta}_j) = \sqrt{\widehat{\sigma}^2c_{jj}}$，$c_{jj}$ 为 $(\bX\T\bX)^{-1}$ 的第 $j$ 个对角元 |
| `res.tvalues`, `res.pvalues` | $H_0:\beta_j = 0$ 下 $T_j = \widehat{\beta}_j/\mathrm{SE}(\widehat{\beta}_j) \sim t(n-p)$，以及双侧 $p$ 值 |
| `res.conf_int(alpha=0.05)` | $\widehat{\beta}_j \pm t_{1-\alpha/2,\,n-p}\,\mathrm{SE}(\widehat{\beta}_j)$ |
| `res.use_t` | 推断时是否使用 $t$ 分布（而非正态分布） |

含截距时 $p = k+1$，不含截距时 $p = k$。

### 方差分析量

| 属性 | 含截距 | 不含截距 |
|---|---|---|
| `res.df_model` | $k$ | $k$ |
| `res.df_resid` | $n-k-1$ | $n-k$ |
| `res.ess`（$\mathit{SS}_R$） | $\sum_i(\widehat{y}_i-\bar{y})^2$ | $\sum_i\widehat{y}_i^2$ |
| `res.ssr`（$\mathit{SS}_E$） | $\sum_i(y_i-\widehat{y}_i)^2$ | $\sum_i(y_i-\widehat{y}_i)^2$ |
| `res.centered_tss`, `res.uncentered_tss` | $\sum_i(y_i-\bar{y})^2$ | $\sum_i y_i^2$ |
| `res.mse_model`（$\mathit{MS}_R$） | $\mathit{SS}_R/k$ | $\mathit{SS}_R/k$ |
| `res.mse_resid`（$\mathit{MS}_E$） | $\mathit{SS}_E/(n-k-1)$ | $\mathit{SS}_E/(n-k)$ |
| `res.mse_total` | $\mathit{SS}_T/(n-1)$ | $\mathit{SS}_T/n$ |
| `res.fvalue`, `res.f_pvalue` | $F = \mathit{MS}_R/\mathit{MS}_E \sim F(k, n-k-1)$ | $F \sim F(k, n-k)$ |

注意 `res.ssr` 是**残差**平方和 $\mathit{SS}_E$，而不是回归平方和；`statsmodels` 把回归平方和称为 `ess`（"explained sum of squares"，解释平方和）。

### 信息准则与拟合优度

| 属性 | 公式 |
|---|---|
| `res.llf` | 最大对数似然 $\ln\widehat{\mathcal{L}} = -\frac{n}{2}\ln\bigl(2\pi\,\mathit{SS}_E/n\bigr) - \frac{n}{2}$ |
| `res.aic` | $2p - 2\ln\widehat{\mathcal{L}}$ |
| `res.bic` | $p\ln(n) - 2\ln\widehat{\mathcal{L}}$ |
| `res.rsquared` | $R^2 = 1 - \mathit{SS}_E/\mathit{SS}_T$ |
| `res.rsquared_adj` | 含截距时为 $1 - \dfrac{\mathit{SS}_E/(n-p)}{\mathit{SS}_T/(n-1)}$；不含截距时为 $1 - \dfrac{\mathit{SS}_E/(n-p)}{\mathit{SS}_T/n}$ |

`statsmodels` 在 AIC 和 BIC 中只计入 $p$ 个回归系数而不计入 $\sigma^2$，并保留了对数似然中的常数项；如 {doc}`15-variable-selection` 所述，这两点都不影响同一数据上模型之间的比较。它不提供修正 AIC，可以用 `res.aic + (2*p**2 + 2*p) / (res.nobs - p - 1)`（其中 `p = len(res.params)`）计算。

### 预测

`res.get_prediction(X0)` 返回 `PredictionResults` 对象。记 $\bx_0$ 为 `X0` 的一行：

| 属性或方法 | 公式 |
|---|---|
| `pred.predicted_mean` | $\widehat{y}_0 = \bx_0\T\widehat{\bbeta}$ |
| `pred.se_mean` | $\widehat{\sigma}\sqrt{\bx_0\T(\bX\T\bX)^{-1}\bx_0}$，用于置信区间 |
| `pred.se_obs` | $\widehat{\sigma}\sqrt{1 + \bx_0\T(\bX\T\bX)^{-1}\bx_0}$，用于预测区间 |
| `pred.conf_int(alpha=0.05)` | 平均响应的置信区间 {eq}`eq-ci-mean` |
| `pred.conf_int(obs=True, alpha=0.05)` | 预测区间 {eq}`eq-pi` |
| `pred.summary_frame(alpha=0.05)` | 以上全部：`mean`、`mean_se`、`mean_ci_*`、`obs_ci_*` |

`res.predict(X0)` 只返回点预测。

## 影响诊断

{doc}`13-prediction-and-model-assessment` 中的杠杆值、学生化残差和 Cook 距离可由 `res.get_influence()` 得到：

```python
infl = res.get_influence()
diag = pd.DataFrame({
    "leverage": infl.hat_matrix_diag,                 # h_ii
    "student_resid": infl.resid_studentized_internal, # r_i
    "cooks_d": infl.cooks_distance[0],                # D_i
}, index=X_train.index)
print(diag.sort_values("cooks_d", ascending=False).head())
```

## 方差膨胀因子

`statsmodels.stats.outliers_influence.variance_inflation_factor(exog, j)` 把 `exog` 的第 `j` 列对 **`exog` 的其余各列**回归，返回 $1/(1-R_j^2)$。它不会自动添加截距，因此传入的矩阵必须包含常数列；否则 $R_j^2$ 是在不含截距的情况下计算的，VIF 会严重偏大。对糖尿病数据，若省略常数列，`bmi` 的 VIF 约为 55，而正确值约为 1.5。

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

def calculate_vif(data):
    """VIF of each column of a DataFrame of predictors (no constant column)."""
    exog = sm.add_constant(data)
    return pd.Series(
        {col: variance_inflation_factor(exog.values, i)
         for i, col in enumerate(exog.columns) if col != "const"})

def stepwise_remove_variable_vif(data, threshold=10.0, verbose=True):
    """Repeatedly drop the predictor with the largest VIF until all VIFs < threshold."""
    removed, history, step = [], {}, 0
    while data.shape[1] > 1:
        vif = calculate_vif(data)
        history[f"step{step}"] = vif
        if (vif < threshold).all():
            break
        worst = vif.idxmax()
        if verbose:
            print(f"Step {step}: remove {worst} (VIF = {vif.max():.3f})")
        removed.append(worst)
        data = data.drop(columns=worst)
        step += 1
    return data, removed, pd.concat(history, axis=1)

print(calculate_vif(data_X).round(2))
kept, removed, history = stepwise_remove_variable_vif(data_X, threshold=10)
```

对糖尿病数据，血清指标 `s1` 和 `s2` 的 VIF 分别约为 59 和 39；逐步程序首先删除 `s1`。如 {doc}`14-standardization-and-multicollinearity` 所述，删除预测变量只是几种处理方法之一，应结合专业知识来选择。

## 从零实现 OLS

下面的类直接根据 {doc}`08-ols-and-ml-estimation`、{doc}`10-anova-and-inference` 和 {doc}`13-prediction-and-model-assessment` 中的公式计算上表中的每一个量。无论是否含截距，其结果与 `statsmodels` 的相对差异都在约 $10^{-9}$ 以内。

:::{dropdown} `LinearRegression` 类
```python
import numpy as np
import pandas as pd
from scipy import stats

class LinearRegression:
    """Ordinary least squares from the matrix formulas of Part II.

    Parameters
    ----------
    data_X : pandas.DataFrame or numpy.ndarray, shape (n, k)
        Predictors, without a column of ones.
    data_y : pandas.Series or numpy.ndarray, shape (n,)
        Response.
    intercept : bool
        Whether to add an intercept (a column of ones).
    """

    def __init__(self, data_X, data_y, intercept=True):
        X = np.asarray(data_X, dtype=float)
        y = np.asarray(data_y, dtype=float).ravel()
        if X.ndim != 2 or X.shape[0] != y.shape[0]:
            raise ValueError(f"incompatible shapes: X {X.shape}, y {y.shape}")

        self.intercept = intercept
        self.nobs, self.n_indep_var = X.shape            # n and k
        self.exog_names = (list(data_X.columns) if isinstance(data_X, pd.DataFrame)
                           else [f"x{j + 1}" for j in range(self.n_indep_var)])
        self.endog_names = getattr(data_y, "name", None) or "y"

        if intercept:
            X = np.column_stack([np.ones(self.nobs), X])
            self.exog_names = ["const"] + self.exog_names
        self.exog, self.endog = X, y
        self.num_params = X.shape[1]                      # p

        self.df_model = self.n_indep_var                  # k
        self.df_resid = self.nobs - self.num_params       # n - p
        self.df_total = self.df_model + self.df_resid     # n - 1 (or n)

    # ------------------------------------------------------------------
    def fit(self):
        X, y = self.exog, self.endog
        self.XTX_inv = np.linalg.inv(X.T @ X)             # C = (X'X)^{-1}
        self.params = self.XTX_inv @ X.T @ y              # beta_hat
        self.hat_diag = np.einsum("ij,jk,ik->i", X, self.XTX_inv, X)  # h_ii
        self.fittedvalues = X @ self.params
        self.resid = y - self.fittedvalues

        # sums of squares (centered total SS with an intercept, uncentered without)
        self.ssr = float(self.resid @ self.resid)         # SS_E
        if self.intercept:
            self.tss = float(np.sum((y - y.mean()) ** 2))
        else:
            self.tss = float(y @ y)
        self.ess = self.tss - self.ssr                    # SS_R

        self.scale = self.ssr / self.df_resid             # sigma^2 hat = MS_E
        self.resid_pearson = self.resid / np.sqrt(self.scale)
        self.bse = np.sqrt(self.scale * np.diag(self.XTX_inv))
        return self

    # ------------------------------------------------------------------
    def anova(self):
        ms_r = self.ess / self.df_model
        ms_e = self.ssr / self.df_resid
        f = ms_r / ms_e
        table = pd.DataFrame(
            {"df": [self.df_model, self.df_resid, self.df_total],
             "SS": [self.ess, self.ssr, self.tss],
             "MS": [ms_r, ms_e, self.tss / self.df_total],
             "F": [f, np.nan, np.nan],
             "P-value": [stats.f.sf(f, self.df_model, self.df_resid), np.nan, np.nan]},
            index=["Regression", "Residual", "Total"])
        return table

    def coef_table(self, alpha=0.05):
        t = self.params / self.bse
        q = stats.t.ppf(1 - alpha / 2, self.df_resid)
        return pd.DataFrame(
            {"coef": self.params, "std_err": self.bse, "t": t,
             "p_value": 2 * stats.t.sf(np.abs(t), self.df_resid),
             "ci_lower": self.params - q * self.bse,
             "ci_upper": self.params + q * self.bse},
            index=self.exog_names)

    def criteria(self):
        n, p = self.nobs, self.num_params
        llf = -n / 2 * (np.log(2 * np.pi * self.ssr / n) + 1)   # max log-likelihood
        aic = 2 * p - 2 * llf             # statsmodels convention: sigma^2 not counted
        return pd.Series({
            "llf": llf,
            "aic": aic,
            "aicc": aic + (2 * p ** 2 + 2 * p) / (n - p - 1),
            "bic": p * np.log(n) - 2 * llf,
            "rsquared": 1 - self.ssr / self.tss,
            "rsquared_adj": 1 - (self.ssr / self.df_resid) / (self.tss / self.df_total),
        })

    def influence(self):
        h = self.hat_diag
        r = self.resid / np.sqrt(self.scale * (1 - h))   # studentized residuals
        cooks = r ** 2 / self.num_params * h / (1 - h)
        return pd.DataFrame({"leverage": h, "student_resid": r, "cooks_d": cooks})

    def predict(self, data_X_new, alpha=0.05):
        X0 = np.asarray(data_X_new, dtype=float)
        if self.intercept:
            X0 = np.column_stack([np.ones(X0.shape[0]), X0])
        mean = X0 @ self.params
        q = np.einsum("ij,jk,ik->i", X0, self.XTX_inv, X0)   # x0' C x0
        se_mean = np.sqrt(self.scale * q)
        se_obs = np.sqrt(self.scale * (1 + q))
        t = stats.t.ppf(1 - alpha / 2, self.df_resid)
        return pd.DataFrame({
            "mean": mean, "mean_se": se_mean, "obs_se": se_obs,
            "mean_ci_lower": mean - t * se_mean, "mean_ci_upper": mean + t * se_mean,
            "obs_ci_lower": mean - t * se_obs, "obs_ci_upper": mean + t * se_obs})
```
:::

下面的代码拟合两种实现，并报告最大的相对差异：

```python
from scipy import stats   # used inside the class

for intercept in (True, False):
    mine = LinearRegression(X_train, y_train, intercept=intercept).fit()
    exog = sm.add_constant(X_train) if intercept else X_train
    ref = sm.OLS(y_train, exog).fit()
    diffs = {
        "params": np.max(np.abs(mine.params - ref.params) / np.abs(ref.params)),
        "bse": np.max(np.abs(mine.bse - ref.bse) / ref.bse),
        "F": abs(mine.anova().loc["Regression", "F"] - ref.fvalue) / ref.fvalue,
        "aic": abs(mine.criteria()["aic"] - ref.aic) / abs(ref.aic),
    }
    print(intercept, {k: f"{v:.1e}" for k, v in diffs.items()})
```

两点实现说明：

- 为了与公式对应，这里使用 `np.linalg.inv(X.T @ X)`。对条件数很大的设计矩阵，QR 分解（`np.linalg.lstsq`）在数值上更稳定；`statsmodels` 使用伪逆。
- 不含截距时，总平方和是未中心化的（$\sum_i y_i^2$），与 {doc}`10-anova-and-inference` 一致。

## 本章回顾
### 小结

- `sm.OLS(y, sm.add_constant(X)).fit()` 给出估计值、$t$ 检验和 $F$ 检验、方差分析平方和以及 AIC/BIC；`get_prediction` 给出置信区间和预测区间。
- `get_influence()` 给出杠杆值、学生化残差和 Cook 距离。
- 传给 `variance_inflation_factor` 的矩阵必须包含常数列。
- 第二部分的矩阵公式可以在数值精度内复现 `statsmodels` 的结果。

### 常见错误

- 忘记 `sm.add_constant`，结果悄悄拟合了过原点的模型。
- 把 `res.ssr` 当作回归平方和；它是 $\mathit{SS}_E$。
- 计算 VIF 时没有常数列。
- 需要预测区间时却使用 `pred.conf_int()`；应传入 `obs=True`。

## 参考文献

- scikit-learn developers. (n.d.). *scikit-learn: Machine learning in Python* [Software documentation]. <https://scikit-learn.org/>
- statsmodels developers. (n.d.). *statsmodels: Statistical models, hypothesis tests, and data exploration* [Software documentation]. <https://www.statsmodels.org/>
