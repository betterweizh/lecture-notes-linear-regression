# 用 Python 实现岭回归与 lasso 回归

## 概述

本章用 `scikit-learn` 拟合 {doc}`16-ridge-and-lasso` 中的惩罚回归。内容包括：该库如何缩放目标函数，从而把它的 `alpha` 参数与理论章节中的惩罚参数 $\lambda$ 对应起来；如何在管道（pipeline）中标准化预测变量；用交叉验证选择惩罚参数；以及验证岭估计的闭式解和 lasso 的坐标下降算法。 本章应用 {doc}`16-ridge-and-lasso` 和 {doc}`24-linear-regression-python` 中的方法。代码已在 `scikit-learn` 1.9 下测试。

:::{admonition} 学习目标
:class: objectives
学完本章后，读者应能够：

- 在标准化预测变量后拟合岭回归、lasso 回归和弹性网回归；
- 在 `scikit-learn` 的 `alpha` 与 {doc}`16-ridge-and-lasso` 中的 $\lambda$ 之间换算；
- 用 $K$ 折交叉验证选择惩罚参数，并计算正则化路径；
- 实现岭回归闭式解和 lasso 坐标下降算法，并与库的结果对照。
:::

## `scikit-learn` 中的目标函数

`scikit-learn` 把惩罚参数称为 `alpha`，并且不同估计器对目标函数的缩放方式不同：

| 估计器 | 最小化的目标函数 | 与 {eq}`eq-penalized-ls` 的关系 |
|---|---|---|
| `Ridge(alpha)` | $\lVert\by - \bX\bbeta\rVert_2^2 + \alpha\lVert\bbeta\rVert_2^2$ | $\lambda = \alpha$ |
| `Lasso(alpha)` | $\dfrac{1}{2n}\lVert\by - \bX\bbeta\rVert_2^2 + \alpha\lVert\bbeta\rVert_1$ | $\lambda = 2n\alpha$ |
| `ElasticNet(alpha, l1_ratio)` | $\dfrac{1}{2n}\lVert\by - \bX\bbeta\rVert_2^2 + \alpha\rho\lVert\bbeta\rVert_1 + \dfrac{\alpha(1-\rho)}{2}\lVert\bbeta\rVert_2^2$，$\rho$ = `l1_ratio` | $\lambda_1 = 2n\alpha\rho$，$\lambda_2 = n\alpha(1-\rho)$ |

三者默认都拟合一个不受惩罚的截距（`fit_intercept=True`），但都不会标准化预测变量，因此必须显式地进行标准化。

## 示例数据

```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge, Lasso, RidgeCV, LassoCV, lasso_path
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_diabetes(return_X_y=True, as_frame=True, scaled=False)
n, k = X.shape
```

## 岭回归

在管道中进行标准化，可以保证预测时对新数据使用相同的中心化和缩放；在交叉验证中，还能保证这些量只由训练折估计。

```python
ridge = make_pipeline(StandardScaler(), Ridge(alpha=25.0))
ridge.fit(X, y)
coef_std = ridge[-1].coef_                       # coefficients on the standardized scale
coef_orig = coef_std / ridge[0].scale_           # back to the original units
print(pd.Series(coef_orig, index=X.columns).round(3))
```

在标准化尺度上，预测变量 $j$ 的系数表示 $x_j$ 变化一个标准差时 $y$ 的变化；除以标准差（`scale_`）即可换算回原始单位，见 {doc}`14-standardization-and-multicollinearity`。注意 `StandardScaler` 计算标准差时分母为 $n$。

### 验证闭式解

岭估计 {eq}`eq-ridge` 可以直接在标准化的预测变量和中心化的响应上计算：

```python
Z = StandardScaler().fit_transform(X)
yc = y - y.mean()
lam = 25.0
beta_closed = np.linalg.solve(Z.T @ Z + lam * np.eye(k), Z.T @ yc)
beta_sklearn = Ridge(alpha=lam).fit(Z, y).coef_
print(np.max(np.abs(beta_closed - beta_sklearn)))     # about 1e-14
```

### 有效自由度

```python
d = np.linalg.svd(Z, compute_uv=False)                 # singular values d_j
df_ridge = np.sum(d**2 / (d**2 + lam))                 # df(lambda)
H = Z @ np.linalg.solve(Z.T @ Z + lam * np.eye(k), Z.T)
print(df_ridge, np.trace(H))                           # equal
```

## lasso 回归

```python
lasso = make_pipeline(StandardScaler(), Lasso(alpha=0.5))
lasso.fit(X, y)
coef = pd.Series(lasso[-1].coef_, index=X.columns)
print(coef.round(3))
print("number of zero coefficients:", int((coef == 0).sum()))
```

当 `alpha = 0.5` 时，十个系数中有两个恰好为零；`alpha` 越大，被置为零的系数越多。

### 从零实现坐标下降

下面的函数以 $\lVert\by-\bX\bbeta\rVert_2^2 + \lambda\lVert\bbeta\rVert_1$ 为目标函数实现 {prf:ref}`alg-lasso-cd`。取 $\lambda = 2n\alpha$ 时，它复现 `Lasso(alpha)` 的结果：

```python
def soft_threshold(z, gamma):
    """S(z, gamma) = sign(z) * max(|z| - gamma, 0)."""
    return np.sign(z) * np.maximum(np.abs(z) - gamma, 0.0)

def lasso_cd(Z, y, lam, tol=1e-12, max_iter=100_000):
    """Lasso by cyclic coordinate descent; Z standardized, y centered."""
    beta = np.zeros(Z.shape[1])
    col_ss = np.sum(Z**2, axis=0)                      # x_j' x_j
    for _ in range(max_iter):
        beta_old = beta.copy()
        for j in range(Z.shape[1]):
            r_j = y - Z @ beta + Z[:, j] * beta[j]     # partial residual
            beta[j] = soft_threshold(Z[:, j] @ r_j, lam / 2) / col_ss[j]
        if np.max(np.abs(beta - beta_old)) < tol:
            break
    return beta

alpha = 0.5
beta_cd = lasso_cd(Z, yc, lam=2 * n * alpha)
beta_skl = Lasso(alpha=alpha, tol=1e-12, max_iter=100_000).fit(Z, y).coef_
print(np.max(np.abs(beta_cd - beta_skl)))             # about 1e-10

lam_max = 2 * np.max(np.abs(Z.T @ yc))                 # all coefficients zero beyond this
print(np.all(lasso_cd(Z, yc, 1.0001 * lam_max) == 0))  # True
```

为了清楚起见，这一实现每次都重新计算完整的残差；实际应用中的代码会增量地更新残差。

## 用交叉验证选择惩罚参数

```python
alphas = np.logspace(-3, 3, 100)

ridge_cv = make_pipeline(StandardScaler(), RidgeCV(alphas=alphas))
ridge_cv.fit(X, y)
print("ridge alpha:", ridge_cv[-1].alpha_)

lasso_cv = make_pipeline(StandardScaler(), LassoCV(cv=10, random_state=0))
lasso_cv.fit(X, y)
print("lasso alpha:", lasso_cv[-1].alpha_)
print(pd.Series(lasso_cv[-1].coef_, index=X.columns).round(3))
```

`RidgeCV` 默认使用 {doc}`16-ridge-and-lasso` 中高效的留一法公式，而 `LassoCV` 在自动选取的网格上进行 $K$ 折交叉验证。`LassoCV` 把交叉验证误差保存在 `mse_path_` 中（每折一列），据此可以应用一倍标准误规则：

```python
cv = lasso_cv[-1]
mean_mse = cv.mse_path_.mean(axis=1)
se_mse = cv.mse_path_.std(axis=1, ddof=1) / np.sqrt(cv.mse_path_.shape[1])
i_min = np.argmin(mean_mse)
alpha_1se = np.max(cv.alphas_[mean_mse <= mean_mse[i_min] + se_mse[i_min]])
print("alpha (min):", cv.alphas_[i_min], " alpha (1-SE):", alpha_1se)
```

## 正则化路径

```python
import matplotlib.pyplot as plt

alphas_path, coefs_path, _ = lasso_path(Z, yc, alphas=np.logspace(-2, 2, 100))
fig, ax = plt.subplots(figsize=(7, 4))
for j, name in enumerate(X.columns):
    ax.plot(np.log10(alphas_path), coefs_path[j], label=name)
ax.set_xlabel(r"$\log_{10}(\alpha)$")
ax.set_ylabel("standardized coefficient")
ax.set_title("Lasso regularization path")
ax.legend(ncol=2, fontsize=8)
plt.show()
```

从右向左看图：随着惩罚减小，系数逐个进入模型；在最左端，它们接近 OLS 估计值。

## 本章回顾
### 小结

- `Ridge(alpha)` 对应 $\lambda = \alpha$；`Lasso(alpha)` 对应 $\lambda = 2n\alpha$。
- 把 `StandardScaler` 与估计器放在同一个管道中。
- `RidgeCV` 和 `LassoCV` 用交叉验证选择惩罚参数；`lasso_path` 计算正则化路径。
- 岭估计的闭式解和简短的坐标下降循环可以在数值精度内复现 `scikit-learn` 的结果。

### 常见错误

- 在未标准化的预测变量上拟合 `Ridge` 或 `Lasso`。
- 在交叉验证之前对整个数据集做标准化，使验证折的信息泄漏到训练中；应在管道中进行标准化。
- 对 `Ridge` 和 `Lasso` 使用相同的 `alpha`，并期望二者的收缩程度相当；两者的目标函数缩放方式不同。
- 未除以 `scale_` 就用原始单位解释标准化系数。

## 参考文献

- Friedman, J., Hastie, T., & Tibshirani, R. (2010). Regularization paths for generalized linear models via coordinate descent. *Journal of Statistical Software, 33*(1), 1–22. <https://doi.org/10.18637/jss.v033.i01>
- scikit-learn developers. (n.d.). *scikit-learn: Machine learning in Python* [Software documentation]. <https://scikit-learn.org/>
