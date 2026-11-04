# Glossary

English and Chinese terms used in this book. The Chinese edition uses the
Chinese term and gives the English term in parentheses at its first use in
each chapter.

## Preliminaries

| English | 中文 | Symbol / note |
|---|---|---|
| random variable / random vector | 随机变量 / 随机向量 | $X$, $\bz$ |
| cumulative distribution function (CDF) | 累积分布函数 | $F$ |
| probability mass / density function | 概率质量函数 / 概率密度函数 | $f$ |
| expectation / variance / covariance | 期望 / 方差 / 协方差 | $\E$, $\Var$, $\Cov$ |
| conditional expectation | 条件期望 | $\E[Y\mid X]$ |
| covariance matrix | 协方差矩阵 | $\Cov(\bz)$ |
| multivariate normal distribution | 多元正态分布 | $\mathcal{N}_m(\boldsymbol\mu,\bSigma)$ |
| chi-square / $t$ / $F$ distribution | 卡方分布 / $t$ 分布 / $F$ 分布 | |
| noncentral distribution / noncentrality parameter | 非中心分布 / 非中心参数 | $\delta$ |
| quadratic form | 二次型 | $\bz\T\mathbf{A}\bz$ |
| Cochran's theorem | Cochran 定理 | |
| Pearson correlation coefficient | Pearson 相关系数 | $r$, $\rho$ |
| Fisher $z$-transformation | Fisher $z$ 变换 | $\operatorname{artanh}(r)$ |
| Spearman / Kendall rank correlation | Spearman / Kendall 秩相关系数 | $R_S$, $\tau$ |
| concordant / discordant pair | 一致对 / 不一致对 | $n_c$, $n_d$ |
| goodness-of-fit test | 拟合优度检验 | |
| empirical distribution function | 经验分布函数 | $F_n$ |
| Kolmogorov–Smirnov test | Kolmogorov–Smirnov 检验（K–S 检验） | $D_n$ |
| quantile–quantile (Q–Q) plot | 分位数–分位数图（Q–Q 图） | |
| bias / mean squared error (MSE) | 偏差 / 均方误差 | $\operatorname{MSE} = \Var + \operatorname{Bias}^2$ |
| type I / type II error, power | 第一类 / 第二类错误，功效 | $\alpha$, $\beta$, $1-\beta$ |
| law of large numbers / central limit theorem | 大数定律 / 中心极限定理 | |
| rank / trace / determinant | 秩 / 迹 / 行列式 | $\rank$, $\tr$, $\det$ |
| eigenvalue / spectral decomposition | 特征值 / 谱分解 | |
| positive (semi)definite | 正定（半正定） | |
| orthogonal projection | 正交投影 | |
| singular value decomposition (SVD) | 奇异值分解 | $\bX = \mathbf{U}\mathbf{D}\mathbf{V}\T$ |
| gradient / Hessian matrix | 梯度 / Hessian 矩阵（海森矩阵） | $\nabla f$ |
| convex function / subgradient | 凸函数 / 次梯度 | $\partial f$ |

## Linear regression

| English | 中文 | Symbol / note |
|---|---|---|
| response (dependent variable) | 响应变量（因变量） | $y$ |
| predictor (independent / explanatory variable, regressor, covariate) | 预测变量（自变量、解释变量、回归变量、协变量） | $x_j$ |
| exogenous / endogenous variable | 外生变量 / 内生变量 | |
| regression coefficient | 回归系数 | $\beta_j$ |
| intercept | 截距 | $\beta_0$ |
| design matrix | 设计矩阵 | $\bX$ |
| error | 误差 | $\varepsilon_i$ |
| residual | 残差 | $e_i$ |
| fitted value | 拟合值 | $\widehat{y}_i$ |
| parameter / estimator / estimate | 参数 / 估计量 / 估计值 | $\bbeta$ / $\widehat{\bbeta}$ / numerical value |
| homoskedasticity | 同方差性 | $\Var(\varepsilon_i)=\sigma^2$ |
| ordinary least squares (OLS) | 普通最小二乘（OLS） | |
| normal equations | 正规方程 | $\bX\T\bX\bbeta=\bX\T\by$ |
| maximum likelihood estimation (MLE) | 极大似然估计 | |
| unbiased estimator | 无偏估计量 | |
| hat matrix / projection matrix | 帽子矩阵 / 投影矩阵 | $\bH$ |
| idempotent | 幂等 | $\bH^2=\bH$ |
| degrees of freedom | 自由度 | |
| total / regression / residual sum of squares | 总平方和 / 回归平方和 / 残差平方和 | $\mathit{SS}_T$, $\mathit{SS}_R$, $\mathit{SS}_E$ |
| mean square | 均方 | $\mathit{MS}$ |
| analysis of variance (ANOVA) | 方差分析 | |
| standard error | 标准误 | $\mathrm{SE}$ |
| confidence interval | 置信区间 | |
| prediction interval | 预测区间 | |
| extra sum of squares | 额外平方和 | $\mathit{SS}_R(\bbeta_1\mid\bbeta_2)$ |
| coefficient of determination | 决定系数 | $R^2$ |
| adjusted $R^2$ | 调整 $R^2$ | $R^2_{\mathrm{adj}}$ |
| leverage | 杠杆值 | $h_{ii}$ |
| standardized / studentized residual | 标准化残差 / 学生化残差 | $d_i$ / $r_i$ |
| influential observation | 强影响点 | |
| Cook's distance | Cook 距离 | $D_i$ |
| one-way analysis of variance | 单因素方差分析 | |
| treatment effect | 处理效应 | $\tau_i$ |
| fixed-effects model | 固定效应模型 | |
| treatment / replicate | 处理 / 重复 | |
| multiple comparisons | 多重比较 | |
| least significant difference (LSD) | 最小显著差异 | $\mathrm{LSD}$ |
| random-effects model / variance component | 随机效应模型 / 方差分量 | $\sigma_\tau^2$ |
| nuisance factor / block | 干扰因素 / 区组 | |
| randomized complete block design (RCBD) | 随机化完全区组设计 | |

## Model building

| English | 中文 | Symbol / note |
|---|---|---|
| standardized regression coefficient | 标准化回归系数 | $\beta_j^{*}$ |
| multicollinearity | 多重共线性 | |
| variance inflation factor (VIF) | 方差膨胀因子 | $\mathrm{VIF}_j$ |
| best subset selection | 最优子集选择 | |
| forward / backward stepwise selection | 向前 / 向后逐步选择 | |
| Mallows' $C_p$ | Mallows $C_p$ 统计量 | |
| Akaike / Bayesian information criterion | 赤池信息准则 / 贝叶斯信息准则 | AIC / BIC |
| upper envelope | 上包络线 | |
| overfitting | 过拟合 | |
| penalized (regularized) regression | 惩罚（正则化）回归 | |
| ridge regression | 岭回归 | $\lambda\lVert\bbeta\rVert_2^2$ |
| lasso | lasso（套索回归） | $\lambda\lVert\bbeta\rVert_1$ |
| elastic net | 弹性网 | |
| tuning (penalty) parameter | 调节（惩罚）参数 | $\lambda$ |
| shrinkage | 收缩 | |
| soft / hard thresholding | 软阈值 / 硬阈值 | $S(z,\gamma)$ |
| coordinate descent | 坐标下降 | |
| effective degrees of freedom | 有效自由度 | $\operatorname{df}(\lambda)$ |
| cross-validation | 交叉验证 | |
| regularization path | 正则化路径 | |

## Generalized linear models

| English | 中文 | Symbol / note |
|---|---|---|
| generalized linear model (GLM) | 广义线性模型 | |
| exponential family | 指数族 | |
| canonical parameter | 典则参数（自然参数） | $\theta$ |
| dispersion parameter | 离散参数 | $\phi$ |
| linear predictor | 线性预测子 | $\eta$ |
| link function / canonical link | 联系函数 / 典则联系函数 | $g$ |
| variance function | 方差函数 | $V(\mu)$ |
| deviance / saturated model | 偏差 / 饱和模型 | $D$ |
| Fisher information / Fisher scoring | Fisher 信息 / Fisher 得分法 | |
| iteratively reweighted least squares (IRLS) | 迭代重加权最小二乘 | |
| logistic regression | logistic 回归（逻辑回归） | |
| odds / odds ratio | 优势 / 优势比 | $e^{\beta_j}$ |
| marginal effect | 边际效应 | |
| separation | （完全）分离 | |
| pseudo-$R^2$ | 伪 $R^2$ | |
| Poisson regression | 泊松回归 | |
| rate ratio (relative risk) | 率比（相对风险） | $e^{\beta_j}$ |
| offset / exposure | 偏移项 / 暴露量 | $\ln E_i$ / $E_i$ |
| drop-in-deviance test | 偏差下降检验 | |
| Wald test | Wald 检验 | |
| Pearson / deviance residual | Pearson 残差 / 偏差残差 | |
| overdispersion | 过度离散 | |
| quasi-Poisson | 拟泊松 | |
| negative binomial regression | 负二项回归 | |
| zero-inflated Poisson (ZIP) | 零膨胀泊松 | |
| ordinal regression / cumulative link model | 有序回归 / 累积联系模型 | |
| proportional odds model | 比例优势模型 | |
| cut-point (threshold) | 切点（阈值） | $\theta_j$ |
| multinomial logit (MNL) / probit (MNP) | 多项 logit / 多项 probit | |
| random utility maximization | 随机效用最大化 | |
| choice set | 选择集 | $C_n$ |
| independence of irrelevant alternatives (IIA) | 无关选项独立性 | |
| likelihood ratio index | 似然比指数 | $\rho^2$ |

## Quantile regression

| English | 中文 | Symbol / note |
|---|---|---|
| quantile | 分位数 | $x_\tau$ |
| quantile regression | 分位数回归 | |
| pinball loss (quantile loss, check function) | 弹球损失（分位数损失、检验函数） | $\rho_\tau$ |
| median regression (least absolute deviations) | 中位数回归（最小一乘） | $\tau=0.5$ |
