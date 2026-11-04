# Notation

The symbols below are used in several chapters. A chapter that introduces further symbols lists them in its own Notation section.

| Symbol | Meaning |
|---|---|
| $n$ | number of observations |
| $k$ | number of predictors (excluding the intercept) |
| $p$ | number of regression coefficients; $p = k + 1$ when an intercept is included |
| $y_i$, $\by$ | observed response, response vector in $\R^n$ |
| $\bx_i$, $\bX$ | predictor vector of observation $i$, design matrix in $\R^{n\times p}$ |
| $\bbeta$, $\widehat{\bbeta}$ | coefficient vector and its estimator |
| $\bvarepsilon$, $\be$ | unobserved errors and observed residuals |
| $\bH$ | hat (projection) matrix $\bX(\bX\T\bX)^{-1}\bX\T$ |
| $\mu_i$, $\eta_i$ | conditional mean $\E[Y_i\mid\bx_i]$ and linear predictor $\bx_i\T\bbeta$ |
| $\lambda$ | penalty (tuning) parameter in ridge and lasso regression |
