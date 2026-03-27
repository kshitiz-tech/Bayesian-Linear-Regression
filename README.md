# Bayesian Linear Regression

This project is a small research-style study of linear regression from a deeper mathematical perspective. Instead of treating linear regression as only a fitting procedure, the notebook builds it as a Bayesian model:

- a Gaussian likelihood for the data
- a Gaussian prior over parameters
- a MAP solution obtained from linear algebra
- a full posterior distribution over coefficients
- predictive uncertainty for test points

The goal is not just to get a decent prediction score. The main goal is to understand why the model works, what each matrix means, and how prior assumptions affect the final regression solution.

## Repository Contents

- `Implementation_BLR.ipynb`  
  Main notebook containing the full implementation, experiments, visualizations, and interpretation.

- `Mathematical_Derivations.pdf`  
  Mathematical derivation of the Bayesian linear regression model used in the notebook.

## Project Focus

This work was done to understand linear regression beyond the standard "fit a line" viewpoint.

The notebook studies:

- how the design matrix is built
- why standardization matters mathematically, not just numerically
- how the prior covariance matrix `B` controls shrinkage of coefficients
- how the MAP estimator arises from maximizing the posterior
- why the posterior mean matches the MAP estimate in this Gaussian setting
- how uncertainty in parameters leads to uncertainty in predictions

In short, this project treats linear regression as a probabilistic inference problem.

## Dataset

The notebook uses the California Housing dataset from `sklearn.datasets`.

Before fitting the model, capped target values are removed. This is important because the Bayesian linear regression setup assumes Gaussian noise with constant variance. The notebook argues that the capped house prices violate that assumption and introduce heteroscedasticity, especially near the upper end of the target range.

## Method Overview

The workflow in the notebook is:

1. Load the California Housing data.
2. Inspect feature-target correlations.
3. Study the target distribution and remove capped target values.
4. Split the data into train and test sets.
5. Standardize features using training statistics only.
6. Build the design matrix `Phi` by adding a bias column.
7. Construct a prior covariance matrix `B` using feature correlations.
8. Solve for the MAP estimate

$$
\theta_{MAP} = (\Phi^T \Phi + \sigma^2 B^{-1})^{-1} \Phi^T y
$$

9. Evaluate the model with RMSE and residual plots.
10. Tune the prior scale `alpha` using validation data.
11. Add selective nonlinear features such as `MedInc^2` and `Latitude * Longitude`.
12. Compute the posterior mean `m_N` and covariance `S_N`.
13. Use the posterior to produce predictive means and predictive uncertainty bands.

## Mathematical Perspective

This project is most useful if read together with the derivations PDF.

The core idea is:

- likelihood: `y | theta ~ N(Phi theta, sigma^2 I)`
- prior: `theta ~ N(0, B)`
- posterior: `theta | D ~ N(m_N, S_N)`

Because both the likelihood and prior are Gaussian, the posterior is also Gaussian. That makes the algebra clean and lets us interpret the model in two complementary ways:

- as optimization through the MAP estimate
- as inference through the full posterior distribution

One of the most important ideas in this project is that `B_inv` acts like a regularization force. A stronger prior shrinks coefficients harder toward zero. A weaker prior lets the data dominate more.

## What the Notebook Shows

The notebook moves from simple to richer modeling:

- A baseline Bayesian linear regression model with degree-1 features.
- Validation-based tuning of the prior scale `alpha`.
- Residual analysis to understand where the model fails.
- Selective feature engineering motivated by residual structure.
- Posterior covariance analysis to measure coefficient uncertainty.
- Predictive intervals to show confidence around predictions.

An important conclusion from the notebook is that better mathematics does not automatically mean a better RMSE. In the selective polynomial experiment, the score does not improve much, but the exercise is still valuable because it reveals what the model is and is not capturing.

## Guidance For Reading This Project

To get the most from this repository, study it in this order:

1. Read `Mathematical_Derivations.pdf` first.
2. Open `Implementation_BLR.ipynb` and follow the notebook cell by cell.
3. Pay close attention to how `Phi`, `B`, `B_inv`, `A`, `m_N`, and `S_N` are constructed.
4. Compare the MAP formula in the PDF with the actual NumPy implementation.
5. Look at the residual plots before jumping to more features.
6. Treat the posterior covariance as a major result, not as an extra calculation.

If your goal is learning, do not skip the interpretation cells. They explain why preprocessing and prior design matter in Bayesian regression.

## Practical Notes

- Feature standardization is essential here because the prior is defined over coefficients. Without standardization, coefficients are not directly comparable across features.
- The prior variances are linked to feature-target correlations. This is a modeling choice meant to encode intuition about feature importance.
- `sigma^2` is fixed in the notebook rather than estimated from data, so the notebook should be understood as an educational implementation rather than a fully optimized production model.
- Cross-validation is used to tune `alpha`, which controls how loose or tight the prior is.

## How To Run

Open the notebook in Jupyter:

```bash
jupyter notebook Implementation_BLR.ipynb
```

You will need a Python environment with common scientific packages such as:

- `numpy`
- `matplotlib`
- `scikit-learn`
- `jupyter`

## Suggested Extensions

If you want to take this further, good next steps are:

- estimate `sigma^2` from the data instead of fixing it
- use a more systematic prior design instead of correlation-based scaling alone
- try richer polynomial or interaction features
- engineer geography-aware features before standardization
- compare this closed-form Bayesian solution with ridge regression
- test how predictive uncertainty changes under different priors

## Final Note

This repository is best understood as a learning document in code form. The implementation is valuable not only because it predicts house prices, but because it makes the mathematics of linear regression visible: priors, shrinkage, posterior inference, and uncertainty quantification are all explicit.
