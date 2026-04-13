# Embarrassment-Principle-research
Oscillating Dark Energy Model from the Embarrassment Principle

## Overview
This repository contains the code, LaTeX manuscript, and figures for an oscillating dark energy model motivated by the **Embarrassment Principle**:
> All real physical systems avoid absolute static equilibrium and naturally oscillate around equilibrium.

We fit the model to **Pantheon+ SNIa** and **BOSS DR12 BAO** data, finding strong statistical evidence against the standard ΛCDM model.

## Key Results
### Free fit (α ≥ 0)
- A = 0.600, α = 0.00, β = 22.5
- Ωm = 0.299, H0 = 72.00
- Δχ² = 109.3 | BIC = 86.9 | ΔAIC = 103.3

### Conservative fit (α ≥ 0.1)
- A = 0.600, α = 0.10, β = 22.5
- Ωm = 0.299, H0 = 72.00
- Δχ² = 108.5 | BIC = 86.2 | ΔAIC = 102.5

A shallow χ² plateau appears at **α ∈ [0, 0.1]**, indicating a parameter degeneracy to be broken by future high-redshift data.

## Model
\[
w(z) = -1 + A e^{-\alpha z} \sin(\beta z)
\]

## Files
- `embarrassment-principle-research.tex`                 - Full LaTeX manuscript for journal submission
- `fig1_residual.pdf`         - Residuals vs ΛCDM
- `fig2_wz.pdf`               - Dark energy equation of state
- `fig3_hz.pdf`               - Hubble parameter comparison
- `chi2_vs_alpha.pdf`         - χ² evolution with damping parameter α
- `embarrassment-principle-research.py`                     - Fitting code (L-BFGS-B minimization)

## Data
- Pantheon+ SNe Ia (full covariance)
- BOSS DR12 BAO
- Planck 2018 priors

## Acknowledgments
We acknowledge the Pantheon+, BOSS, and Planck collaborations for publicly releasing data products.

## Author
Li KaiBing
with contributions from DouBao & DeepSeek