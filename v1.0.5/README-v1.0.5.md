# The Embarrassment Principle: Data-Driven Preference for Undamped Oscillating Dark Energy

**Author:** Kaibing Li  
**Version:** 1.0.5  
**DOI:** (will be assigned by Zenodo)

## Overview

This repository contains the full code and data necessary to reproduce the results reported in the paper:

> Li Kaibing, "The Embarrassment Principle: Data-Driven Preference for Undamped Oscillating Dark Energy in a Continuous and Reversible Way" (2026)

**Key finding:** When the matter density parameter Ωm is not constrained by Planck priors, Pantheon+ SN and BAO data strongly prefer an undamped oscillating dark energy model over ΛCDM, with Δχ² = 73.89 (for 5 parameters; effective 4 parameters due to α=0). The oscillation amplitude decreases continuously to zero as Ωm is forced toward 0.315, exactly recovering ΛCDM.

## Reproduction Instructions

### Requirements
- Python 3.8+
- numpy, scipy, matplotlib

### Data Files
You need to download the Pantheon+ data and covariance matrix from the official repository:
- `pantheon+_data.txt`
- `Pantheon+SH0ES_STAT+SYS.cov`

Place them in the same directory as the code.

### Run the Fit
```bash
python fit_osc-1.0.5.py

The script will:
Load SN and BAO data
Fit ΛCDM and the oscillating model under four Ωm constraints
Print the results table (matching Table 1 in the paper)
Generate figures: fig_residual_final.png, fig_wz_final.png, fig_Hz_final.png

ΛCDM     : Ωm=0.1139, H0=74.83, χ²=2475.22
振荡模型 : A=0.7718, α=0.0000, β=21.23, Ωm=0.0832, H0=73.06, χ²=2401.32
Δχ² = 73.89

Key Implementation Notes
The oscillating dark energy density evolution uses positive sign in the exponential integral:
ρ_de(z) = ρ_de(0) exp(3A ∫ e^{-αx} sin(βx)/(1+x) dx)
Covariance matrix is not scaled; only a tiny diagonal jitter (1e-8) is added for numerical stability.
BAO data use D_V/r_d with fixed r_d = 147.78 Mpc.
Redshift sampling density is 600 points from z=0 to z=2.5 to ensure numerical accuracy.
Negative energy densities are truncated to zero (physical safeguard).
No caching or precomputed interpolation is used; all integrals are computed in real time to avoid memory/pointer issues.
Negative energy densities are truncated to zero (physical safeguard).

Version History
v1.0.5 (current): Increased sampling to 600 points, Ωm lower bound 0.01, Δχ²=73.89.
v1.0.4 (previous Zenodo release): 250 points, Ωm lower bound 0.1, Δχ²=69.76.

Acknowledgments
The author thanks DouBao, DeepSeek, QianWen for extensive support in theoretical derivation, code implementation, debugging, and scientific writing.

If you use this code or the results, please cite:
Li, K. (2026). The Embarrassment Principle: Oscillating Dark Energy Preferred by Low-Redshift Data... [Zenodo/arXiv]. DOI: (to be assigned)