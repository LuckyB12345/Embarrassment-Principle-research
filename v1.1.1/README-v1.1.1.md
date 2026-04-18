# The Embarrassment Principle: Data-Driven Preference for Undamped Oscillating Dark Energy

**Author:** Kaibing Li  
**Version:** 1.1.1  
**DOI:** (to be assigned)

## Overview

This repository contains the full code and data necessary to reproduce the results reported in the paper:

> Li Kaibing, "Pantheon+ and DESI BAO Data Reveal a Geometric Tension: A Δχ²≈147 Signal for Late-Time Oscillatory Residuals" (2026)

**Key finding:** When the matter density parameter Ωm is not constrained by Planck priors, Pantheon+ SN and DESI BAO data strongly prefer an undamped oscillating dark energy model over ΛCDM, with Δχ² = 147.28 (fixed rd = 147.09 Mpc). The oscillation amplitude decreases continuously to zero as Ωm is forced toward 0.315, exactly recovering ΛCDM. Monte Carlo simulations rule out overfitting (max simulated Δχ² = 4.89, p < 0.03).

## Reproduction Instructions

### Requirements
- Python 3.8+
- numpy, scipy, matplotlib

### Data Files
You need to download the Pantheon+ data and covariance matrix from the official repository:
- `pantheon+_data.txt`
- `Pantheon+SH0ES_STAT+SYS.cov`

Place them in the same directory as the code.

### Run the Main Fit
```bash
python fit_osc-1.1.1.py

This will produce Table 1 results and the figures fig_residual_highprecision.png, fig_wz_highprecision.png, fig_Hz_highprecision.png.

Run the Monte Carlo Test (Optional)
python montecarlo_final_v1.1.1.py
This will perform 30 simulations under the null hypothesis and output the distribution of Δχ², confirming the signal is not due to overfitting.

Key Implementation Notes
Integration tolerance: 1e-7, limit=500.
Covariance matrix is not scaled; only a tiny diagonal jitter (1e-8) is added.
BAO data use D_V/r_d with fixed r_d = 147.09 Mpc (DESI fiducial).
Parallel processing for constraints (up to 4 processes) speeds up the main fit.
Monte Carlo uses 16 parallel processes (auto-detects cores).


Version History
v1.1.1 (current): High precision (1e-7), parallel constraints, Monte Carlo with 30 simulations.
v1.1.0 (current): DESI BAO, rd=147.09, fixed and free rd fits, Δχ²=147.28.Initial high-precision version (no Monte Carlo).
v1.0.5: BOSS BAO, rd=147.78, Δχ²=73.89.
v1.0。4: initial Zenodo release (Ωm≥0.1, Δχ²=69.76).

License
MIT License.

Acknowledgments
The author thanks computational tools for code development and data analysis.