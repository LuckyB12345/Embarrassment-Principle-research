# The Embarrassment Principle: Data-Driven Preference for Undamped Oscillating Dark Energy

**Author:** Kaibing Li  
**Version:** 1.1.0  
**DOI:** (to be assigned by Zenodo)

## Overview

This repository contains the full code and data necessary to reproduce the results reported in the paper:

> Li Kaibing, "Pantheon+ and DESI BAO Data Reveal a Statistically Significant Oscillatory Residual Unaccounted for by ΛCDM under Weak Ωm Priors" (2026)

**Key finding:** When the matter density parameter Ωm is not constrained by Planck priors, Pantheon+ SN and DESI BAO data strongly prefer an undamped oscillating dark energy model over ΛCDM, with Δχ² = 147.28 (fixed rd = 147.09 Mpc) and Δχ² = 145.12 (free rd). The oscillation amplitude decreases continuously to zero as Ωm is forced toward 0.315, exactly recovering ΛCDM.

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
python fit_osc-1.1.0.py

The script will:
Load SN and BAO data (DESI default)Fit ΛCDM and the oscillating model under four Ωm constraints (fixed rd)

Also perform a free-rd robustness test
Print the results tables (matching Table 1 in the paper)
Generate figures: fig_residual_final.png, fig_wz_final.png, fig_Hz_final.png    

Expected Output (unconstrained fixed rd)
ΛCDM     : Ωm=0.0177, H0=76.25, χ²=3749.77
振荡模型 : A=0.7691, α=0.0000, β=18.18, Ωm=0.0100, H0=73.54, χ²=3602.49
Δχ² = 147.28

Key Implementation Notes
The oscillating dark energy density evolution uses positive sign in the exponential integral.
Covariance matrix is not scaled; only a tiny diagonal jitter (1e-8) is added for numerical stability.
BAO data use D_V/r_d with fixed r_d = 147.09 Mpc (DESI fiducial) or free.
Real-time integration, no caching, high precision.
Negative energy densities are truncated to zero.

Version History
v1.1.0 (current): DESI BAO, rd=147.09, fixed and free rd fits, Δχ²=147.28.
v1.0.5: BOSS BAO, rd=147.78, Δχ²=73.89.
v1.0。4: initial Zenodo release (Ωm≥0.1, Δχ²=69.76).

License
MIT License.

Acknowledgments
The author thanks DouBao, DeepSeek, QianWen for extensive support.