# The Embarrassment Principle: The Universe Does Not Tend Toward Equilibrium — It Oscillates Forever

**Author:** Kaibing Li  
**Version:** 1.1.1.5  
**DOI:** [10.5281/zenodo.19642665](https://doi.org/10.5281/zenodo.19642665)

## Overview

This repository contains the full code and data necessary to reproduce the results reported in the paper:

> Li Kaibing, "The Embarrassment Principle: The Universe Does Not Tend Toward Equilibrium — It Oscillates Forever" (2026)

**Key finding:** When the matter density parameter Ωm is not constrained by Planck priors, Pantheon+ SN and DESI BAO data strongly prefer an undamped oscillating dark energy model over ΛCDM, with Δχ² = 147.28 (fixed rd = 147.09 Mpc). The oscillation amplitude decreases continuously to zero as Ωm is forced toward 0.315, exactly recovering ΛCDM. Monte Carlo simulations rule out overfitting (max simulated Δχ² = 4.89, p < 0.03). Redshift-cut tests confirm that the oscillation frequency β ≈ 18.2 is stable for z < 0.1, and the signal is confined to z < 0.1 — consistent with a late‑time dynamical effect. A null test on mock ΛCDM data yields Δχ² = 0.00, confirming no numerical bias. The vanishing damping (α=0) implies the universe does not tend toward static equilibrium; it oscillates forever. This is the **Embarrassment Principle**.

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

Run the Redshift-Cut Test (Optional)
python redshift_cut_detailed.py   # (included in the package)
This performs a detailed redshift cut test (z_min = 0.00, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.20) and produces Table 2 of the paper, demonstrating that the oscillation frequency β is stable (≈18.2) for z_min ≤ 0.09 and that the signal disappears at z_min = 0.10.

Run the Smoke Test (Optional)
python smoke_test.py
This generates mock data from the best-fit ΛCDM model and fits it with the oscillatory model. The output Δχ² = 0.00 confirms that the code does not artificially create oscillatory signals from pure ΛCDM data.

Key Implementation Notes
Integration tolerance: 1e-7, limit=500.
Covariance matrix is not scaled; only a tiny diagonal jitter (1e-8) is added.
BAO data use D_V/r_d with fixed r_d = 147.09 Mpc (DESI fiducial).
Parallel processing for constraints (up to 4 processes) speeds up the main fit.
Monte Carlo uses 16 parallel processes (auto-detects cores).


Version History
v1.1.1.5 (current): Added detailed redshift-cut test (0.05–0.20) and smoke test; refined discussion of signal localisation to z<0.1.
v1.1.1.4 : Added redshift-cut robustness test, refined dynamical inconsistency argument (point attractor vs. limit cycle), and included smoke test (Δχ²=0 on mock ΛCDM data). Core results unchanged.
v1.1.1.3: Revised title, abstract, and added dynamical inconsistency argument (point attractor vs. limit cycle). Core results unchanged.
v1.1.1 : High precision (1e-7), parallel constraints, Monte Carlo with 30 simulations.
v1.1.0 : DESI BAO, rd=147.09, fixed and free rd fits, Δχ²=147.28.Initial high-precision version (no Monte Carlo).
v1.0.5: BOSS BAO, rd=147.78, Δχ²=73.89.
v1.0。4: initial Zenodo release (Ωm≥0.1, Δχ²=69.76).

License
MIT License.

Acknowledgments
The author thanks computational tools for code development and data analysis.