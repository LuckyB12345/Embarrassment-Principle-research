# Embarrassment Principle: v1.2.4 — Globally Consistent Oscillating Dark Energy Model

This repository contains the code and data used to produce the results in the paper  
*“Globally Consistent Oscillating Dark Energy: A Phenomenological Projection of the Embarrassment Principle”* (Li, 2026).  
All scripts are self-contained and rely only on public Pantheon⁺, DESI BAO, and Planck 2018 compressed likelihood data.

## 1. Data Requirements

Before running the main script, ensure the following files are present in the same directory:

- `pantheon+_data.txt` – Pantheon⁺ supernova distance modulus data
- `Pantheon+SH0ES_STAT+SYS.cov` – full statistical+systematic covariance matrix
- `Pantheon+SH0ES.dat` – full Pantheon+ catalog (optional, for reference only)

The DESI BAO measurements and Planck 2018 compressed likelihood are hard-coded into the main script.

## 2. Environment & Dependencies

The main script is written in Python 3.9+ and requires only standard scientific packages:

```bash
numpy scipy
```

Install them with:
```bash
pip install numpy scipy
```

No additional cosmology libraries (Cobaya, CAMB) are required.

## 3. Overview of Scripts

| Script | Purpose | Model Form | Output |
|--------|---------|------------|--------|
| `run_planck_combined.py` | Main joint fit to SN+BAO+Planck | $1+w(z) = A\sin(bz)/(1+z)^2$ | ΛCDM vs oscillating model parameters, $\chi^2$, $\Delta\chi^2$, $\Delta$AIC, $\Delta$BIC |

## 4. Execution Order (Suggested)

1. **Main joint fit (SN + BAO + Planck 2018)**  
   ```bash
   python run_planck_combined.py
   ```  
   This script will:
   - Fit the standard ΛCDM model
   - Fit the globally consistent oscillating dark energy model
   - Print best-fit parameters, $\chi^2$, $\Delta\chi^2$, $\Delta$AIC, and $\Delta$BIC
   - Verify high-redshift behavior of $1+w(z)$

## 5. Expected Key Numbers

- **Oscillating model vs ΛCDM**:
  - $\Delta\chi^2 = 56.54$
  - $\Delta$AIC = 52.5
  - $\Delta$BIC = 41.7
- **Best-fit parameters**:
  - $\Omega_m = 0.2742$
  - $H_0 = 71.16$ km/s/Mpc
  - $A = 0.60$
  - $b = 19.84$
- **Physical checks**:
  - High-redshift convergence: $1+w(z) \to 0$ as $z\to\infty$
  - No numerical instabilities or boundary overflow

All results are deterministic given the fixed integration settings and optimizer.

## 6. Notes on Reproducibility

- The Planck 2018 likelihood uses the standard compressed form for $(\Omega_m, H_0)$ with central values $(0.3111, 67.66)$ and correlation coefficient $r=0.18$.
- The BAO data points are fixed at $z=0.5,0.7,1.0$ with DESI 2024 measurements.
- The best-fit amplitude $A$ saturates the upper bound of $0.6$. Relaxing the bound to $1.0$ does not change $\Delta\chi^2$, confirming no need for larger oscillations.

## 7. Citation

If you use these codes or results, please cite the accompanying paper:

> Li, K. (2026). Globally Consistent Oscillating Dark Energy: A Phenomenological Projection of the Embarrassment Principle. [arXiv/DOI]

## 8. License

© 2026 Li Kaibing.  
This project is released under the **MIT License**.

## 9. Contact

For questions or to report issues, please contact the author: **Kaibing Li** (806255397@qq.com)
```
