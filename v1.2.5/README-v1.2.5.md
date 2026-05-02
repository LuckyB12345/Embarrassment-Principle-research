# 🟢 最终定稿 README.md（仅 Zenodo、无 arXiv、完全同步 v1.2.5）
```markdown
# Embarrassment Principle: v1.2.5 — Globally Consistent Oscillating Dark Energy Model

This repository contains the code and data used to produce the results in the paper  
*“Globally Consistent Oscillating Dark Energy: A Phenomenological Projection of the Embarrassment Principle”* (Li, 2026).  
All scripts are self-contained and rely only on public Pantheon⁺, BAO, and Planck 2018 compressed likelihood data.

## 1. Data Requirements

Before running the main script, ensure the following files are present in the same directory:

- `pantheon+_data.txt` – Pantheon⁺ supernova distance modulus data
- `Pantheon+SH0ES_STAT+SYS.cov` – full statistical+systematic covariance matrix
- `Pantheon+SH0ES.dat` – full Pantheon+ catalog (optional, for reference only)

The BAO measurements and Planck 2018 compressed likelihood are hard-coded into the main script.

## 2. Environment & Dependencies

The main script is written in Python 3.9+ and requires only standard scientific packages:

```bash
numpy scipy matplotlib
```

Install them with:
```bash
pip install numpy scipy matplotlib
```

No additional cosmology libraries (Cobaya, CAMB) are required.

## 3. Overview of Scripts

| Script | Purpose | Model Form | Output |
|--------|---------|------------|--------|
| `fisher_v1.2.5.py` | Main joint fit + Fisher matrix | $1+w(z) = A\sin(bz)/(1+z)^2$ | Best-fit params, $\chi^2$, $\Delta\chi^2$, Fisher 1$\sigma$ errors |
| `plot_figures.py` | Plot final paper figures | Best-fit model | $1+w(z)$, $H(z)$, Pantheon+ residuals |

## 4. Execution Order (Suggested)

1. **Main joint fit (SN + BAO + Planck 2018)**  
   ```bash
   python fisher_v1.2.5.py
   ```  
   This script will:
   - Fit the standard $\Lambda$CDM model
   - Fit the globally consistent oscillating dark energy model
   - Print best-fit parameters, $\chi^2$, $\Delta\chi^2$
   - Compute Fisher matrix and 1$\sigma$ uncertainties
   - Verify high-redshift behavior of $1+w(z)$

2. **Plot final paper figures**
   ```bash
   python plot_figures.py
   ```

## 5. Expected Key Numbers

- **Oscillating model vs $\Lambda$CDM**:
  - $\Delta\chi^2 = 104.41$
- **Best-fit parameters**:
  - $\Omega_m = 0.2691$
  - $H_0 = 70.16$ km/s/Mpc
  - $A = 1.4087$
  - $b = 26.60$
- **Fisher 1$\sigma$ uncertainties**:
  - $A = \pm 0.0960$
  - $b = \pm 0.4204$
  - $\Omega_m = \pm 0.0039$
  - $H_0 = \pm 0.1623$
- **Physical checks**:
  - High-redshift convergence: $1+w(z) \to 0$ as $z\to\infty$
  - No numerical instabilities or boundary saturation
  - Robust, well-constrained oscillation signal

All results are deterministic given the fixed integration settings and optimizer.

## 6. Notes on Reproducibility

- The Planck 2018 likelihood uses the standard compressed form for $(\Omega_m, H_0)$ with central values $(0.3111, 67.66)$ and correlation coefficient $r=0.18$.
- The BAO data points are fixed at $z=0.5,0.7,1.0$.
- The best-fit solution converges naturally without hitting parameter boundaries.

## 7. Citation

If you use these codes or results, please cite both the paper and the Zenodo repository:

> Li, K. (2026). Globally Consistent Oscillating Dark Energy: A Phenomenological Projection of the Embarrassment Principle.
> Li, K. (2026). Embarrassment-Principle-research (v1.2.5). Zenodo. https://doi.org/10.5281/zenodo.19828321

## 8. License

© 2026 Li Kaibing.  
This project is released under the **MIT License**.

## 9. Contact

For questions or to report issues, please contact the author: **Kaibing Li** (806255397@qq.com)
```
