```markdown
# Embarrassment Principle: v1.2.5.1 — Globally Consistent Oscillating Dark Energy Model

This repository contains the code and data used to produce the results in the paper  
*“Globally Consistent Oscillating Dark Energy: A Phenomenological Projection of the Embarrassment Principle”* (Li, 2026).  
All scripts are self-contained and rely only on public Pantheon⁺, BAO, and Planck 2018 compressed likelihood data.

**Version v1.2.5.1** includes **full MCMC validation** (2000 steps, 16 chains) and an updated corner plot (`corner_mcmc.png`), confirming the Fisher matrix uncertainties and demonstrating Gaussianity of the posterior.

---

## 1. Data Requirements

Before running the main script, ensure the following files are present in the same directory:

- `pantheon+_data.txt` – Pantheon⁺ supernova distance modulus data
- `Pantheon+SH0ES_STAT+SYS.cov` – full statistical+systematic covariance matrix
- `Pantheon+SH0ES.dat` – full Pantheon+ catalog (optional, for reference only)

The BAO measurements and Planck 2018 compressed likelihood are hard-coded into the main script.

---

## 2. Environment & Dependencies

The main script is written in Python 3.9+ and requires only standard scientific packages:

```bash
numpy scipy matplotlib
```

For MCMC and corner plots, additionally install:

```bash
pip install emcee corner tqdm
```

No other cosmology libraries (Cobaya, CAMB) are required.

---

## 3. Overview of Scripts

| Script | Purpose | Model Form | Output |
|--------|---------|------------|--------|
| `fisher_v1.2.5.py` | Main joint fit + Fisher matrix | $1+w(z) = A\sin(bz)/(1+z)^2$ | Best-fit params, $\chi^2$, $\Delta\chi^2$, Fisher 1$\sigma$ errors |
| `mcmc_validation.py` | MCMC posterior sampling (16 chains, 2000 steps) | Same model | MCMC medians, credible intervals, `corner_mcmc.png` |
| `plot_figures.py` | Plot final paper figures | Best-fit model | $1+w(z)$, $H(z)$, Pantheon+ residuals |

---

## 4. Execution Order (Suggested)

1. **Main joint fit (SN + BAO + Planck 2018)**  
   ```bash
   python fisher_v1.2.5.py
   ```  
   This script will:
   - Fit $\Lambda$CDM and the oscillating dark energy model
   - Print best-fit parameters, $\chi^2$, $\Delta\chi^2$
   - Compute Fisher matrix and 1$\sigma$ uncertainties
   - Verify high-redshift behavior of $1+w(z)$

2. **MCMC validation (optional but recommended)**  
   ```bash
   python mcmc_validation.py
   ```  
   This runs a short MCMC chain (2000 steps, 16 chains) to cross‑check Fisher errors.  
   On a 16‑core machine it takes **~4–5 hours**.  
   Output: posterior medians, 68% credible intervals, and `corner_mcmc.png`.

3. **Plot final paper figures**  
   ```bash
   python plot_figures.py
   ```

---

## 5. Expected Key Numbers (v1.2.5.1)

| Quantity | Value (Oscillating DE) |
|----------|------------------------|
| $\Delta\chi^2$ (vs $\Lambda$CDM) | $104.41$ |
| Best‑fit $A$ | $1.4087$ |
| Best‑fit $b$ | $26.60$ |
| Best‑fit $\Omega_m$ | $0.2691$ |
| Best‑fit $H_0$ (km/s/Mpc) | $70.16$ |

**MCMC 68% credible intervals (adopted in paper):**  
- $A = 1.4151^{+0.1527}_{-0.1374}$  
- $b = 26.7751^{+0.7598}_{-0.6512}$  
- $\Omega_m = 0.2687^{+0.0056}_{-0.0063}$  
- $H_0 = 70.1977^{+0.2002}_{-0.2350}$

**Physical checks:**  
- High‑redshift convergence: $1+w(z)\to0$ as $z\to\infty$  
- No numerical instabilities or boundary saturation  
- Robust oscillation signal, Gaussian posterior (see `corner_mcmc.png`)

---

## 6. Notes on Reproducibility

- The Planck 2018 compressed likelihood is hard‑coded with $(\Omega_m=0.3111, H_0=67.66)$ and correlation $r=0.18$.
- BAO data points are fixed at $z=0.5,0.7,1.0$.
- The best‑fit solution converges naturally without hitting parameter boundaries.
- The MCMC prior restricts $A\in[0.5,2.5]$, $b\in[20,35]$, $\Omega_m\in[0.22,0.32]$, $H_0\in[69,72]$ – a physically justified range that excludes a statistically favorable but unphysical secondary minimum.

---

## 7. Citation

If you use these codes or results, please cite both the paper and the Zenodo repository:

> Li, K. (2026). Globally Consistent Oscillating Dark Energy: A Phenomenological Projection of the Embarrassment Principle.  
> Li, K. (2026). Embarrassment‑Principle‑research (v1.2.5.1). Zenodo. https://doi.org/10.5281/zenodo.19828321

---

## 8. License

© 2026 Li Kaibing.  
This project is released under the **MIT License**.

---

## 9. Contact

For questions or to report issues, please contact the author: **Kaibing Li** (806255397@qq.com)
```
