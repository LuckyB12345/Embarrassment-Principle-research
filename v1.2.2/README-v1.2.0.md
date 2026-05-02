```markdown
# Embarrassment Principle: Final Code for α = -12 Cosmological Oscillation Model

This repository contains the final code used to produce the results in the paper  
*“The Embarrassment Principle: The Universe Does Not Tend Toward Equilibrium – It Oscillates Forever”* (final version, α = -12, Δχ² = 459.34).  
All scripts are self‑contained and rely only on public Pantheon⁺+DESI BAO data.

## 1. Data Requirements

Before running any script, download the following files and place them in the same directory as the code:

- `pantheon+_data.txt` – Pantheon⁺ supernova data (from [Pantheon⁺ repository](https://github.com/PantheonPlusSN/Data))
- `Pantheon+SH0ES_STAT+SYS.cov` – corresponding covariance matrix (same source)

The DESI BAO measurements are hard‑coded in the scripts (z = 0.5,0.7,1.0 with D_V/r_d and errors).

## 2. Environment & Dependencies

All scripts are written in Python 3.9+ and require the following packages:

```bash
numpy scipy matplotlib
```

Install them with, e.g.:

```bash
pip install numpy scipy matplotlib
```

## 3. Overview of Scripts

| Script | Purpose | α treatment | Output |
|--------|---------|--------------|--------|
| `fit_osc_final_alpha_-12_to_30.py` | Main fit: four Ωₘ constraints | free in [-12,30] | Table 1 (ΛCDM & oscillatory parameters, χ², Δχ²) |
| `fine_scan_alpha.py` | Fine scan α from -8 to -12, plus free fit | free in [-12,30] | χ²(α) table, best‑fit α = -12 |
| `loose_alpha_test.py` | Loose bound test: α ∈ [-20,50] | free in [-20,50] | Demonstrates α=-12 is numerical limit (overflow warnings) |
| `redshift_cut_detailedv1.2.0.py` | Redshift cut test, fixed α = -12 | fixed α = -12 | Δχ² vs z_min table |
| `montecarlo_final_v1.2.0.py` | Monte Carlo (10 realisations) under ΛCDM null | free in [-12,30] | max Δχ² (simulated) ≪ 459.34 |
| `smoke_test-v1.2.0.py` | Smoke test: recover true parameters from mock data | free in [-12,30] | recovered α ≈ -12, Δχ² ≈ 250–275 |

## 4. Execution Order (Suggested)

Run the scripts in the following sequence to reproduce the paper’s results:

1. **Main fit**  
   ```bash
   python fit_osc_final_alpha_-12_to_30.py
   ```  
   → prints the main table (four constraints). The unconstrained case gives Δχ² = 459.34 and α = -12.00.

2. **Fine scan** (confirms α = -12 is the preferred boundary)  
   ```bash
   python fine_scan_alpha.py
   ```  
   → shows χ² decreasing monotonically toward α = -12 and free fit converging to -12.

3. **Loose bound test** (verifies that α cannot be pushed below -12 numerically)  
   ```bash
   python loose_alpha_test.py
   ```  
   → will produce overflow/roundoff warnings; the optimal α remains at -12.

4. **Redshift cut test** (signal robustness)  
   ```bash
   python redshift_cut_detailedv1.2.0.py
   ```  
   → outputs a table with z_min, N_SN, χ² values and Δχ². The improvement stays above 435 even after removing z<0.2 SNe.

5. **Monte Carlo** (null test)  
   ```bash
   python montecarlo_final_v1.2.0.py
   ```  
   → runs 10 low‑precision realisations. Expected output: max Δχ² ≈ 4.0, far below 459.34.

6. **Smoke test** (code self‑consistency)  
   ```bash
   python smoke_test-v1.2.0.py
   ```  
   → performs 5 mock‑data generations and fits; reports average recovered parameters. Expected: α_recovered ≈ -12, Δχ² ≈ 250–275.

## 5. Expected Key Numbers

- **Unconstrained fit**: Δχ² = 459.34, α = -12.00, Ωₘ(osc) = 0.0484, H₀(osc) = 75.94 km/s/Mpc  
- **Redshift cut (z_min=0.20)**: Δχ² = 436.59  
- **Monte Carlo max Δχ²** = 4.00 (10 realisations)  
- **Smoke test**: α_recovered = -11.9 … -12.0

All scripts are fully deterministic given the fixed random seeds.

## 6. Notes on Reproducibility

- The Monte Carlo script uses **low integration precision** (`epsrel=1e-5`, `limit=200`) for speed, but the conclusion (max Δχ² ≪ 459) holds regardless.  
- The redshift cut script **fixes α = -12** because this is the global best‑fit value; it tests the signal’s redshift distribution under the final model.  
- The loose bound test extends α to -20; numerical warnings appear, confirming that α = -12 is the stable limit.

## 7. Citation

If you use these codes, please cite the accompanying paper:

> Li, K. (2026). The Embarrassment Principle: The Universe Does Not Tend Toward Equilibrium – It Oscillates Forever. [arXiv/DOI]

## 8. License

© 2026 Li Kaibing.  
This project is released under the **MIT License**.

## 9. Contact

For questions or to report issues, please contact the author: **Kaibing Li** (806255397@qq.com)
```
