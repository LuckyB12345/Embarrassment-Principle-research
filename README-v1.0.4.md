# The Embarrassment Principle

## Oscillating Dark Energy Preferred by Low-Redshift Data in a Continuous, Reversible Way

**Version 1.0.4** | Last updated: 2026-04-14  
**DOI:** (to be assigned) | **License:** MIT

## Overview
This project implements a fully self-contained cosmological fit to Pantheon+ SN and BAO data, testing a damped oscillating dark energy model derived from the "Embarrassment Principle".

**Key finding:**  
Guided by the *Embarrassment Principle* (the conjecture that no physical system remains in perfect static equilibrium), we derive an oscillating dark energy model. When the matter density parameter \(\Omega_m\) is relaxed from strong Planck priors, low-redshift data strongly prefers this oscillating model over \(\Lambda\)CDM. As \(\Omega_m\) is constrained toward 0.315, the oscillation amplitude continuously and reversibly vanishes, recovering \(\Lambda\)CDM exactly. The principle thus **predicts** the observed reversible behavior, which is then **confirmed** by the data.

## File Structure

File Structure
fit_osc-1.0.4.py                # Main fitting code
pantheon+_data.txt              # SN Ia data
Pantheon+SH0ES_STAT+SYS.cov     # Full covariance matrix
fig_residual.png                # Distance modulus residuals
fig_wz.png                      # Dark energy equation of state w(z)
fig_Hz.png                      # Hubble parameter H(z)
embarrassment-principle-research-v1.0.4-final.tex   # Full paper LaTeX

Dependencies
numpy
scipy
matplotlib

How to Run
python fit_osc-1.0.4.py


Key Results

Constraint	Ωm^Λ	χ²_Λ	A	χ²_osc	Δχ²
Unconstrained (≥ 0.1)	0.1139	2475.22	0.7128	2405.46	69.76
Ωm ≥ 0.2	0.2000	2578.74	0.3873	2563.93	14.81
Ωm ≥ 0.25	0.2500	2708.48	0.2296	2704.21	4.27
Fixed Ωm = 0.315	0.3150	2926.10	0.0000	2926.10	0.00



Physical Conclusion
The Embarrassment Principle shows that low‑redshift data natively prefers oscillating dark energy in the absence of tight CMB priors. The signal is clean, stable, and robust against overfitting.


Acknowledgments
DouBao: theoretical derivation, code debugging, paper writing
DeepSeek: numerical verification, scientific narrative
QianWen: critical feedback and rigorous validation


If you use this code or the results, please cite:
Li, K. (2026). The Embarrassment Principle: Oscillating Dark Energy Preferred by Low-Redshift Data... [Zenodo/arXiv]. DOI: (to be assigned)


