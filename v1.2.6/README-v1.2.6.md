由于发现V1.2.5的BAO数据全是错的，故更新V1.2.6

最后，走了这么久的一路，发现标准宇宙学下的各个数据之间已经一团乱麻，我很无奈地拿我的震荡模型拟合。

Embarrassment Principle: v1.2.6 — Globally Consistent Oscillating Dark Energy
This repository contains the code and data used to produce the results in the paper“Globally Consistent Oscillating Dark Energy: A Phenomenological Projection of the Embarrassment Principle” (Li, 2026).
Version v1.2.6 is the final, corrected version based on official DESI DR2 BAO data (DE.2.5SI, 6-point galaxy/quasar sample).All previous BAO data from unverified standard test sets have been removed and replaced.

1. Data Sources (public, cited, official)

Pantheon+ SN Ia (Scolnic et al. 2022) – 1701 supernovae with full covariance.
DESI DR2 BAO (Abdul-Karim et al., arXiv:2503.14738) – 6-point \(D_V/r_d\) at\(z_{\mathrm{eff}} = 0.20, 0.51, 0.70, 0.93, 1.32, 1.48\).
Planck 2018 compressed likelihood (Planck Collaboration 2020) – \((\Omega_m, H_0)\) with\(\Omega_m=0.3111\pm0.0056\), \(H_0=67.66\pm0.42\), \(r=0.18\).

No artificial priors.

2. Model
Oscillating dark energy EoS:\(1+w(z) = \frac{A\sin(bz)}{(1+z)^2}\)

Damped at high redshift (\(1+w\to0\)), consistent with CMB/BBN.
Late-time oscillations from dynamical dark energy.

Physical constraint:Late-time acceleration enforced: \(1+w(z) < 2/3\) at \(z=0.05,0.1,0.15,0.2\).Unphysical solutions automatically rejected.

3. Key Results (v1.2.6 final)
Joint fit: DESI DR2 6-point BAO + Pantheon+ + Planck 2018
表格Model\(\Omega_m\)\(H_0\)Ab\(\chi^2\)\(\Lambda\)CDM0.132272.95——5917.26Oscillating DE\(0.1252\pm0.0023\)\(69.91\pm0.22\)\(1.5377\pm0.0961\)\(26.4174\pm0.2223\)5669.69
\(\boldsymbol{\Delta\chi^2 = 247.57}\) (\(>15\sigma\))
Fisher 1σ errors

\(A = 1.5353 \pm 0.0692\)
\(b = 26.42 \pm 0.1560\)
\(\Omega_m = 0.1254 \pm 0.0016\)
\(H_0 = 69.91 \pm 0.1592\)

MCMC: unimodal, Gaussian, no secondary modes.

4. Robustness Checks
All subsets confirm strong preference for oscillating DE.
表格Data combination\(\Delta\chi^2\)SN+BAO+Planck (full)247.57SN+Planck (no BAO)101.9SN+BAO (no Planck)78.8BAO+Planck (no SN)216.0

5. Scripts

final_fit_physical.py – main fit (optimization + Fisher)
mcmc_desi6.py – MCMC for DESI DR2 6-point BAO
fit_no_planck.py, fit_no_bao.py, fit_no_sn.py – robustness tests
plot_figures.py – paper figures
multi_start_physical.py – multi-start validation


6. How to Runbash运行python final_fit_physical.py
python mcmc_desi6.py


7. Citation
Li, K. (2026). Globally Consistent Oscillating Dark Energy: A Phenomenological Projection of the Embarrassment Principle.Li, K. (2026). Embarrassment-Principle-research (v1.2.6). Zenodo. https://doi.org/10.5281/zenodo.19828321

8. License
MIT License © 2026 Li Kaibing

9. Contact
Kaibing Li – 806255397@qq.com