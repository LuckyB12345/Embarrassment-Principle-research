"""
联合拟合：只使用 SN + Planck 先验，无 BAO
模型：1+w(z) = A sin(bz)/(1+z)^2
"""

import numpy as np
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve
from astropy.cosmology import FLRW, FlatLambdaCDM
from astropy import units as u
from scipy.integrate import quad
import warnings
warnings.filterwarnings("ignore")

C_LIGHT = 299792.458
RD_FID = 147.09

def load_data():
    data = np.genfromtxt("pantheon+_data.txt", skip_header=1, usecols=(4,10,11))
    z, mu, err = data.T
    msk = (z > 0) & np.isfinite(mu)
    z, mu = z[msk], mu[msk]
    with open("Pantheon+SH0ES_STAT+SYS.cov", 'r') as f:
        n = int(f.readline())
    cov = np.loadtxt("Pantheon+SH0ES_STAT+SYS.cov", skiprows=1).reshape(n, n)
    cov = cov[np.ix_(msk, msk)]
    cov = (cov + cov.T) / 2
    cov += np.eye(cov.shape[0]) * 1e-8
    return z, mu, cov

def planck2018_chi2(Om, H0):
    Om_p, H0_p = 0.3111, 67.66
    s_Om, s_H0 = 0.0056, 0.42
    corr = 0.18
    return ((Om-Om_p)/s_Om)**2 + ((H0-H0_p)/s_H0)**2 - 2*corr*(Om-Om_p)*(H0-H0_p)/(s_Om*s_H0)

class OscillatingDE(FLRW):
    def __init__(self, H0, Om0, A, b, name="OscillatingDE"):
        self._A = A
        self._b = b
        super().__init__(H0=H0*u.km/u.s/u.Mpc, Om0=Om0, Ode0=1-Om0,
                         Tcmb0=2.7255*u.K, Neff=3.046, name=name)
    def w(self, z):
        return -1.0 + self._A * np.sin(self._b * z) / (1+z)**2

def mu_cosmo(cosmo, z):
    d_L = cosmo.luminosity_distance(z).value
    return 5 * np.log10(d_L) + 25

def chi2_total(params, z_sn, mu_sn, cov_sn):
    A, b, Om, H0 = params
    if -1.0 + A * np.sin(b*0.2)/(1.2)**2 >= -1/3:
        return 1e10
    if not (0 <= A <= 10 and 10 <= b <= 50 and 0.01 <= Om <= 0.50 and 65 <= H0 <= 78):
        return 1e10
    cosmo = OscillatingDE(H0=H0, Om0=Om, A=A, b=b)
    mu_model = np.array([mu_cosmo(cosmo, zi) for zi in z_sn])
    if not np.all(np.isfinite(mu_model)):
        return 1e10
    resid = mu_sn - mu_model
    try:
        cho = cho_factor(cov_sn)
        chi2_sn = resid @ cho_solve(cho, resid)
    except:
        return 1e10
    chi2_planck = planck2018_chi2(Om, H0)
    return chi2_sn + chi2_planck

def chi2_lcd(params, z_sn, mu_sn, cov_sn):
    Om, H0 = params
    if not (0.01 <= Om <= 0.50 and 65 <= H0 <= 78):
        return 1e10
    cosmo = FlatLambdaCDM(H0=H0, Om0=Om)
    mu_model = np.array([mu_cosmo(cosmo, zi) for zi in z_sn])
    resid = mu_sn - mu_model
    try:
        cho = cho_factor(cov_sn)
        chi2_sn = resid @ cho_solve(cho, resid)
    except:
        return 1e10
    chi2_planck = planck2018_chi2(Om, H0)
    return chi2_sn + chi2_planck

if __name__ == "__main__":
    print("="*70)
    print("无 BAO：只用 SN + Planck")
    print("="*70)
    z_sn, mu_sn, cov_sn = load_data()
    print(f"SN 点数: {len(z_sn)}")

    print("\n>>> 拟合 ΛCDM ...")
    res_lcd = minimize(lambda p: chi2_lcd(p, z_sn, mu_sn, cov_sn),
                       [0.31, 67.7], bounds=[(0.01,0.50),(65,78)],
                       method='L-BFGS-B')
    chi2_lcd = res_lcd.fun
    Om_l, H0_l = res_lcd.x
    print(f"ΛCDM: Ωm={Om_l:.4f}, H0={H0_l:.2f}, χ²={chi2_lcd:.2f}")

    print("\n>>> 拟合振荡模型 ...")
    init = [1.5, 26.0, 0.15, 70.0]
    bounds = [(0,10),(10,50),(0.01,0.50),(65,78)]
    res_osc = minimize(lambda p: chi2_total(p, z_sn, mu_sn, cov_sn),
                       init, bounds=bounds, method='L-BFGS-B', options={'maxiter':50000})
    A, b, Om, H0 = res_osc.x
    chi2_osc = res_osc.fun
    print(f"振荡模型: A={A:.4f}, b={b:.2f}, Ωm={Om:.4f}, H0={H0:.2f}, χ²={chi2_osc:.2f}")
    print(f"Δχ² = {chi2_lcd - chi2_osc:.2f} (振荡 vs ΛCDM)")