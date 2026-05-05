"""
联合拟合：只使用 SN + BAO（DESI 6点），无 Planck 先验
模型：1+w(z) = A sin(bz)/(1+z)^2
"""

import numpy as np
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve
from astropy.cosmology import FLRW
from astropy import units as u
from scipy.integrate import quad
import warnings
warnings.filterwarnings("ignore")

C_LIGHT = 299792.458
RD_FID = 147.09

def load_data():
    # SN
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
    # DESI 6点
    bao = {
        0.20: (3.861, 0.045),
        0.51: (13.767, 0.061),
        0.70: (18.651, 0.071),
        0.93: (22.616, 0.068),
        1.32: (27.660, 0.110),
        1.48: (29.550, 0.150),
    }
    return z, mu, cov, bao

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

def bao_D_V(cosmo, z):
    d_c = cosmo.comoving_distance(z).value
    H_z = cosmo.H(z).value
    term = (d_c**2) * (C_LIGHT * z / H_z)
    D_V = term ** (1/3)
    return D_V / RD_FID

def chi2_total(params, z_sn, mu_sn, cov_sn, bao_data):
    A, b, Om, H0 = params
    if -1.0 + A * np.sin(b*0.2)/(1.2)**2 >= -1/3:
        return 1e10
    if not (0 <= A <= 10 and 10 <= b <= 50 and 0.01 <= Om <= 0.50 and 65 <= H0 <= 78):
        return 1e10
    cosmo = OscillatingDE(H0=H0, Om0=Om, A=A, b=b)
    # SN
    mu_model = np.array([mu_cosmo(cosmo, zi) for zi in z_sn])
    if not np.all(np.isfinite(mu_model)):
        return 1e10
    resid = mu_sn - mu_model
    try:
        cho = cho_factor(cov_sn)
        chi2_sn = resid @ cho_solve(cho, resid)
    except:
        return 1e10
    # BAO
    chi2_bao = 0.0
    for z_bao, (obs, err) in bao_data.items():
        pred = bao_D_V(cosmo, z_bao)
        chi2_bao += ((pred - obs) / err)**2
    # 无 Planck 项
    return chi2_sn + chi2_bao

# ΛCDM 版本
def chi2_lcd(params, z_sn, mu_sn, cov_sn, bao_data):
    Om, H0 = params
    if not (0.01 <= Om <= 0.50 and 65 <= H0 <= 78):
        return 1e10
    from astropy.cosmology import FlatLambdaCDM
    cosmo = FlatLambdaCDM(H0=H0, Om0=Om)
    mu_model = np.array([mu_cosmo(cosmo, zi) for zi in z_sn])
    resid = mu_sn - mu_model
    try:
        cho = cho_factor(cov_sn)
        chi2_sn = resid @ cho_solve(cho, resid)
    except:
        return 1e10
    chi2_bao = 0.0
    for z_bao, (obs, err) in bao_data.items():
        pred = bao_D_V(cosmo, z_bao)
        chi2_bao += ((pred - obs) / err)**2
    return chi2_sn + chi2_bao

if __name__ == "__main__":
    print("="*70)
    print("无 Planck 先验：只用 SN + BAO (DESI 6点)")
    print("="*70)
    z_sn, mu_sn, cov_sn, bao_data = load_data()
    print(f"SN: {len(z_sn)} points, BAO: {len(bao_data)} points")

    print("\n>>> 拟合 ΛCDM ...")
    res_lcd = minimize(lambda p: chi2_lcd(p, z_sn, mu_sn, cov_sn, bao_data),
                       [0.31, 67.7], bounds=[(0.01,0.50),(65,78)],
                       method='L-BFGS-B')
    chi2_lcd = res_lcd.fun
    Om_l, H0_l = res_lcd.x
    print(f"ΛCDM: Ωm={Om_l:.4f}, H0={H0_l:.2f}, χ²={chi2_lcd:.2f}")

    print("\n>>> 拟合振荡模型 ...")
    init = [1.5, 26.0, 0.15, 70.0]
    bounds = [(0,10),(10,50),(0.01,0.50),(65,78)]
    res_osc = minimize(lambda p: chi2_total(p, z_sn, mu_sn, cov_sn, bao_data),
                       init, bounds=bounds, method='L-BFGS-B', options={'maxiter':50000})
    A, b, Om, H0 = res_osc.x
    chi2_osc = res_osc.fun
    print(f"振荡模型: A={A:.4f}, b={b:.2f}, Ωm={Om:.4f}, H0={H0:.2f}, χ²={chi2_osc:.2f}")
    print(f"Δχ² = {chi2_lcd - chi2_osc:.2f} (振荡 vs ΛCDM)")