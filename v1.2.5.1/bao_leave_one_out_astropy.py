"""
留一法检验（基于 astropy）
数据集: SN + Planck + (BAO 去掉一个点)
模型: 振荡暗能量 1+w(z)=A sin(bz)/(1+z)^2
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
    bao_full = {0.50: (18.65, 0.25), 0.70: (24.30, 0.30), 1.00: (31.80, 0.45)}
    return z, mu, cov, bao_full

# ========== 振荡模型 ==========
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

def planck2018_chi2(Om, H0):
    Om_p, H0_p = 0.3111, 67.66
    s_Om, s_H0 = 0.0056, 0.42
    corr = 0.18
    return ((Om-Om_p)/s_Om)**2 + ((H0-H0_p)/s_H0)**2 - 2*corr*(Om-Om_p)*(H0-H0_p)/(s_Om*s_H0)

def chi2_total(params, z_sn, mu_sn, cov_sn, bao):
    A, b, Om, H0 = params
    # 物理约束
    if -1.0 + A * np.sin(b*0.2)/(1.2)**2 >= -1/3:
        return 1e10
    if not (0 <= A <= 10 and 10 <= b <= 50 and 0.15 <= Om <= 0.40 and 65 <= H0 <= 78):
        return 1e10
    cosmo = OscillatingDE(H0=H0, Om0=Om, A=A, b=b)
    mu_model = np.array([mu_cosmo(cosmo, zi) for zi in z_sn])
    if not np.isfinite(mu_model).all():
        return 1e10
    resid = mu_sn - mu_model
    try:
        cho = cho_factor(cov_sn)
        chi2_sn = resid @ cho_solve(cho, resid)
    except:
        return 1e10
    chi2_bao = 0.0
    for z_bao, (obs, err) in bao.items():
        pred = bao_D_V(cosmo, z_bao)
        chi2_bao += ((pred - obs) / err)**2
    chi2_planck = planck2018_chi2(Om, H0)
    return chi2_sn + chi2_bao + chi2_planck

def leave_one_out():
    z_sn, mu_sn, cov_sn, bao_full = load_data()
    z_bao_list = sorted(bao_full.keys())
    results = {}
    for leave_z in z_bao_list:
        print(f"\n=== 去掉 BAO 点 z = {leave_z} ===")
        bao_rest = {z: bao_full[z] for z in z_bao_list if z != leave_z}
        init = [1.4082, 26.60, 0.2691, 70.16]   # 来自 astropy 最佳拟合
        bounds = [(0,10), (10,50), (0.15,0.40), (65,78)]
        res = minimize(lambda p: chi2_total(p, z_sn, mu_sn, cov_sn, bao_rest),
                       init, bounds=bounds, method='L-BFGS-B', options={'maxiter':50000})
        if not res.success:
            print("  优化失败")
            continue
        A, b, Om, H0 = res.x
        chi2 = res.fun
        print(f"  拟合结果: A={A:.4f}, b={b:.2f}, Om={Om:.4f}, H0={H0:.2f}, χ²={chi2:.2f}")
        cosmo = OscillatingDE(H0=H0, Om0=Om, A=A, b=b)
        pred = bao_D_V(cosmo, leave_z)
        obs, err = bao_full[leave_z]
        diff = pred - obs
        sigma = diff / err
        print(f"  预言 D_V/r_d = {pred:.3f}, 观测 = {obs:.3f} ± {err:.3f}, 偏差 = {diff:.3f} ({sigma:.2f}σ)")
        results[leave_z] = (pred, obs, err, sigma)
    return results

if __name__ == "__main__":
    print("="*60)
    print("留一法（Astropy 正确版）")
    print("="*60)
    results = leave_one_out()
    print("\n" + "="*50)
    print("留一法总结:")
    for z, (pred, obs, err, sigma) in results.items():
        print(f"z={z}: 预言 {pred:.3f} vs 观测 {obs:.3f}±{err:.3f} → {sigma:.2f}σ")