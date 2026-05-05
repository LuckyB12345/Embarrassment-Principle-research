"""
最终完整版：DESI DR2 6点 BAO + Pantheon+ SN + Planck 2018 联合拟合
模型：1+w(z) = A * sin(b * z) / (1+z)^2
BAO 数据来自 DESI DR2 (Abdul‑Karim et al. 2025, arXiv:2503.14738)
ΛCDM 的 Ωm 边界已放宽至 [0.01, 0.50] 以确保公平比较。
"""

import numpy as np
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve, inv
from astropy.cosmology import FLRW, FlatLambdaCDM
from astropy import units as u
from scipy.integrate import quad
import warnings
warnings.filterwarnings("ignore")

# ========== 常量 ==========
C_LIGHT = 299792.458      # km/s
RD_FID = 147.09           # Mpc

# ========== 数据加载（SN + DESI DR2 BAO）==========
def load_data():
    # 加载 Pantheon+ SN 数据
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

    # DESI DR2 BAO 数据（6个点）
    bao = {
        0.20: (3.861, 0.045),   # BGS
        0.51: (13.767, 0.061),  # LRG1
        0.70: (18.651, 0.071),  # LRG2
        0.93: (22.616, 0.068),  # LRG3+ELG1
        1.32: (27.660, 0.110),  # ELG2
        1.48: (29.550, 0.150),  # QSO
    }
    return z, mu, cov, bao

# ========== 振荡暗能量宇宙学模型（基于 astropy）==========
class OscillatingDE(FLRW):
    """
    w(z) = -1 + A * sin(b * z) / (1+z)^2
    """
    def __init__(self, H0, Om0, A, b, name="OscillatingDE"):
        self._A = A
        self._b = b
        super().__init__(H0=H0*u.km/u.s/u.Mpc, Om0=Om0, Ode0=1-Om0,
                         Tcmb0=2.7255*u.K, Neff=3.046, name=name)

    def w(self, z):
        return -1.0 + self._A * np.sin(self._b * z) / (1+z)**2

    def de_density_scale(self, z):
        def integrand(zp):
            return 3 * (self.w(zp) + 1) / (1+zp)
        val = quad(integrand, 0, z, epsrel=1e-7, limit=500)[0]
        val = np.clip(val, -50, 50)
        return np.exp(val)

# ========== 辅助函数 ==========
def mu_cosmo(cosmo, z):
    """距离模量"""
    d_L = cosmo.luminosity_distance(z).value   # Mpc
    return 5 * np.log10(d_L) + 25

def bao_D_V(cosmo, z):
    """D_V / r_d"""
    d_c = cosmo.comoving_distance(z).value     # Mpc
    H_z = cosmo.H(z).value                     # km/s/Mpc
    term = (d_c**2) * (C_LIGHT * z / H_z)
    D_V = term ** (1/3)
    return D_V / RD_FID

# ========== Planck 压缩似然 ==========
def planck2018_chi2(Om, H0):
    Om_p, H0_p = 0.3111, 67.66
    sig_Om, sig_H0 = 0.0056, 0.42
    corr = 0.18
    chi2 = ((Om - Om_p)/sig_Om)**2 + ((H0 - H0_p)/sig_H0)**2 \
           - 2*corr*(Om-Om_p)*(H0-H0_p)/(sig_Om*sig_H0)
    return chi2

# ========== 总 χ²（振荡模型）==========
def chi2_total_osc(params, z_sn, mu_sn, cov_sn, bao_data):
    A, b, Om, H0 = params
    # 物理约束：z=0.2 处必须加速膨胀（1+w < 2/3）
    w0p2 = -1.0 + A * np.sin(b * 0.2) / (1+0.2)**2
    if w0p2 >= -1/3:
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
    for z_bao, (obs_val, obs_err) in bao_data.items():
        pred = bao_D_V(cosmo, z_bao)
        chi2_bao += ((pred - obs_val) / obs_err)**2

    # Planck
    chi2_planck = planck2018_chi2(Om, H0)

    return chi2_sn + chi2_bao + chi2_planck

# ========== ΛCDM χ² ==========
def chi2_lcd(params, z_sn, mu_sn, cov_sn, bao_data):
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
    chi2_bao = 0.0
    for z_bao, (obs_val, obs_err) in bao_data.items():
        pred = bao_D_V(cosmo, z_bao)
        chi2_bao += ((pred - obs_val) / obs_err)**2
    chi2_planck = planck2018_chi2(Om, H0)
    return chi2_sn + chi2_bao + chi2_planck

# ========== Fisher 矩阵辅助 ==========
def num_hessian(f, x, eps=1e-4):
    n = len(x)
    H = np.zeros((n, n))
    f0 = f(x)
    for i in range(n):
        xp = x.copy(); xp[i] += eps
        xm = x.copy(); xm[i] -= eps
        H[i,i] = (f(xp) - 2*f0 + f(xm)) / (eps**2)
        for j in range(i+1, n):
            xpp = x.copy(); xpp[i] += eps; xpp[j] += eps
            xpm = x.copy(); xpm[i] += eps; xpm[j] -= eps
            xmp = x.copy(); xmp[i] -= eps; xmp[j] += eps
            xmm = x.copy(); xmm[i] -= eps; xmm[j] -= eps
            H[i,j] = (f(xpp) - f(xpm) - f(xmp) + f(xmm)) / (4 * eps**2)
            H[j,i] = H[i,j]
    return H

# ========== 主程序 ==========
if __name__ == "__main__":
    print("="*70)
    print("DESI DR2 真实 BAO (6点) + Pantheon+ SN + Planck 2018 联合拟合")
    print("振荡暗能量模型: 1+w(z) = A sin(bz)/(1+z)^2")
    print("ΛCDM Ωm 边界已放宽至 [0.01, 0.50]")
    print("="*70)

    z_sn, mu_sn, cov_sn, bao_data = load_data()
    print(f"SN 数据点: {len(z_sn)}")
    print(f"BAO 数据点 (DESI DR2): {sorted(bao_data.keys())}")

    # ---------- 拟合 ΛCDM ----------
    print("\n>>> 拟合 ΛCDM ...")
    lcd_res = minimize(lambda p: chi2_lcd(p, z_sn, mu_sn, cov_sn, bao_data),
                       [0.3, 69.0], bounds=[(0.01, 0.50), (65, 78)],
                       method='L-BFGS-B', options={'maxiter':20000})
    chi2_lcd = lcd_res.fun
    Om_lcd, H0_lcd = lcd_res.x

    # ---------- 拟合振荡模型 ----------
    print("\n>>> 拟合振荡模型 ...")
    # 初始值可使用之前 MCMC 中位数，也可以使用宽松初值
    init_osc = [1.5377, 26.4174, 0.1252, 69.9076]   # MCMC 中位数
    bounds_osc = [(0, 10.0), (10, 50), (0.01, 0.50), (65, 78)]
    osc_res = minimize(lambda p: chi2_total_osc(p, z_sn, mu_sn, cov_sn, bao_data),
                       init_osc, bounds=bounds_osc, method='L-BFGS-B',
                       options={'maxiter':50000})
    A_best, b_best, Om_best, H0_best = osc_res.x
    chi2_osc = osc_res.fun
    delta_chi2 = chi2_lcd - chi2_osc

    print("\n" + "="*50)
    print("ΛCDM 最佳拟合 (Ωm 自由):")
    print(f"  Ωm = {Om_lcd:.4f}, H0 = {H0_lcd:.2f}, χ² = {chi2_lcd:.2f}")
    print("振荡模型最佳拟合 (尴尬模型):")
    print(f"  A = {A_best:.4f}, b = {b_best:.2f}, Ωm = {Om_best:.4f}, H0 = {H0_best:.2f}, χ² = {chi2_osc:.2f}")
    print(f"Δχ² = {delta_chi2:.2f} (振荡模型 vs ΛCDM)")

    # ---------- Fisher 矩阵 ----------
    def chi2_wrap_osc(params):
        return chi2_total_osc(params, z_sn, mu_sn, cov_sn, bao_data)
    best = np.array([A_best, b_best, Om_best, H0_best])
    print("\n>>> 计算 Hessian 矩阵 (Fisher) ...")
    H_fish = num_hessian(chi2_wrap_osc, best, eps=1e-4)
    cov_fish = inv(H_fish)
    err_fish = np.sqrt(np.diag(cov_fish))
    names = ['A', 'b', 'Ωm', 'H0']
    print("\nFisher 1σ 误差:")
    for n, e in zip(names, err_fish):
        print(f"  {n:5s} = ±{e:.4f}")
    corr = cov_fish / np.outer(err_fish, err_fish)
    print("\n相关系数矩阵:")
    print(corr)

    # ---------- 物理检验 ----------
    z_check = [0.2, 0.4, 0.6, 0.8, 1.0]
    print("\n📌 1+w(z) 物理检验 (主拟合):")
    for zi in z_check:
        wp = A_best * np.sin(b_best * zi) / (1+zi)**2
        print(f"z={zi:.1f} | 1+w = {wp:.4f} (w = {wp-1:.4f})")

    print("\n所有分析完成。")