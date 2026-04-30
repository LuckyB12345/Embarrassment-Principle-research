# redshift_cut_detailedv1.2.0.py
# 红移截断测试：固定 α = -12.00（全局最佳拟合值）
# 目的：检验在最终最佳模型下，振荡信号是否仍然集中在低红移（z<0.1）
# 方法：对 SN 红移进行下限截断，每个子集固定 α=-12，重新拟合 A, β, Ωm, H0，
#       并与 ΛCDM 拟合对比，计算 Δχ²。
# 注：固定 α 避免了小样本下 α 拟合不稳定，且直接检验最佳模型的红移分布。
#     若需自由拟合 α，可将下方 alpha_fixed 取消固定并放开边界。
#
# 运行：python redshift_cut_detailedv1.2.0.py
# 依赖：pantheon+_data.txt, Pantheon+SH0ES_STAT+SYS.cov
# 输出：表格 (z_min, N_SN, χ²_Λ, χ²_osc, Δχ², 以及两种模型的参数)

import numpy as np
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve
from scipy.integrate import quad
import time
import warnings
warnings.filterwarnings("ignore")

C_LIGHT = 299792.458
RD_FID = 147.09
INTEGRAL_EPSREL = 1e-7
INTEGRAL_LIMIT = 500

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
    cov = np.asarray(cov, dtype=float)
    cov += np.eye(cov.shape[0]) * 1e-8

    bao = {0.50: (18.65, 0.25), 0.70: (24.30, 0.30), 1.00: (31.80, 0.45)}
    return z, mu, cov, bao

def E_lcdm(z, Om):
    return np.sqrt(Om*(1+z)**3 + (1-Om))

def mu_lcdm(z, Om, H0):
    try:
        dc, _ = quad(lambda x: C_LIGHT / E_lcdm(x, Om), 0, z,
                     limit=INTEGRAL_LIMIT, epsrel=INTEGRAL_EPSREL)
    except:
        return 1e10
    dc /= H0
    return 5*np.log10(dc * (1+z)) + 25

def DM_H_z_lcdm(z, Om, H0):
    try:
        dc, _ = quad(lambda x: C_LIGHT / E_lcdm(x, Om), 0, z,
                     limit=INTEGRAL_LIMIT, epsrel=INTEGRAL_EPSREL)
    except:
        return 1e10, 1e10
    DM = dc / H0
    Hz = H0 * E_lcdm(z, Om)
    return DM, Hz

def rho_osc_at_z(z, A, a, b):
    try:
        val, _ = quad(lambda x: 3*A*np.exp(-a*x)*np.sin(b*x)/(1+x), 0, z,
                      limit=INTEGRAL_LIMIT, epsrel=INTEGRAL_EPSREL)
    except:
        val = -np.inf
    if val < -300:
        val = -300
    return np.exp(val)

def mu_osc_fixed_alpha(z, A, b, Om, H0, alpha_fixed):
    def integrand(x):
        rho_x = rho_osc_at_z(x, A, alpha_fixed, b)
        if rho_x < 0:
            rho_x = 0.0
        return C_LIGHT / np.sqrt(Om*(1+x)**3 + (1-Om)*rho_x)
    try:
        dc, _ = quad(integrand, 0, z, limit=INTEGRAL_LIMIT, epsrel=INTEGRAL_EPSREL)
    except:
        return 1e10
    return 5*np.log10((dc / H0) * (1+z)) + 25

def DM_H_z_osc_fixed_alpha(z, A, b, Om, H0, alpha_fixed):
    def integrand(x):
        rho_x = rho_osc_at_z(x, A, alpha_fixed, b)
        if rho_x < 0:
            rho_x = 0.0
        return C_LIGHT / np.sqrt(Om*(1+x)**3 + (1-Om)*rho_x)
    try:
        dc, _ = quad(integrand, 0, z, limit=INTEGRAL_LIMIT, epsrel=INTEGRAL_EPSREL)
    except:
        return 1e10, 1e10
    DM = dc / H0
    rho_z = rho_osc_at_z(z, A, alpha_fixed, b)
    if rho_z < 0:
        rho_z = 0.0
    Hz = H0 * np.sqrt(Om*(1+z)**3 + (1-Om)*rho_z)
    return DM, Hz

def chi2_total_lcdm(theta, z, mu, cov, bao):
    Om, H0 = theta
    mu_model = np.array([mu_lcdm(zi, Om, H0) for zi in z])
    if not np.isfinite(mu_model).all() or np.any(np.abs(mu_model) > 100):
        return 1e10
    resid = mu - mu_model
    try:
        cho = cho_factor(cov)
        chi2_sn = resid @ cho_solve(cho, resid)
    except:
        return 1e10
    chi2_bao = 0.0
    for z_bao, (obs_val, obs_err) in bao.items():
        D_M, H_z = DM_H_z_lcdm(z_bao, Om, H0)
        if D_M > 1e9 or H_z < 1e-10 or not np.isfinite(D_M) or not np.isfinite(H_z):
            return 1e10
        term = (D_M**2) * (C_LIGHT * z_bao / H_z)
        if term <= 0:
            return 1e10
        D_V = term ** (1/3)
        model_val = D_V / RD_FID
        chi2_bao += ((model_val - obs_val) / obs_err) ** 2
    return chi2_sn + chi2_bao

def chi2_total_osc_fixed_alpha(theta, z, mu, cov, bao, alpha_fixed):
    A, b, Om, H0 = theta
    if not (0 <= A <= 1.2 and 0 <= b <= 50 and 0.01 <= Om <= 0.5):
        return 1e10
    mu_model = np.array([mu_osc_fixed_alpha(zi, A, b, Om, H0, alpha_fixed) for zi in z])
    if not np.isfinite(mu_model).all() or np.any(np.abs(mu_model) > 100):
        return 1e10
    resid = mu - mu_model
    try:
        cho = cho_factor(cov)
        chi2_sn = resid @ cho_solve(cho, resid)
    except:
        return 1e10
    chi2_bao = 0.0
    for z_bao, (obs_val, obs_err) in bao.items():
        D_M, H_z = DM_H_z_osc_fixed_alpha(z_bao, A, b, Om, H0, alpha_fixed)
        if D_M > 1e9 or H_z < 1e-10 or not np.isfinite(D_M) or not np.isfinite(H_z):
            return 1e10
        term = (D_M**2) * (C_LIGHT * z_bao / H_z)
        if term <= 0:
            return 1e10
        D_V = term ** (1/3)
        model_val = D_V / RD_FID
        chi2_bao += ((model_val - obs_val) / obs_err) ** 2
    return chi2_sn + chi2_bao

def fit_subset(z_sub, mu_sub, cov_sub, bao):
    # ΛCDM
    res_lcdm = minimize(lambda p: chi2_total_lcdm(p, z_sub, mu_sub, cov_sub, bao),
                        [0.3, 70.0],
                        bounds=[(0.01, 0.5), (60, 85)],
                        method='L-BFGS-B', options={'maxiter': 5000, 'disp': False})
    chi2_l = res_lcdm.fun
    Om_l, H0_l = res_lcdm.x
    # 振荡模型固定α=-12
    alpha_fixed = -12.0
    res_osc = minimize(lambda p: chi2_total_osc_fixed_alpha(p, z_sub, mu_sub, cov_sub, bao, alpha_fixed),
                       [0.0046, 17.82, 0.05, 75.0],
                       bounds=[(0, 1.2), (10, 50), (0.01, 0.5), (60, 85)],
                       method='L-BFGS-B', options={'maxiter': 5000, 'disp': False})
    chi2_o = res_osc.fun
    A_o, b_o, Om_o, H0_o = res_osc.x
    delta = chi2_l - chi2_o
    return chi2_l, chi2_o, delta, Om_l, H0_l, A_o, b_o, Om_o, H0_o

def main():
    z_full, mu_full, cov_full, bao = load_data()
    z_min_list = [0.00, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.20]
    print("="*80)
    print("红移截断测试 (固定 α = -12.00)")
    print("SN 红移下限 z_min 从 0.00 到 0.20，BAO 数据不变")
    print("="*80)
    header = (f"{'z_min':<6} {'N_SN':<6} {'χ²_Λ':<10} {'χ²_osc':<10} {'Δχ²':<10} "
              f"{'Ωm_Λ':<8} {'H0_Λ':<8} {'A':<8} {'β':<8} {'Ωm_osc':<8} {'H0_osc':<8}")
    print(header)
    for z_min in z_min_list:
        mask = z_full >= z_min
        z_sub = z_full[mask]
        mu_sub = mu_full[mask]
        idx = np.where(mask)[0]
        cov_sub = cov_full[np.ix_(idx, idx)]
        N = len(z_sub)
        chi2_l, chi2_o, delta, Om_l, H0_l, A_o, b_o, Om_o, H0_o = fit_subset(z_sub, mu_sub, cov_sub, bao)
        print(f"{z_min:<6.2f} {N:<6} {chi2_l:<10.2f} {chi2_o:<10.2f} {delta:<10.2f} "
              f"{Om_l:<8.4f} {H0_l:<8.2f} {A_o:<8.4f} {b_o:<8.2f} {Om_o:<8.4f} {H0_o:<8.2f}")

if __name__ == "__main__":
    main()